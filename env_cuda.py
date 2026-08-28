import math

import torch
import numpy as np
import quadsim_cuda 

class DiffDriveDynamics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p_2d, v, theta, omega, dt, grad_decay):
        p_2d = p_2d.contiguous()
        v = v.contiguous()
        theta = theta.contiguous()
        omega = omega.contiguous()
        
        ctx.save_for_backward(v, theta)
        ctx.dt = dt
        ctx.grad_decay = grad_decay
        
        results = quadsim_cuda.run_forward(p_2d, v, theta, omega, dt)
        return results[0], results[1], results[2], results[3]

    @staticmethod
    def backward(ctx, d_p_next, d_v_next, d_theta_next, d_omega_next):
        v, theta = ctx.saved_tensors
        dt = ctx.dt
        grad_decay = ctx.grad_decay
        
        grads = quadsim_cuda.run_backward(
            v.contiguous(),      
            theta.contiguous(), 
            d_p_next.contiguous(), 
            d_v_next.contiguous(), 
            d_theta_next.contiguous(), 
            grad_decay, dt
        )
        
        d_p, d_v, d_theta, d_omega = grads[0], grads[1], grads[2], grads[3]
        if d_omega_next is not None:
            d_omega += d_omega_next

        return d_p, d_v, d_theta, d_omega, None, None

class NearestPointFunction(torch.autograd.Function):
    """Differentiable nearest-obstacle-point query.

    Wraps ``quadsim_cuda.find_nearest_pt`` / ``find_nearest_pt_backward`` so
    that obstacle-avoidance losses receive exact analytic gradients with
    respect to the robot position, instead of the previous approximation that
    froze the nearest point and only propagated gradients through ``-p``.
    """

    @staticmethod
    def forward(ctx, pos, balls, cylinders, cylinders_h, voxels,
                drone_radius, n_drones_per_group):
        pos_c = pos.detach().contiguous()
        nearest_pt = pos_c.new_zeros(pos_c.shape)
        quadsim_cuda.find_nearest_pt(
            nearest_pt.unsqueeze(0),
            balls, cylinders, cylinders_h, voxels,
            pos_c.unsqueeze(0),
            drone_radius, n_drones_per_group
        )
        ctx.save_for_backward(
            balls, cylinders, cylinders_h, voxels, pos_c
        )
        ctx.drone_radius = drone_radius
        ctx.n_drones_per_group = n_drones_per_group
        return nearest_pt

    @staticmethod
    def backward(ctx, d_nearest_pt):
        balls, cylinders, cylinders_h, voxels, pos_c = ctx.saved_tensors
        d_pos = torch.zeros_like(pos_c).unsqueeze(0)
        quadsim_cuda.find_nearest_pt_backward(
            d_pos,
            d_nearest_pt.detach().contiguous().unsqueeze(0),
            balls, cylinders, cylinders_h, voxels,
            pos_c.unsqueeze(0),
            ctx.drone_radius, ctx.n_drones_per_group
        )
        return d_pos.squeeze(0), None, None, None, None, None, None

class Env:
    def __init__(self, batch_size, width, height, grad_decay, device,
                 fov_x_half_tan=0.82, 
                 single=False, gate=False, ground_voxels=True, scaffold=False, pitch_noise=0.175, # 弧度制，0.26约15度
                 diff_nearest_pt=True,
                 map_size=20.0, num_cyl=25, num_balls=15, num_vox=15,
                 robot_radius=0.15,
                 cyl_radius_min=0.2, cyl_radius_max=0.5,
                 ball_radius_min=0.2, ball_radius_max=0.4,
                 ball_radius_floor=0.0,
                 start_pos=(-4.5, -4.5), target_pos=(4.5, 4.5),
                 randomize_start_goal=False, initial_yaw_noise=0.26,
                 protected_zone_radius=2.0,
                 obstacle_min_surface_gap=0.0,
                 obstacle_resample_attempts=128,
                 obstacle_scene_restarts=8,
                 obstacle_layout='nonoverlap',
                 obstacle_grid_jitter=0.75,
                 obstacle_candidate_multiplier=2.0,
                 dynamic_obstacle_scene_prob=0.0,
                 dynamic_obstacle_ratio=0.3,
                 dynamic_obstacle_speed_min=0.2,
                 dynamic_obstacle_speed_max=1.0,
                 dynamic_obstacle_seed=None):  # 使用 CUDA 解析梯度回传最近障碍物点
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.device = device
        self.grad_decay = grad_decay
        self.fov_x_half_tan = fov_x_half_tan
        self.ground_voxels = ground_voxels 
        self.camera_height = 0.25 
        self.dtype = torch.float32
        self.pitch_noise = pitch_noise 
        
        self.p = torch.zeros((batch_size, 3), device=device)
        self.p_start = torch.zeros((batch_size, 3), device=device)
        self.v = torch.zeros((batch_size, 1), device=device)
        self.theta = torch.zeros((batch_size, 3), device=device)
        self.omega = torch.zeros((batch_size, 3), device=device)
        self.R = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        self.p_target = torch.zeros((batch_size, 3), device=device)

        self.R_cam = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        self.single = single
        self.gate = gate
        self.scaffold = scaffold
        self.diff_nearest_pt = diff_nearest_pt
        
        self.cyl = torch.zeros((self.batch_size, 0, 3), device=self.device)
        self.cyl_h = torch.zeros((self.batch_size, 0, 3), device=self.device)
        self.balls = torch.zeros((self.batch_size, 0, 4), device=self.device) 
        self.voxels = torch.zeros((self.batch_size, 0, 6), device=self.device) 
        self.cyl_velocity = torch.zeros(
            (self.batch_size, 0, 2), device=self.device)
        self.ball_velocity = torch.zeros(
            (self.batch_size, 0, 2), device=self.device)
        self.dynamic_scene_mask = torch.zeros(
            self.batch_size, dtype=torch.bool, device=self.device)
        
        self.canvas = torch.zeros((batch_size, height, width), device=device)
        self.flow = torch.zeros((batch_size, 2, height, width), device=device)
        self.nearest_pt = torch.zeros((batch_size, 3), device=device)
        
        self.p_old = self.p.clone()
        self.R_old = self.R.clone()
        
        self.n_drones_per_group = 1 

        # Keep the historical attribute name because the CUDA kernels and
        # losses already consume ``drone_radius``.  The public constructor
        # parameter is named after the actual differential-drive robot.
        self.drone_radius = float(robot_radius)
        self.map_size = float(map_size)
        self.num_cyl = int(num_cyl)
        self.num_balls = int(num_balls)
        self.num_vox = int(num_vox)
        self.cyl_radius_min = float(cyl_radius_min)
        self.cyl_radius_max = float(cyl_radius_max)
        self.ball_radius_min = float(ball_radius_min)
        self.ball_radius_max = float(ball_radius_max)
        self.ball_radius_floor = float(ball_radius_floor)
        self.start_pos = tuple(start_pos)
        self.target_pos = tuple(target_pos)
        # Corner pairs are fixed per Env, so build them once instead of
        # re-uploading small CPU tensors on every episode reset.
        sx, sy = self.start_pos
        tx, ty = self.target_pos
        self._corner_starts = torch.tensor(
            [[sx, sy], [tx, ty], [sx, ty], [tx, sy]],
            device=self.device, dtype=self.dtype)
        self._corner_targets = torch.tensor(
            [[tx, ty], [sx, sy], [tx, sy], [sx, ty]],
            device=self.device, dtype=self.dtype)
        # Zero columns reused every dynamics step to build omega inputs.
        self._zeros_b1 = torch.zeros(
            (batch_size, 1), device=self.device)
        # The ground plane voxel is constant for the whole run; build it once
        # instead of re-creating and filling tensors on every reset.
        self._ground_voxel = None
        if self.ground_voxels:
            ground = torch.zeros((1, 1, 6), device=self.device)
            ground[:, :, 0:3] = torch.tensor(
                [0.0, 0.0, -0.05], device=self.device)
            ground_extent = self.map_size * 2.0
            ground[:, :, 3:6] = torch.tensor(
                [ground_extent, ground_extent, 0.05], device=self.device)
            self._ground_voxel = ground
        self.randomize_start_goal = bool(randomize_start_goal)
        self.initial_yaw_noise = float(initial_yaw_noise)
        self.protected_zone_radius = float(protected_zone_radius)
        self.obstacle_min_surface_gap = float(obstacle_min_surface_gap)
        self.obstacle_resample_attempts = int(obstacle_resample_attempts)
        self.obstacle_scene_restarts = int(obstacle_scene_restarts)
        self.obstacle_layout = str(obstacle_layout)
        self.obstacle_grid_jitter = float(obstacle_grid_jitter)
        self.obstacle_candidate_multiplier = float(
            obstacle_candidate_multiplier)
        self.dynamic_obstacle_scene_prob = float(
            dynamic_obstacle_scene_prob)
        self.dynamic_obstacle_ratio = float(dynamic_obstacle_ratio)
        self.dynamic_obstacle_speed_min = float(dynamic_obstacle_speed_min)
        self.dynamic_obstacle_speed_max = float(dynamic_obstacle_speed_max)
        if self.map_size <= 0.0:
            raise ValueError('map_size must be positive')
        if self.drone_radius <= 0.0:
            raise ValueError('robot_radius must be positive')
        if self.protected_zone_radius < 0.0:
            raise ValueError('protected_zone_radius must be non-negative')
        if self.obstacle_min_surface_gap < 0.0:
            raise ValueError('obstacle_min_surface_gap must be non-negative')
        if self.obstacle_resample_attempts < 1:
            raise ValueError('obstacle_resample_attempts must be positive')
        if self.obstacle_scene_restarts < 1:
            raise ValueError('obstacle_scene_restarts must be positive')
        if self.obstacle_layout not in ('nonoverlap', 'stratified'):
            raise ValueError(
                "obstacle_layout must be 'nonoverlap' or 'stratified'")
        if not 0.0 <= self.obstacle_grid_jitter <= 1.0:
            raise ValueError('obstacle_grid_jitter must be in [0, 1]')
        if self.obstacle_candidate_multiplier < 1.0:
            raise ValueError(
                'obstacle_candidate_multiplier must be at least 1')
        if min(self.num_cyl, self.num_balls, self.num_vox) < 0:
            raise ValueError('obstacle counts must be non-negative')
        if not 0.0 < self.cyl_radius_min <= self.cyl_radius_max:
            raise ValueError('invalid cylinder radius range')
        if not 0.0 < self.ball_radius_min <= self.ball_radius_max:
            raise ValueError('invalid ball radius range')
        if not 0.0 <= self.ball_radius_floor <= self.ball_radius_max:
            raise ValueError('ball_radius_floor must be in [0, maximum]')
        if not 0.0 <= self.dynamic_obstacle_scene_prob <= 1.0:
            raise ValueError('dynamic_obstacle_scene_prob must be in [0, 1]')
        if not 0.0 <= self.dynamic_obstacle_ratio <= 1.0:
            raise ValueError('dynamic_obstacle_ratio must be in [0, 1]')
        if self.dynamic_obstacle_speed_min < 0.0:
            raise ValueError('dynamic_obstacle_speed_min must be non-negative')
        if self.dynamic_obstacle_speed_max < self.dynamic_obstacle_speed_min:
            raise ValueError(
                'dynamic_obstacle_speed_max must be >= minimum')
        self.has_dynamic_obstacles = (
            self.dynamic_obstacle_scene_prob > 0.0
            and self.dynamic_obstacle_ratio > 0.0
            and self.dynamic_obstacle_speed_max > 0.0
        )
        self.dynamic_generator = None
        if self.has_dynamic_obstacles:
            if dynamic_obstacle_seed is None:
                dynamic_obstacle_seed = torch.initial_seed() + 104729
            self.dynamic_generator = torch.Generator(device=self.device)
            self.dynamic_generator.manual_seed(
                int(dynamic_obstacle_seed) % (2 ** 63 - 1))

    def sample_start_and_target(self):
        """Pick one of four map corners and its opposite corner per episode."""
        choice = torch.randint(
            0, 4, (self.batch_size,), device=self.device)
        return self._corner_starts[choice], self._corner_targets[choice]

    def randomize_obstacles(self):
        if self.single:
            self.cyl = torch.zeros((self.batch_size, 0, 6), device=self.device)
            self.balls = torch.zeros((self.batch_size, 0, 8), device=self.device)
            self.voxels = torch.zeros((self.batch_size, 0, 12), device=self.device)
            if self.has_dynamic_obstacles:
                self._sample_dynamic_obstacle_velocities()
            return

        start_pos = self.p[:, :2]
        end_pos = self.p_target[:, :2]
        n_cyl = self.num_cyl
        cyl_r = torch.rand(
            (self.batch_size, n_cyl, 1), device=self.device
        ) * (self.cyl_radius_max - self.cyl_radius_min) + self.cyl_radius_min
        n_balls = self.num_balls
        ball_r = torch.rand(
            (self.batch_size, n_balls, 1), device=self.device
        ) * (self.ball_radius_max - self.ball_radius_min) + self.ball_radius_min
        if self.ball_radius_floor > 0.0:
            ball_r = ball_r.clamp_min(self.ball_radius_floor)
        n_vox = self.num_vox
        vox_rx = torch.rand(
            (self.batch_size, n_vox, 1), device=self.device
        ) * 0.3 + 0.2
        vox_ry = torch.rand(
            (self.batch_size, n_vox, 1), device=self.device
        ) * 0.3 + 0.2
        # A bounding circle makes the 2 m protected region conservative for
        # every orientation around an axis-aligned rectangular obstacle.
        vox_extent = torch.sqrt(vox_rx.square() + vox_ry.square())

        # Place all primitive types in one pass. If sequential rejection gets
        # trapped for an individual scene, restart that scene's complete layout
        # instead of terminating the full training batch.
        all_extents = torch.cat([cyl_r, ball_r, vox_extent], dim=1)
        num_obstacles = all_extents.shape[1]
        all_xy = torch.empty(
            (self.batch_size, num_obstacles, 2),
            device=self.device,
            dtype=self.dtype,
        )
        half_map = self.map_size * 0.5
        if self.obstacle_layout == 'stratified':
            # Oversample jittered grid cells, discard cells in the start/goal
            # protected regions, then select a random subset. This removes the
            # expensive pairwise rejection loop while avoiding the large empty
            # patches and clusters produced by independent uniform sampling.
            grid_side = max(
                1,
                math.ceil(math.sqrt(
                    num_obstacles * self.obstacle_candidate_multiplier)),
            )
            axis = (
                (torch.arange(
                    grid_side, device=self.device, dtype=self.dtype) + 0.5)
                / grid_side * 2.0 - 1.0
            )
            grid_y, grid_x = torch.meshgrid(axis, axis, indexing='ij')
            normalized_grid = torch.stack(
                [grid_x.flatten(), grid_y.flatten()], dim=-1)
            candidate_count = normalized_grid.shape[0]

            max_extent = all_extents[..., 0].amax(dim=1)
            usable_half = (half_map - max_extent).clamp_min(0.0)
            candidates = (
                normalized_grid[None, :, :]
                * usable_half[:, None, None]
            )
            normalized_cell_width = 2.0 / grid_side
            jitter = (
                torch.rand(
                    (self.batch_size, candidate_count, 2),
                    device=self.device,
                    dtype=self.dtype,
                ) - 0.5
            ) * (
                normalized_cell_width * self.obstacle_grid_jitter
            )
            candidates = candidates + jitter * usable_half[:, None, None]

            protected_distance = self.protected_zone_radius + max_extent
            valid = (
                (candidates - start_pos[:, None, :]).norm(dim=-1)
                >= protected_distance[:, None]
            ) & (
                (candidates - end_pos[:, None, :]).norm(dim=-1)
                >= protected_distance[:, None]
            )
            valid_count = valid.sum(dim=1)
            # Keep the safety check on device. A Python ``if tensor.any()``
            # synchronized all CUDA work once per reset, which is costly for
            # this otherwise GPU-resident training loop.
            torch._assert_async(
                (valid_count >= num_obstacles).all(),
                'Stratified obstacle layout has too few valid cells; '
                'increase obstacle_candidate_multiplier.',
            )

            scores = torch.rand(
                (self.batch_size, candidate_count),
                device=self.device,
                dtype=self.dtype,
            ).masked_fill(~valid, float('inf'))
            selected = scores.topk(
                num_obstacles, dim=1, largest=False).indices
            all_xy = candidates.gather(
                1, selected[..., None].expand(-1, -1, 2))
        else:
            # Draw every rejection-sampling attempt for an obstacle at once
            # and pick the first valid candidate on the GPU.  This keeps the
            # original sequential semantics (obstacle i only checks against
            # obstacles j < i) while removing the per-attempt CPU<->GPU
            # synchronization of the previous inner loop.
            attempts = self.obstacle_resample_attempts
            pending = torch.arange(self.batch_size, device=self.device)
            for _ in range(self.obstacle_scene_restarts):
                if pending.numel() == 0:
                    break
                extents = all_extents[pending, :, 0]
                starts = start_pos[pending]
                ends = end_pos[pending]
                pending_count = pending.numel()
                trial = torch.empty(
                    (pending_count, num_obstacles, 2),
                    device=self.device,
                    dtype=self.dtype,
                )
                scene_valid = torch.ones(
                    pending_count, dtype=torch.bool, device=self.device)

                for obstacle_index in range(num_obstacles):
                    extent = extents[:, obstacle_index]
                    available = (half_map - extent).clamp_min(0.0)
                    candidates = (
                        torch.rand(
                            (pending_count, attempts, 2),
                            device=self.device,
                            dtype=self.dtype,
                        ) * 2.0 - 1.0
                    ) * available[:, None, None]

                    protected_distance = (
                        self.protected_zone_radius + extent)[:, None]
                    conflict = (
                        (candidates - starts[:, None, :]).norm(dim=-1)
                        < protected_distance
                    ) | (
                        (candidates - ends[:, None, :]).norm(dim=-1)
                        < protected_distance
                    )
                    if obstacle_index > 0:
                        required_distance = (
                            extent[:, None, None]
                            + extents[:, None, :obstacle_index]
                            + self.obstacle_min_surface_gap
                        )
                        overlaps_previous = (
                            (
                                candidates[:, :, None, :]
                                - trial[:, None, :obstacle_index, :]
                            ).norm(dim=-1)
                            < required_distance
                        ).any(dim=-1)
                        conflict = conflict | overlaps_previous

                    candidate_valid = ~conflict
                    has_valid = candidate_valid.any(dim=1)
                    # argmax on the int mask returns the first valid attempt;
                    # rows without any valid attempt are masked out below.
                    first_valid = candidate_valid.int().argmax(dim=1)
                    chosen = candidates.gather(
                        1, first_valid[:, None, None].expand(-1, 1, 2)
                    ).squeeze(1)
                    scene_valid = scene_valid & has_valid
                    trial[:, obstacle_index] = chosen

                all_xy[pending[scene_valid]] = trial[scene_valid]
                pending = pending[~scene_valid]

            if pending.numel() > 0:
                raise RuntimeError(
                    'Unable to generate non-overlapping obstacle layouts after '
                    f'{self.obstacle_scene_restarts} complete-scene restarts: '
                    f'{pending.numel()}/{self.batch_size} scenes remain invalid. '
                    'Reduce obstacle count/gap or increase map size.'
                )

        cyl_xy = all_xy[:, :n_cyl]
        ball_xy = all_xy[:, n_cyl:n_cyl + n_balls]
        vox_xy = all_xy[:, n_cyl + n_balls:]

        self.cyl = torch.cat([cyl_xy, cyl_r], dim=-1)
        # Ball centre height varies per instance, sampled uniformly from
        # half-buried (centre at z = 0) to floating 0.2 m above the ground
        # (centre at z = r + 0.2). This diversifies the rendered depth
        # signature; planar clearance ignores ball height, so it does not
        # change the collision/avoidance geometry.
        ball_lift = torch.rand(
            ball_r.shape, device=self.device) * (ball_r + 0.2)
        ball_zr = torch.cat([ball_lift, ball_r], dim=-1)
        self.balls = torch.cat([ball_xy, ball_zr], dim=-1)

        vox_rz = torch.full((self.batch_size, n_vox, 1), 0.5, device=self.device)
        vox_z = vox_rz.clone()
        vox_params = torch.cat([vox_z, vox_rx, vox_ry, vox_rz], dim=-1)
        obstacles_vox = torch.cat([vox_xy, vox_params], dim=-1)

        if self.ground_voxels:
            ground = self._ground_voxel.expand(self.batch_size, -1, -1)
            self.voxels = torch.cat([ground, obstacles_vox], dim=1)
        else:
            self.voxels = obstacles_vox

        self.cyl_h = torch.zeros((self.batch_size, 0, 3), device=self.device)
        if self.has_dynamic_obstacles:
            self._sample_dynamic_obstacle_velocities()

    def _sample_planar_velocity(self, count, scene_mask):
        if count == 0:
            return torch.zeros(
                (self.batch_size, 0, 2), device=self.device)
        moving = (
            torch.rand(
                (self.batch_size, count, 1), device=self.device,
                generator=self.dynamic_generator)
            < self.dynamic_obstacle_ratio
        ) & scene_mask[:, None, None]
        angle = torch.rand(
            (self.batch_size, count), device=self.device,
            generator=self.dynamic_generator) * (2.0 * torch.pi)
        speed = torch.rand(
            (self.batch_size, count), device=self.device,
            generator=self.dynamic_generator)
        speed = speed * (
            self.dynamic_obstacle_speed_max
            - self.dynamic_obstacle_speed_min
        ) + self.dynamic_obstacle_speed_min
        velocity = torch.stack(
            [torch.cos(angle) * speed, torch.sin(angle) * speed], dim=-1)
        return velocity * moving.to(velocity.dtype)

    def _sample_dynamic_obstacle_velocities(self):
        self.dynamic_scene_mask = (
            torch.rand(
                self.batch_size, device=self.device,
                generator=self.dynamic_generator)
            < self.dynamic_obstacle_scene_prob
        )
        self.cyl_velocity = self._sample_planar_velocity(
            self.cyl.shape[1], self.dynamic_scene_mask)
        self.ball_velocity = self._sample_planar_velocity(
            self.balls.shape[1], self.dynamic_scene_mask)

    @staticmethod
    def _reflect_planar(position, velocity, half_extent, radius):
        upper = (half_extent - radius).clamp_min(0.0)
        lower = -upper
        moving = velocity.abs().sum(dim=-1, keepdim=True) > 0.0
        above = (position > upper) & moving
        position = torch.where(above, 2.0 * upper - position, position)
        velocity = torch.where(above, -velocity.abs(), velocity)
        below = (position < lower) & moving
        position = torch.where(below, 2.0 * lower - position, position)
        velocity = torch.where(below, velocity.abs(), velocity)
        return position, velocity

    def _reflect_from_protected_zones(self, position, velocity, radius):
        """Keep moving obstacle surfaces outside start/goal protected disks."""
        if self.protected_zone_radius <= 0.0 or position.shape[1] == 0:
            return position, velocity
        moving = velocity.norm(dim=-1, keepdim=True) > 0.0
        min_distance = self.protected_zone_radius + radius
        centers = (
            self.p_start[:, :2].detach(),
            self.p_target[:, :2].detach(),
        )
        for center in centers:
            delta = position - center[:, None, :]
            distance = delta.norm(dim=-1, keepdim=True)
            fallback = -velocity / velocity.norm(
                dim=-1, keepdim=True).clamp_min(1e-6)
            normal = torch.where(
                distance > 1e-6,
                delta / distance.clamp_min(1e-6),
                fallback,
            )
            inside = moving & (distance < min_distance)
            position = torch.where(
                inside, center[:, None, :] + normal * min_distance, position)
            radial_speed = (velocity * normal).sum(dim=-1, keepdim=True)
            reflected = velocity - 2.0 * radial_speed.clamp_max(0.0) * normal
            velocity = torch.where(inside, reflected, velocity)
        return position, velocity

    def step_obstacles(self, dt):
        """Advance moving cylinders/balls and reflect them at map bounds."""
        if not self.has_dynamic_obstacles:
            return
        half_extent = self.map_size * 0.5
        if self.cyl.shape[1] > 0:
            radius = self.cyl[..., 2:3]
            position, self.cyl_velocity = self._reflect_planar(
                self.cyl[..., :2] + self.cyl_velocity * dt,
                self.cyl_velocity, half_extent, radius)
            position, self.cyl_velocity = self._reflect_from_protected_zones(
                position, self.cyl_velocity, radius)
            self.cyl = torch.cat([position, radius], dim=-1).detach()
        if self.balls.shape[1] > 0:
            radius = self.balls[..., 3:4]
            position, self.ball_velocity = self._reflect_planar(
                self.balls[..., :2] + self.ball_velocity * dt,
                self.ball_velocity, half_extent, radius)
            position, self.ball_velocity = self._reflect_from_protected_zones(
                position, self.ball_velocity, radius)
            self.balls = torch.cat(
                [position, self.balls[..., 2:4]], dim=-1).detach()

    def reset(self):
        # 1. start 
        start_noise = (torch.rand(self.batch_size, 2, device=self.device) - 0.5) * 1.0
        if self.randomize_start_goal:
            start_base, target_base = self.sample_start_and_target()
        else:
            start_base = torch.tensor(
                self.start_pos, device=self.device
            ).expand(self.batch_size, 2)
            target_base = torch.tensor(
                self.target_pos, device=self.device
            ).expand(self.batch_size, 2)
        
        self.p = torch.zeros((self.batch_size, 3), device=self.device)
        self.p[:, :2] = start_base + start_noise
        self.p[:, 2] = self.camera_height
        self.p_start = self.p.clone()

        # goal
        target_noise = (torch.rand(self.batch_size, 2, device=self.device) - 0.5) * 1.0
        self.p_target = torch.zeros((self.batch_size, 3), device=self.device)
        self.p_target[:, :2] = target_base + target_noise
        self.p_target[:, 2] = self.camera_height 
        

        self.randomize_obstacles()
        

        self.v = torch.zeros((self.batch_size, 1), device=self.device)
        self.omega = torch.zeros((self.batch_size, 3), device=self.device)
        self.theta = torch.zeros((self.batch_size, 3), device=self.device)


        vec_to_target = self.p_target[:, :2] - self.p[:, :2]
        ideal_yaw = torch.atan2(vec_to_target[:, 1], vec_to_target[:, 0])
        

        yaw_noise = (
            torch.rand(self.batch_size, device=self.device) - 0.5
        ) * self.initial_yaw_noise
        self.theta[:, 2] = ideal_yaw + yaw_noise

        self.R = quadsim_cuda.update_state_vec(self.theta)


        rand_pitch = (torch.rand(self.batch_size, device=self.device) - 0.5) * 2.0 *self.pitch_noise

        c = torch.cos(rand_pitch)
        s = torch.sin(rand_pitch)
        zeros = torch.zeros_like(c)
        ones = torch.ones_like(c)
        
        self.R_cam = torch.stack([
            c, zeros, -s,
            zeros, ones, zeros,
            s, zeros, c
        ], dim=-1).reshape(self.batch_size, 3, 3)
        
        self.p_old = self.p.clone()
        self.R_old = self.R.clone()

    def run(self, action, dt):
        self.p_old = self.p.clone()
        self.R_old = self.R.clone()
        
        v_cmd = action[:, 0:1] 
        w_cmd = action[:, 1:2] 

        omega_in = torch.cat(
            [self._zeros_b1, self._zeros_b1, w_cmd], dim=1)


        # v_cmd_target = action[:, 0:1] 
        # w_cmd_target = action[:, 1:2]
        # max_acc_v = 2.0
        # max_acc_w = 3.0
        # dv = v_cmd_target - self.v
        # dw = w_cmd_target - self.omega[:, 2:3] 
        # dv_clamped = torch.clamp(dv, -max_acc_v * dt, max_acc_v * dt)
        # dw_clamped = torch.clamp(dw, -max_acc_w * dt, max_acc_w * dt)
        # v_actual = self.v + dv_clamped
        # w_actual = self.omega[:, 2:3] + dw_clamped
        # zeros = torch.zeros_like(v_actual)
        # omega_in = torch.cat([zeros, zeros, w_actual], dim=1)
        
        p_next_2d, _, theta_next, _ = DiffDriveDynamics.apply(
            self.p[:, :2], 
            # v_actual,
            v_cmd,        
            self.theta, 
            omega_in, 
            dt, 
            self.grad_decay
        )
        
        self.p = torch.cat([p_next_2d, self.p[:, 2:3]], dim=1)
        self.theta = theta_next
        # self.v = v_actual
        self.v = v_cmd      
        self.omega = omega_in
        
        self.R = quadsim_cuda.update_state_vec(self.theta)
        if self.has_dynamic_obstacles:
            self.step_obstacles(dt)

    def render(self, dt):
        quadsim_cuda.render(
            self.canvas, self.flow,
            self.balls, self.cyl, self.cyl_h, 
            self.voxels, 
            self.R @ self.R_cam,   # current camera R
            self.R_old @ self.R_cam, # old camera R 
            self.p, self.p_old,
            self.drone_radius, self.n_drones_per_group,
            self.fov_x_half_tan
        )
        # depth = 1.0 / (self.canvas.unsqueeze(1) + 1e-5)
        #  (B, H, W) -> (B, 1, H, W)
        depth = self.canvas.unsqueeze(1) + 1e-5
        return depth, self.flow

    def find_vec_to_nearest_pt(self):
        if self.ground_voxels and self.voxels.shape[1] > 0:
            collision_voxels = self.voxels[:, 1:, :].contiguous()
        else:
            collision_voxels = self.voxels
        
        if self.diff_nearest_pt:
            nearest_pt = NearestPointFunction.apply(
                self.p, self.balls, self.cyl, self.cyl_h,
                collision_voxels,
                self.drone_radius, self.n_drones_per_group
            )
            return nearest_pt - self.p

        nearest_pt_ex = self.nearest_pt.unsqueeze(0)
        pos_ex = self.p.unsqueeze(0)

        quadsim_cuda.find_nearest_pt(
            nearest_pt_ex,
            self.balls, self.cyl, self.cyl_h, 
            collision_voxels, 
            pos_ex,
            self.drone_radius, self.n_drones_per_group
        )
        
        vec = nearest_pt_ex[0] - self.p
        return vec

    def signed_clearance(
        self,
        pos=None,
        include_ground=False,
        subtract_robot_radius=False,
        cylinders=None,
        balls=None,
        voxels=None,
    ):
        """Return planar differentiable signed clearance to an obstacle.

        Positive values are outside obstacles, zero lies on an obstacle
        footprint, and negative values are inside. Balls and boxes are
        projected onto the x-y plane because the differential-drive robot
        cannot act on a z gradient. By default clearance is measured from the
        robot centre; ``subtract_robot_radius=True`` returns signed robot-body
        clearance instead.
        """
        if pos is None:
            pos = self.p
        squeeze_time = pos.ndim == 2
        if squeeze_time:
            pos_t = pos.unsqueeze(0)
        elif pos.ndim == 3:
            pos_t = pos
        else:
            raise ValueError('pos must have shape [B,3] or [T,B,3]')
        timesteps, batch_size, _ = pos_t.shape

        def obstacle_history(value, name):
            if value.ndim == 3:
                if value.shape[0] != batch_size:
                    raise ValueError(f'{name} batch dimension mismatch')
                return value.unsqueeze(0).expand(timesteps, -1, -1, -1)
            if value.ndim == 4 and value.shape[:2] == (
                    timesteps, batch_size):
                return value
            raise ValueError(
                f'{name} must have shape [B,N,D] or [T,B,N,D]')

        cylinders_t = obstacle_history(
            self.cyl if cylinders is None else cylinders, 'cylinders')
        balls_t = obstacle_history(
            self.balls if balls is None else balls, 'balls')
        voxels_t = obstacle_history(
            self.voxels if voxels is None else voxels, 'voxels')
        distances = []

        if cylinders_t.shape[2] > 0:
            delta_xy = pos_t[:, :, None, :2] - cylinders_t[..., :2]
            distances.append(
                (delta_xy.norm(dim=-1) - cylinders_t[..., 2])
                .min(dim=-1).values)

        if balls_t.shape[2] > 0:
            delta_xy = pos_t[:, :, None, :2] - balls_t[..., :2]
            distances.append(
                (delta_xy.norm(dim=-1) - balls_t[..., 3])
                .min(dim=-1).values)

        if self.ground_voxels and not include_ground and voxels_t.shape[2] > 0:
            voxels_t = voxels_t[:, :, 1:, :]
        if voxels_t.shape[2] > 0:
            delta_xy = pos_t[:, :, None, :2] - voxels_t[..., :2]
            q = delta_xy.abs() - voxels_t[..., 3:5]
            outside_distance = q.clamp_min(0.0).norm(dim=-1)
            inside_distance = q.max(dim=-1).values.clamp_max(0.0)
            distances.append(
                (outside_distance + inside_distance).min(dim=-1).values)

        if distances:
            clearance = torch.stack(distances, dim=-1).min(dim=-1).values
        else:
            clearance = pos_t.new_full((timesteps, batch_size), 10.0)
        if subtract_robot_radius:
            clearance = clearance - self.drone_radius
        return clearance.squeeze(0) if squeeze_time else clearance
