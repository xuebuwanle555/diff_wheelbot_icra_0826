import torch
import torch.nn as nn
import torch.nn.functional as F


def depth_to_local_obstacle_points(
    depth,
    fov_x_half_tan=0.82,
    num_points=16,
    height_fraction=0.4,
    depth_quantile=0.1,
    min_range=0.2,
    max_range=6.0,
):
    """Convert a depth image to robust 2-D obstacle-surface points.

    The CUDA renderer stores ray parameter ``t``.  In the robot frame its
    horizontal ray is ``(x, y) = (t, -t * tan(angle))``; this is also the usual
    pinhole back-projection for a forward-facing metric depth camera.  The
    lower image rows are excluded because they frequently observe the ground.

    A fixed number of horizontal sectors keeps the safety MPC cost bounded.
    Each sector uses a low vertical depth quantile instead of a single minimum,
    making isolated invalid/noisy pixels much less influential.

    Returns:
        points: ``[B, K, 2]`` local obstacle-surface points.
        valid: ``[B, K]`` mask for points within the configured sensing range.
    """
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("depth must have shape [B,H,W] or [B,1,H,W]")
    if num_points < 1:
        raise ValueError("num_points must be positive")
    if not 0.0 < height_fraction <= 1.0:
        raise ValueError("height_fraction must be in (0, 1]")
    if not 0.0 < depth_quantile <= 1.0:
        raise ValueError("depth_quantile must be in (0, 1]")
    if min_range < 0.0 or max_range <= min_range:
        raise ValueError("Require 0 <= min_range < max_range")

    # The renderer is not differentiated.  Detaching also prevents its reused
    # CUDA canvas from becoming part of the long training graph.
    metric_depth = depth.detach().nan_to_num(
        nan=max_range + 1.0,
        posinf=max_range + 1.0,
        neginf=min_range,
    )
    height = metric_depth.shape[-2]
    crop_height = max(1, min(
        height, int(round(height * height_fraction))
    ))
    crop = metric_depth[..., :crop_height, :]

    # Horizontal min-pooling retains thin obstacles before the robust vertical
    # quantile.  The output has one depth measurement per angular sector.
    sector_depth = -F.adaptive_max_pool2d(
        -crop, output_size=(crop_height, num_points)
    )
    vertical = sector_depth.squeeze(1).transpose(1, 2)  # [B,K,H]
    kth = max(1, min(
        crop_height, int(round(crop_height * depth_quantile))
    ))
    radial_depth = vertical.kthvalue(kth, dim=-1).values
    valid = (
        torch.isfinite(radial_depth)
        & (radial_depth >= min_range)
        & (radial_depth <= max_range)
    )
    radial_depth = radial_depth.clamp(min_range, max_range)

    sector = torch.arange(
        num_points, device=depth.device, dtype=depth.dtype
    )
    horizontal_tan = (
        2.0 * (sector + 0.5) / num_points - 1.0
    ) * fov_x_half_tan
    x = radial_depth
    y = -radial_depth * horizontal_tan.unsqueeze(0)
    return torch.stack([x, y], dim=-1), valid


def estimate_local_obstacle_velocity(
    obstacle_points,
    obstacle_mask,
    previous_points,
    previous_mask,
    current_speed,
    current_omega,
    dt,
    association_distance=0.75,
    max_obstacle_speed=1.5,
):
    """Estimate planar obstacle velocity from consecutive perceived points.

    Previous points are first transformed into the current robot frame using
    odometry, so static obstacles yield approximately zero velocity.  Current
    sectors are then associated with their nearest compensated previous point.
    Invalid/ambiguous matches fall back to the static-obstacle assumption.
    """
    velocity = torch.zeros_like(obstacle_points)
    if previous_points is None or previous_mask is None:
        return velocity
    if previous_points.shape != obstacle_points.shape:
        return velocity
    if previous_mask.shape != obstacle_mask.shape:
        return velocity
    if current_speed.ndim == 2:
        current_speed = current_speed[:, 0]
    if current_omega.ndim == 2:
        current_omega = current_omega[:, 0]
    if current_speed.shape != obstacle_points.shape[:1]:
        raise ValueError("current_speed must have shape [B] or [B,1]")
    if current_omega.shape != obstacle_points.shape[:1]:
        raise ValueError("current_omega must have shape [B] or [B,1]")
    if association_distance <= 0.0 or max_obstacle_speed < 0.0:
        raise ValueError("Invalid obstacle velocity estimator limits")
    if max_obstacle_speed == 0.0:
        return velocity

    dt_safe = torch.as_tensor(
        dt, dtype=obstacle_points.dtype, device=obstacle_points.device
    ).clamp_min(1e-3)
    delta_yaw = current_omega.detach() * dt_safe
    c = torch.cos(delta_yaw)
    s = torch.sin(delta_yaw)
    translated_x = (
        previous_points[..., 0]
        - current_speed.detach()[:, None] * dt_safe
    )
    previous_y = previous_points[..., 1]
    compensated_previous = torch.stack(
        [
            c[:, None] * translated_x + s[:, None] * previous_y,
            -s[:, None] * translated_x + c[:, None] * previous_y,
        ],
        dim=-1,
    )

    pairwise = torch.cdist(
        obstacle_points.detach(), compensated_previous.detach()
    )
    pairwise = pairwise.masked_fill(
        ~previous_mask[:, None, :], float("inf")
    )
    match_distance, match_index = pairwise.min(dim=-1)
    gather_index = match_index.unsqueeze(-1).expand(-1, -1, 2)
    matched_previous = compensated_previous.gather(1, gather_index)
    matched_valid = (
        obstacle_mask
        & torch.isfinite(match_distance)
        & (match_distance <= association_distance)
    )
    velocity = (obstacle_points - matched_previous) / dt_safe
    speed = velocity.norm(dim=-1, keepdim=True)
    velocity = velocity * (
        max_obstacle_speed / speed.clamp_min(max_obstacle_speed)
    )
    return torch.where(
        matched_valid.unsqueeze(-1), velocity, torch.zeros_like(velocity)
    ).detach()


def estimate_emergency_risk(
    depth,
    previous_clearance,
    dt,
    robot_radius=0.15,
    emergency_distance=1.5,
    emergency_ttc=0.8,
    depth_quantile=0.02,
    depth_height_fraction=0.5,
):
    """Estimate a deployable [0, 1] emergency level from consecutive depths.

    A low percentile is used instead of a single minimum pixel so isolated
    depth noise does not switch the controller into emergency mode.  The
    returned clearance is detached deliberately: this signal selects controller
    authority and is not a shortcut for back-propagating through the renderer.
    """
    if depth.ndim < 2:
        raise ValueError("depth must include a batch dimension")
    if emergency_distance <= 0.0 or emergency_ttc <= 0.0:
        raise ValueError("Emergency distance and TTC must be positive")
    if not 0.0 < depth_quantile <= 1.0:
        raise ValueError("depth_quantile must be in (0, 1]")
    if not 0.0 < depth_height_fraction <= 1.0:
        raise ValueError("depth_height_fraction must be in (0, 1]")

    # The lower image rows see the ground at roughly 0.4 m even in an empty
    # map. Restrict risk estimation to the upper field of view so the ground
    # cannot keep emergency mode permanently active. This crop works for both
    # [B,H,W] CUDA depth and [B,1,H,W] ROS depth tensors.
    height = depth.shape[-2]
    crop_height = max(1, min(height, int(round(height * depth_height_fraction))))
    risk_depth = depth[..., :crop_height, :]
    flat = risk_depth.detach().flatten(start_dim=1)
    kth = max(1, min(flat.shape[1], int(round(flat.shape[1] * depth_quantile))))
    near_depth = flat.kthvalue(kth, dim=1).values
    clearance = (near_depth - robot_radius).clamp_min(0.0)
    distance_risk = (
        (emergency_distance - clearance) / emergency_distance
    ).clamp(0.0, 1.0)

    if previous_clearance is None:
        ttc_risk = torch.zeros_like(distance_risk)
    else:
        dt_safe = torch.as_tensor(
            dt, dtype=clearance.dtype, device=clearance.device
        ).clamp_min(1e-3)
        approach_speed = ((previous_clearance - clearance) / dt_safe).clamp_min(0.0)
        ttc = clearance / approach_speed.clamp_min(1e-3)
        ttc_risk = ((emergency_ttc - ttc) / emergency_ttc).clamp(0.0, 1.0)

    risk = torch.maximum(distance_risk, ttc_risk).unsqueeze(-1)
    return risk, clearance


class DifferentiableWaypointMPC(nn.Module):
    """Batched differentiable finite-horizon trajectory MPC.

    The MPC solves a convex quadratic trajectory tracking problem in the current
    robot frame: waypoint-reference error + second-difference smoothness +
    initial-velocity consistency.  Its normal equations are solved from a cached
    Cholesky factor with PyTorch operations, so gradients from the executed
    command pass through the planned trajectory to every predicted waypoint.

    The policy's desired speed is filtered by curvature, acceleration and hard
    command limits.  Angular velocity comes from a differential-drive-compatible
    pure-pursuit projection of the optimized path.  Re-running this module at
    every control step gives the usual receding-horizon MPC behavior.
    """

    def __init__(
        self,
        num_waypoints=3,
        horizon=12,
        control_lookahead=3,
        max_v=4.0,
        max_omega=3.0,
        max_acc_v=8.0,
        max_acc_omega=8.0,
        max_lateral_acc=8.0,
        track_weight=8.0,
        smooth_weight=30.0,
        emergency_smooth_weight=None,
        initial_velocity_weight=4.0,
        emergency_max_acc_omega=None,
        perception_safety_enabled=False,
        obstacle_safety_clearance=0.30,
        obstacle_temperature=0.15,
        obstacle_weight=1.0,
        obstacle_refine_steps=2,
        obstacle_step_size=0.06,
        intervention_weight=3.0,
        obstacle_slowdown_gain=0.35,
        obstacle_min_speed_scale=0.25,
        ttc_clearance_inflation=0.15,
        collect_diagnostics=True,
    ):
        super().__init__()
        if num_waypoints < 1:
            raise ValueError("num_waypoints must be positive")
        if horizon < num_waypoints:
            raise ValueError("horizon must be at least num_waypoints")
        if not 1 <= control_lookahead <= horizon:
            raise ValueError("control_lookahead must be in [1, horizon]")
        if max_v <= 0.0 or max_omega <= 0.0:
            raise ValueError("max_v and max_omega must be positive")
        if max_acc_v <= 0.0 or max_acc_omega <= 0.0 or max_lateral_acc <= 0.0:
            raise ValueError("MPC acceleration limits must be positive")
        if track_weight <= 0.0:
            raise ValueError("track_weight must be positive")
        if smooth_weight < 0.0 or initial_velocity_weight < 0.0:
            raise ValueError("MPC regularization weights must be non-negative")
        if emergency_smooth_weight is None:
            emergency_smooth_weight = smooth_weight
        if emergency_max_acc_omega is None:
            emergency_max_acc_omega = max_acc_omega
        if emergency_smooth_weight < 0.0:
            raise ValueError("emergency_smooth_weight must be non-negative")
        if emergency_max_acc_omega < max_acc_omega:
            raise ValueError(
                "emergency_max_acc_omega must be at least max_acc_omega"
            )
        if obstacle_safety_clearance <= 0.0 or obstacle_temperature <= 0.0:
            raise ValueError(
                "Obstacle safety clearance and temperature must be positive"
            )
        if obstacle_weight < 0.0 or obstacle_refine_steps < 0:
            raise ValueError(
                "Obstacle weight/refinement steps must be non-negative"
            )
        if obstacle_step_size < 0.0 or intervention_weight < 0.0:
            raise ValueError(
                "Obstacle step size/intervention weight must be non-negative"
            )
        if not 0.0 <= obstacle_slowdown_gain <= 1.0:
            raise ValueError("obstacle_slowdown_gain must be in [0, 1]")
        if not 0.0 < obstacle_min_speed_scale <= 1.0:
            raise ValueError("obstacle_min_speed_scale must be in (0, 1]")
        if ttc_clearance_inflation < 0.0:
            raise ValueError("ttc_clearance_inflation must be non-negative")

        self.num_waypoints = num_waypoints
        self.horizon = horizon
        self.control_lookahead = control_lookahead
        self.max_v = max_v
        self.max_omega = max_omega
        self.max_acc_v = max_acc_v
        self.max_acc_omega = max_acc_omega
        self.max_lateral_acc = max_lateral_acc
        self.track_weight = track_weight
        self.smooth_weight = smooth_weight
        self.emergency_smooth_weight = emergency_smooth_weight
        self.initial_velocity_weight = initial_velocity_weight
        self.emergency_max_acc_omega = emergency_max_acc_omega
        self.perception_safety_enabled = bool(perception_safety_enabled)
        self.obstacle_safety_clearance = obstacle_safety_clearance
        self.obstacle_temperature = obstacle_temperature
        self.obstacle_weight = obstacle_weight
        self.obstacle_refine_steps = obstacle_refine_steps
        self.obstacle_step_size = obstacle_step_size
        self.intervention_weight = intervention_weight
        self.obstacle_slowdown_gain = obstacle_slowdown_gain
        self.obstacle_min_speed_scale = obstacle_min_speed_scale
        self.ttc_clearance_inflation = ttc_clearance_inflation
        self.collect_diagnostics = bool(collect_diagnostics)

        # D2 maps [x_1, ..., x_H] to discrete accelerations.  x_0 is the
        # current origin and x_-1 is supplied from the measured forward speed.
        d2 = torch.zeros(horizon, horizon)
        for row in range(horizon):
            d2[row, row] = 1.0
            if row >= 1:
                d2[row, row - 1] = -2.0
            if row >= 2:
                d2[row, row - 2] = 1.0

        def make_system(weight):
            matrix = track_weight * torch.eye(horizon)
            matrix = matrix + weight * (d2.transpose(0, 1) @ d2)
            matrix[0, 0] += initial_velocity_weight
            return matrix

        system = make_system(smooth_weight)
        emergency_system = make_system(emergency_smooth_weight)

        self.register_buffer("d2", d2)
        self.register_buffer(
            "horizon_steps", torch.arange(1, horizon + 1, dtype=d2.dtype)
        )
        self.register_buffer("system_matrix", system)
        self.register_buffer("system_cholesky", torch.linalg.cholesky(system))
        self.register_buffer(
            "emergency_system_cholesky", torch.linalg.cholesky(emergency_system)
        )
        self.last_info = {}

    def _reference_trajectory(self, waypoints):
        origin = torch.zeros_like(waypoints[:, :1])
        knots = torch.cat([origin, waypoints], dim=1)
        # With three waypoints and H=12, the knots land exactly at indices
        # 0, 4, 8 and 12.  Other horizon lengths are handled continuously.
        return F.interpolate(
            knots.transpose(1, 2),
            size=self.horizon + 1,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)

    def time_parameterize(self, trajectory, desired_speed, current_speed, dt,
                          num_samples):
        """Resample a geometric path at dynamically reachable timestamps.

        ``trajectory`` is the MPC's geometric path; its original point spacing
        is not a time discretization.  This method builds a speed sequence
        under ``max_acc_v``, converts it to cumulative travel distance, and
        interpolates the path by arc length.  Segment selection is piecewise
        constant while interpolation remains differentiable with respect to
        path coordinates and desired speed.
        """
        if trajectory.ndim != 3 or trajectory.shape[-1] != 2:
            raise ValueError("trajectory must have shape [B,H,2]")
        if desired_speed.shape != current_speed.shape or current_speed.ndim != 2:
            raise ValueError("desired_speed/current_speed must have shape [B,1]")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")

        dt_safe = torch.as_tensor(
            dt, dtype=trajectory.dtype, device=trajectory.device
        ).clamp_min(1e-3)
        target_speed = desired_speed.clamp(0.0, self.max_v)
        speed = current_speed.clamp(0.0, self.max_v)
        max_delta_v = self.max_acc_v * dt_safe
        if num_samples == 1:
            query_distance = torch.zeros_like(speed)
        else:
            # The target is constant over this short horizon, so the original
            # recurrent acceleration limiter has this closed form.  Computing
            # every sample together avoids a Python/kernel-launch loop while
            # preserving the same reachable-speed sequence.
            if num_samples - 1 <= self.horizon_steps.numel():
                sample_steps = self.horizon_steps[:num_samples - 1]
            else:
                sample_steps = torch.arange(
                    1, num_samples,
                    device=trajectory.device,
                    dtype=trajectory.dtype,
                )
            max_speed_change = max_delta_v * sample_steps.unsqueeze(0)
            speed_samples = speed + torch.clamp(
                target_speed - speed,
                min=-max_speed_change,
                max=max_speed_change,
            )
            speed_samples = speed_samples.clamp(0.0, self.max_v)
            travel_samples = (speed_samples * dt_safe).cumsum(dim=1)
            query_distance = torch.cat(
                [torch.zeros_like(speed), travel_samples], dim=1
            )

        segment_length = trajectory.diff(dim=1).norm(dim=-1)
        cumulative_length = torch.cat(
            [torch.zeros_like(segment_length[:, :1]), segment_length.cumsum(dim=1)],
            dim=1,
        )
        total_length = cumulative_length[:, -1:]
        query_distance = torch.minimum(query_distance, total_length)

        # Pick interpolation segments without asking autograd to differentiate
        # through the discrete search itself.
        segment_index = (
            cumulative_length.detach()[:, None, :]
            <= query_distance.detach()[:, :, None]
        ).sum(dim=-1) - 1
        segment_index = segment_index.clamp(0, trajectory.shape[1] - 2)
        next_index = segment_index + 1

        gather_xy = segment_index.unsqueeze(-1).expand(-1, -1, 2)
        gather_next_xy = next_index.unsqueeze(-1).expand(-1, -1, 2)
        p0 = trajectory.gather(1, gather_xy)
        p1 = trajectory.gather(1, gather_next_xy)
        s0 = cumulative_length.gather(1, segment_index)
        s1 = cumulative_length.gather(1, next_index)
        alpha = (
            (query_distance - s0) / (s1 - s0).clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        return p0 + alpha.unsqueeze(-1) * (p1 - p0)

    def _solve_trajectory(self, reference_future, boundary, smooth_weight,
                          cholesky):
        rhs = self.track_weight * reference_future
        rhs = rhs - smooth_weight * torch.matmul(
            self.d2.transpose(0, 1), boundary
        )
        rhs = rhs.clone()
        return rhs, cholesky

    def _refine_trajectory_for_obstacles(
        self,
        trajectory,
        obstacle_points,
        obstacle_mask,
        obstacle_velocity,
        emergency_risk,
        dt,
    ):
        """Apply differentiable proximal safety steps to the horizon trajectory.

        This is a hand-derived descent step on a soft obstacle-clearance cost.
        It avoids nested ``autograd.grad(create_graph=True)`` calls, keeping
        long differentiable rollouts practical while retaining gradients from
        the refined trajectory back to the waypoint policy.
        """
        batch_size = trajectory.shape[0]
        zero_risk = trajectory.new_zeros((batch_size, 1))
        far_clearance = trajectory.new_full((batch_size, 1), 1e3)
        zero_intervention = trajectory.new_zeros((batch_size, 1))
        if (
            not self.perception_safety_enabled
            or self.obstacle_refine_steps == 0
            or obstacle_points is None
            or obstacle_mask is None
            or obstacle_points.shape[1] == 0
        ):
            return trajectory, zero_risk, far_clearance, zero_intervention
        if obstacle_points.ndim != 3 or obstacle_points.shape[0] != batch_size:
            raise ValueError("obstacle_points must have shape [B,K,2]")
        if obstacle_points.shape[-1] != 2:
            raise ValueError("obstacle_points must have shape [B,K,2]")
        if obstacle_mask.shape != obstacle_points.shape[:2]:
            raise ValueError("obstacle_mask must have shape [B,K]")
        if obstacle_velocity is None:
            obstacle_velocity = torch.zeros_like(obstacle_points)
        if obstacle_velocity.shape != obstacle_points.shape:
            raise ValueError("obstacle_velocity must have shape [B,K,2]")

        nominal = trajectory[:, 1:]
        refined = nominal
        valid = obstacle_mask[:, None, :]
        valid_f = valid.to(trajectory.dtype)
        effective_clearance = (
            self.obstacle_safety_clearance
            + emergency_risk * self.ttc_clearance_inflation
        )
        perception_risk = zero_risk
        predicted_min_clearance = far_clearance
        horizon_time = (
            self.horizon_steps[:nominal.shape[1]]
            * torch.as_tensor(
                dt, device=trajectory.device, dtype=trajectory.dtype
            ).clamp_min(1e-3)
        )
        future_obstacle_points = (
            obstacle_points[:, None, :, :]
            + horizon_time[None, :, None, None]
            * obstacle_velocity[:, None, :, :]
        )

        # Pick one consistent passing side for every perceived obstacle.  A
        # purely radial potential pushes points before an obstacle backwards
        # and points after it forwards, which can kink the path without choosing
        # a valid left/right route.  Prefer the side already suggested by the
        # nominal waypoint path; when it is ambiguous, pass on the side away
        # from the obstacle's bearing relative to the robot centerline.
        nominal_delta = (
            nominal[:, :, None, :]
            - future_obstacle_points
        )
        nominal_distance = nominal_delta.square().sum(dim=-1)
        closest_index = nominal_distance.argmin(dim=1)
        closest_y = nominal[..., 1].gather(1, closest_index)
        closest_obstacle_y = future_obstacle_points[..., 1].gather(
            1, closest_index.unsqueeze(1)
        ).squeeze(1)
        lateral_offset = closest_y - closest_obstacle_y
        endpoint_side = torch.where(
            nominal[:, -1:, 1] >= 0.0,
            torch.ones_like(nominal[:, -1:, 1]),
            -torch.ones_like(nominal[:, -1:, 1]),
        )
        bearing_side = torch.where(
            obstacle_points[..., 1] > 0.05,
            -torch.ones_like(lateral_offset),
            torch.where(
                obstacle_points[..., 1] < -0.05,
                torch.ones_like(lateral_offset),
                endpoint_side.expand_as(lateral_offset),
            ),
        )
        passing_side = torch.where(
            lateral_offset.abs() > 0.05,
            torch.sign(lateral_offset),
            bearing_side,
        ).detach()

        for _ in range(self.obstacle_refine_steps):
            delta = (
                refined[:, :, None, :]
                - future_obstacle_points
            )
            distance = delta.square().sum(dim=-1).add(1e-6).sqrt()
            surface_clearance = distance - self.obstacle_safety_clearance
            masked_clearance = surface_clearance.masked_fill(
                ~valid, float("inf")
            )
            predicted_min_clearance = torch.minimum(
                predicted_min_clearance,
                masked_clearance.amin(dim=(1, 2), keepdim=False).unsqueeze(-1),
            )

            activation = torch.sigmoid(
                (
                    effective_clearance[:, None, :]
                    - distance
                ) / self.obstacle_temperature
            ) * valid_f
            point_risk = activation.amax(dim=-1, keepdim=True)
            perception_risk = torch.maximum(
                perception_risk,
                point_risk.amax(dim=1),
            )

            lateral_direction = (
                activation * passing_side[:, None, :]
            ).sum(dim=2, keepdim=True) / activation.sum(
                dim=2, keepdim=True
            ).clamp_min(1e-6)
            direction = torch.cat(
                [torch.zeros_like(lateral_direction), lateral_direction],
                dim=-1,
            )
            displacement = (
                self.obstacle_step_size
                * self.obstacle_weight
                * point_risk
                * direction
            )
            candidate = refined + displacement
            proximal_scale = 1.0 / (
                1.0 + self.obstacle_step_size * self.intervention_weight
            )
            refined = nominal + proximal_scale * (candidate - nominal)

        refined_trajectory = torch.cat(
            [trajectory[:, :1], refined], dim=1
        )
        intervention = (
            refined - nominal
        ).square().sum(dim=-1).mean(dim=1, keepdim=True).sqrt()
        return (
            refined_trajectory,
            perception_risk,
            predicted_min_clearance,
            intervention,
        )

    def forward(self, waypoints, desired_speed, current_speed, dt,
                current_omega=None, emergency_risk=None,
                obstacle_points=None, obstacle_mask=None,
                obstacle_velocity=None):
        """Return ``(command, local_trajectory)``.

        Args:
            waypoints: ``(B, N, 2)`` local waypoint positions.
            desired_speed: ``(B, 1)`` nominal speed proposed by the policy.
            current_speed: ``(B, 1)`` measured forward speed.
            current_omega: optional ``(B, 1)`` measured angular speed. When
                supplied, angular acceleration is limited around this value.
            emergency_risk: optional ``(B, 1)`` value in ``[0, 1]``. Higher
                risk blends toward a less-smoothed path and a larger angular
                acceleration limit while preserving the normal controller in
                safe states.
            obstacle_points: optional ``(B,K,2)`` perceived local obstacle
                surface points.
            obstacle_mask: optional ``(B,K)`` validity mask.
            obstacle_velocity: optional ``(B,K,2)`` estimated obstacle motion
                in the current robot frame after ego-motion compensation.
            dt: scalar control period in seconds.
        """
        if waypoints.ndim != 3 or waypoints.shape[1:] != (self.num_waypoints, 2):
            raise ValueError(
                f"waypoints must have shape (B, {self.num_waypoints}, 2), "
                f"got {tuple(waypoints.shape)}"
            )
        if current_speed.ndim != 2 or current_speed.shape[1] != 1:
            raise ValueError("current_speed must have shape (B, 1)")
        if desired_speed.ndim != 2 or desired_speed.shape != current_speed.shape:
            raise ValueError("desired_speed must have shape (B, 1)")
        if current_omega is not None and current_omega.shape != current_speed.shape:
            raise ValueError("current_omega must have shape (B, 1)")
        if emergency_risk is not None and emergency_risk.shape != current_speed.shape:
            raise ValueError("emergency_risk must have shape (B, 1)")
        if emergency_risk is not None:
            emergency_risk = emergency_risk.clamp(0.0, 1.0)

        dt_tensor = torch.as_tensor(dt, dtype=waypoints.dtype, device=waypoints.device)
        if dt_tensor.numel() != 1:
            raise ValueError("dt must be a scalar")
        dt_safe = dt_tensor.clamp_min(1e-3)

        reference = self._reference_trajectory(waypoints)
        reference_future = reference[:, 1:]

        # Boundary term for x_-1 = [-v*dt, 0].  In D2*x + boundary, the first
        # row therefore represents x_1 - v*dt and anchors the planned initial
        # velocity to the actual motor state.
        boundary = torch.zeros_like(reference_future)
        boundary[:, 0, 0] = -current_speed[:, 0] * dt_safe

        rhs, cholesky = self._solve_trajectory(
            reference_future, boundary, self.smooth_weight,
            self.system_cholesky,
        )
        rhs[:, 0, 0] += self.initial_velocity_weight * current_speed[:, 0] * dt_safe

        # The Hessian is fixed for a given MPC configuration, so reuse its
        # Cholesky factor instead of factorizing it at every rollout step.
        planned_future = torch.cholesky_solve(rhs, cholesky)
        if (
            emergency_risk is not None
            and self.emergency_smooth_weight != self.smooth_weight
        ):
            emergency_rhs, emergency_cholesky = self._solve_trajectory(
                reference_future, boundary, self.emergency_smooth_weight,
                self.emergency_system_cholesky,
            )
            emergency_rhs[:, 0, 0] += (
                self.initial_velocity_weight * current_speed[:, 0] * dt_safe
            )
            emergency_future = torch.cholesky_solve(
                emergency_rhs, emergency_cholesky
            )
            blend = emergency_risk.unsqueeze(-1)
            planned_future = planned_future + blend * (
                emergency_future - planned_future
            )
        origin = torch.zeros_like(planned_future[:, :1])
        trajectory = torch.cat([origin, planned_future], dim=1)
        if self.perception_safety_enabled:
            if emergency_risk is None:
                emergency_risk = torch.zeros_like(current_speed)
            (
                trajectory,
                perception_risk,
                predicted_min_clearance,
                intervention_norm,
            ) = self._refine_trajectory_for_obstacles(
                trajectory,
                obstacle_points,
                obstacle_mask,
                obstacle_velocity,
                emergency_risk,
                dt_safe,
            )
        else:
            perception_risk = None
            predicted_min_clearance = None
            intervention_norm = None

        # Convert the first part of the geometric trajectory to a feasible
        # differential-drive command.  Curvature kappa=2y/L^2 is the standard
        # local pure-pursuit projection and stays fully differentiable.
        first_segment = trajectory[:, 1] - trajectory[:, 0]
        lookahead_point = trajectory[:, self.control_lookahead]
        lookahead_sq = lookahead_point.square().sum(dim=-1, keepdim=True)
        curvature = 2.0 * lookahead_point[:, 1:2] / (lookahead_sq + 1e-4)

        # The policy selects a nominal speed while the MPC/controller selects
        # what can actually be executed on this path.  Path curvature, the hard
        # velocity bound and acceleration from the measured speed are applied
        # before producing the command.  Waypoint spacing is geometric rather
        # than temporal, so it is deliberately not treated as a speed request.
        desired_speed_bounded = desired_speed.clamp(min=0.0, max=self.max_v)
        if perception_risk is None:
            obstacle_speed_scale = None
            perception_speed_limit = desired_speed_bounded
        else:
            obstacle_speed_scale = (
                1.0 - self.obstacle_slowdown_gain * perception_risk
            ).clamp(min=self.obstacle_min_speed_scale, max=1.0)
            perception_speed_limit = desired_speed_bounded * obstacle_speed_scale
        curvature_speed_limit = torch.sqrt(
            self.max_lateral_acc / (curvature.abs() + 1e-4)
        ).clamp(max=self.max_v)
        feasible_speed = torch.minimum(
            perception_speed_limit, curvature_speed_limit
        )
        max_delta_v = self.max_acc_v * dt_safe
        v_cmd = current_speed + (feasible_speed - current_speed).clamp(
            min=-max_delta_v,
            max=max_delta_v,
        )
        v_cmd = v_cmd.clamp(min=0.0, max=self.max_v)

        omega_unbounded = v_cmd * curvature
        omega_desired = self.max_omega * torch.tanh(
            omega_unbounded / self.max_omega
        )
        if current_omega is None:
            omega_cmd = omega_desired
            angular_acceleration_limited = None
        else:
            if (
                emergency_risk is None
                or self.emergency_max_acc_omega == self.max_acc_omega
            ):
                effective_max_acc_omega = self.max_acc_omega
            else:
                effective_max_acc_omega = (
                    self.max_acc_omega
                    + emergency_risk
                    * (self.emergency_max_acc_omega - self.max_acc_omega)
                )
            max_delta_omega = effective_max_acc_omega * dt_safe
            omega_error = omega_desired - current_omega
            omega_cmd = current_omega + omega_error.clamp(
                min=-max_delta_omega, max=max_delta_omega
            )
            omega_cmd = omega_cmd.clamp(-self.max_omega, self.max_omega)
            angular_acceleration_limited = None

        command = torch.cat([v_cmd, omega_cmd], dim=-1)
        if self.collect_diagnostics:
            if emergency_risk is None:
                emergency_risk = torch.zeros_like(current_speed)
            if perception_risk is None:
                perception_risk = torch.zeros_like(current_speed)
                predicted_min_clearance = torch.full_like(
                    current_speed, 1e3)
                intervention_norm = torch.zeros_like(current_speed)
                obstacle_speed_scale = torch.ones_like(current_speed)
            planned_speed = torch.linalg.vector_norm(
                first_segment, dim=-1, keepdim=True
            ) / dt_safe
            if current_omega is None:
                angular_acceleration_limited = torch.zeros_like(
                    omega_cmd, dtype=torch.bool)
            else:
                angular_acceleration_limited = (
                    omega_error.abs() > max_delta_omega + 1e-6)
            self.last_info = {
                "desired_speed": desired_speed_bounded,
                "planned_speed": planned_speed,
                "curvature_speed_limited": (
                    curvature_speed_limit + 1e-6
                    < desired_speed_bounded
                ),
                "acceleration_limited": (
                    (feasible_speed - current_speed).abs()
                    > max_delta_v + 1e-6
                ),
                "angular_acceleration_limited": angular_acceleration_limited,
                "curvature": curvature,
                "emergency_risk": emergency_risk,
                "perception_risk": perception_risk,
                "predicted_min_clearance": predicted_min_clearance,
                "intervention_norm": intervention_norm,
                "safety_intervened": perception_risk > 0.1,
                "obstacle_speed_scale": obstacle_speed_scale,
            }
        return command, trajectory
