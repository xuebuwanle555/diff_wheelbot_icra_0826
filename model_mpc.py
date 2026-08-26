import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Depth/goal policy that predicts local waypoints and desired speed.

    Waypoints are expressed in the current robot frame (x forward, y left).  The
    output parameterization uses positive forward increments because this robot
    is controlled with forward-only linear velocity.  Lateral increments remain
    signed, so the policy can describe left/right turns around obstacles.  A
    separate speed head expresses intent; the MPC enforces execution limits.
    """

    def __init__(
        self,
        dim_obs=6,
        num_waypoints=3,
        hidden_dim=192,
        input_w=32,
        input_h=24,
        min_forward_step=0.0,
        max_forward_step=1.5,
        max_lateral_step=1.0,
        max_speed=4.0,
        initial_desired_speed=3.2,
        min_desired_speed=0.0,
        direct_action=False,
    ):
        """
        input: depth_raw + goal state
        output: local waypoints and desired speed when direct_action=False
                action (v, omega) (B, 2) when direct_action=True
        """
        super().__init__()

        self.num_waypoints = num_waypoints
        self.min_forward_step = min_forward_step
        self.max_forward_step = max_forward_step
        self.max_lateral_step = max_lateral_step
        if max_speed <= 0.0:
            raise ValueError("max_speed must be positive")
        if not 0.0 <= min_desired_speed < max_speed:
            raise ValueError("min_desired_speed must be in [0, max_speed)")
        if not min_desired_speed < initial_desired_speed < max_speed:
            raise ValueError(
                "initial_desired_speed must be between min_desired_speed "
                "and max_speed")
        self.max_speed = max_speed
        self.min_desired_speed = min_desired_speed
        self.direct_action = direct_action

        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=2, bias=False),  
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, kernel_size=3, bias=False), 
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, kernel_size=3, bias=False), 
            nn.LeakyReLU(0.05),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_h, input_w)
            out = self.conv_net(dummy)
            self.conv_out_dim = out.shape[1]

        self.fc_visual = nn.Linear(self.conv_out_dim, 192, bias=False)


        self.fc_state = nn.Linear(dim_obs, 64)

        self.fc_state.weight.data.mul_(0.5)

        self.gru = nn.GRUCell(192 + 64, hidden_dim)

        self.waypoint_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.05),
            nn.Linear(128, num_waypoints * 2),
        )
        # Start from a nearly straight, evenly-spaced local path.  Keeping the
        # final layer small also avoids saturated waypoint transforms early on.
        self.waypoint_head[2].weight.data.mul_(0.01)
        self.waypoint_head[2].bias.data.zero_()

        # The policy proposes a nominal speed; the MPC remains responsible for
        # enforcing path, curvature, acceleration and hard command limits.
        self.speed_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.05),
            nn.Linear(128, 1),
        )
        self.speed_head[2].weight.data.mul_(0.01)
        initial_fraction = torch.tensor(
            (initial_desired_speed - min_desired_speed)
            / (max_speed - min_desired_speed)
        )
        initial_logit = torch.logit(initial_fraction, eps=1e-4).item()
        self.speed_head[2].bias.data.fill_(initial_logit)

        # Direct action head: outputs (v, omega) without MPC
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.05),
            nn.Linear(128, 2),
        )
        # Initialise near zero so the policy starts with gentle commands
        self.action_head[2].weight.data.mul_(0.01)
        self.action_head[2].bias.data.zero_()

    def reset(self):
        pass

    def forward(self, depth, state, h=None):
        B = depth.size(0)
        
        visual_feat = self.fc_visual(self.conv_net(depth))
        state_feat = F.leaky_relu(self.fc_state(state), 0.05)
        
        fusion = torch.cat([visual_feat, state_feat], dim=1)
        
        if h is None:
            h = torch.zeros(B, self.gru.hidden_size, device=depth.device, dtype=depth.dtype)
        h_new = self.gru(fusion, h)
        
        if self.direct_action:
            raw_action = self.action_head(h_new)
            # raw_action[:, 0] -> forward speed (sigmoid, scaled by max_v)
            # raw_action[:, 1] -> angular rate  (tanh, scaled by max_omega)
            return raw_action, h_new

        raw_waypoints = self.waypoint_head(h_new).view(B, self.num_waypoints, 2)

        forward_step = self.min_forward_step + (
            self.max_forward_step - self.min_forward_step
        ) * torch.sigmoid(raw_waypoints[..., 0])
        lateral_step = self.max_lateral_step * torch.tanh(raw_waypoints[..., 1])
        waypoint_steps = torch.stack([forward_step, lateral_step], dim=-1)
        waypoints = torch.cumsum(waypoint_steps, dim=1)
        desired_speed = self.min_desired_speed + (
            self.max_speed - self.min_desired_speed
        ) * torch.sigmoid(self.speed_head(h_new))

        return waypoints, desired_speed, h_new
