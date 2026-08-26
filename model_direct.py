import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, dim_obs=6, dim_action=2, hidden_dim=192, input_w=32, input_h=24):
        """
        input：depth_raw + state
        output：action (v, omega)
        """
        super().__init__()

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

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.05),
            nn.Linear(128, dim_action),
            nn.Tanh()
        )
        self.action_head[2].weight.data.mul_(0.01)
        
        # --- velocity limitation ---
        self.max_v = 4.0     
        self.max_omega = 3.0

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
        
        raw_action = self.action_head(h_new)
        
        # plan A: forward 
        v = (raw_action[:, 0:1] + 1.0) / 2.0 * self.max_v 
        
        # plan B 
        # v = raw_action[:, 0:1] * self.max_v 

        omega = raw_action[:, 1:2] * self.max_omega
        
        action = torch.cat([v, omega], dim=1)

        return action, h_new