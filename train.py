import argparse
import math
import numpy as np
from random import normalvariate
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from env_cuda import Env
from model import Model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_iters', type=int, default=30000)
    

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_decay', type=float, default=0.8) 
    parser.add_argument('--fov_x_half_tan', type=float, default=0.82) 
    parser.add_argument('--timesteps', type=int, default=120) 
    

    parser.add_argument('--coef_pos', type=float, default=1.0)
    parser.add_argument('--coef_v', type=float, default=1.0) 
    parser.add_argument('--coef_heading', type=float, default=0.5) 
    parser.add_argument('--coef_obj_avoidance', type=float, default=1.0)
    parser.add_argument('--coef_collide', type=float, default=3.5) 
    parser.add_argument('--coef_smooth', type=float, default=0.1) 
    # parser.add_argument('--coef_omega', type=float, default=0.1) 
    parser.add_argument('--coef_bias', type=float, default=0.5) 
    parser.add_argument('--coef_energy', type=float, default=0.05) 
    
    return parser.parse_args()

def barrier_loss(dist_to_obs, v_approach):
    safe_dist = 1.0
    coeff = 10.0
    # k = 5.0
    # mask = (dist_to_obs < safe_dist).float()
    # pennetration = F.relu(safe_dist - dist_to_obs)
    # potential = torch.clamp(torch.exp(k * pennetration) - 1.0, max=100.0) 
    dist_diff = (safe_dist - dist_to_obs) 
    potential = F.relu(dist_diff).pow(2)
    
    # return (potential * v_approach * mask).mean()
    return (potential * v_approach * coeff).mean()

def plot_trajectory(env, p_history, target_pos, i, writer, required_clearance=0.3):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if hasattr(env, 'cyl'):
        cylinders = env.cyl[0].detach().cpu().numpy()
        for obs in cylinders:
            circle = plt.Circle((obs[0], obs[1]), obs[2], color='gray', alpha=0.4)
            ax.add_artist(circle)
            circle_safe = plt.Circle((obs[0], obs[1]), obs[2] + required_clearance, color='red', fill=False, linestyle='--', alpha=0.2)
            ax.add_artist(circle_safe)

    if hasattr(env, 'balls'):
        balls = env.balls[0].detach().cpu().numpy()
        for b in balls:
            circle = plt.Circle((b[0], b[1]), b[3], color='skyblue', alpha=0.4)
            ax.add_artist(circle)

    if hasattr(env, 'voxels'):
        voxels = env.voxels[0].detach().cpu().numpy()
        for v in voxels:
            cx, cy = v[0], v[1]
            rx, ry = v[3], v[4]
            if rx >5.0 or ry >5.0:
                continue
            rect = patches.Rectangle((cx - rx, cy - ry), 2*rx, 2*ry, color='orange', alpha=0.4)
            ax.add_artist(rect)

    # draw trajectory
    traj = p_history[:, 0, :2].detach().cpu().numpy()
    target = target_pos[0, :2].detach().cpu().numpy()
    start_pos = traj[0]
    
    ax.plot(traj[:, 0], traj[:, 1], label='Path', linewidth=2, color='royalblue')
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=8, label='Start') 
    ax.scatter(target[0], target[1], c='red', marker='x', s=100, label='Target', zorder=10) 
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_title(f"iter {i+1}")


    all_x = np.concatenate([traj[:, 0], [target[0]], [start_pos[0]]])
    all_y = np.concatenate([traj[:, 1], [target[1]], [start_pos[1]]])
    

    margin = 2.0
    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, all_y.max() + margin
    
    span = max(x_max - x_min, y_max - y_min)
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    
    ax.set_xlim(x_mid - span/2, x_mid + span/2)
    ax.set_ylim(y_mid - span/2, y_mid + span/2)


    writer.add_figure('Trajectory/Batch0', fig, i + 1)
    plt.close(fig)

def main():
    args = parse_args()
    writer = SummaryWriter() 
    device = torch.device('cuda')

    env = Env(args.batch_size, 64, 48, args.grad_decay, device,
              fov_x_half_tan=args.fov_x_half_tan, ground_voxels=True)

    model = Model(dim_obs=6, dim_action=2, input_w=32, input_h=24).to(device)
    
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume))

    optim = AdamW(model.parameters(), args.lr)
    sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)
    
    scaler_q = defaultdict(list)
    pbar = tqdm(range(args.num_iters), ncols=100)
    B = args.batch_size

    AVG_DT = 1.0 / 15.0  

    for i in pbar:
        env.reset()
        model.reset()
        h = None
        

        p_hist_list, act_hist_list, vec_obs_hist_list = [], [], []
        theta_hist_list = [] 
        v_hist_list = [] 

        response_rate_k = torch.rand((args.batch_size, 2), device=device) * 5.0 + 3.0
        
        # Latent State
        motor_actual = torch.zeros((args.batch_size, 2), device=device)
        
        # === Rollout ===
        for t in range(args.timesteps):
            ctl_dt = max(0.01, normalvariate(1/15.0, 0.005))

            #原版
            # depth, _ = env.render(ctl_dt)
            # # depth_input = F.max_pool2d(depth, 2, 2)
            # depth_input = -F.max_pool2d(-depth, 2, 2)
            # noise = torch.randn_like(depth_input) * 0.02
            # depth_input = 3.0 / depth_input.clamp(min=0.2,max=10.0) - 0.6 + noise

            # 新版深度图处理
            depth, _ = env.render(ctl_dt)
            depth_inv = 3.0 / depth.clamp(min=0.2, max=10.0) - 0.6
            noise = torch.randn_like(depth_inv) * 0.02
            depth_noisy = depth_inv + noise
            depth_input = F.max_pool2d(depth_noisy, 2, 2)


            
            vec_global = env.p_target - env.p
            cos_th = torch.cos(env.theta[:, 2])
            sin_th = torch.sin(env.theta[:, 2])
            
            local_x = vec_global[:, 0] * cos_th + vec_global[:, 1] * sin_th
            local_y = vec_global[:, 0] * -sin_th + vec_global[:, 1] * cos_th
            dist_target = torch.sqrt(local_x**2 + local_y**2)
            scale_mask = dist_target > 10.0
            scale_factor = 10.0 / (dist_target + 1e-6)
            scale = torch.where(scale_mask, scale_factor, torch.ones_like(scale_factor))

            local_x = local_x * scale   
            local_y = local_y * scale
            dist_target = dist_target * scale 
            
            state = torch.stack([
                local_x / 10.0, 
                local_y / 10.0, 
                cos_th, 
                sin_th, 
                dist_target / 10.0, 
                env.v[:, 0]
            ], dim=1)
            
            action_cmd, h = model(depth_input, state, h)
            
            motor_alpha = torch.exp(-response_rate_k * ctl_dt)
            motor_actual = motor_alpha * motor_actual + (1.0 - motor_alpha) * action_cmd

            env.run(motor_actual, ctl_dt)
            
            p_hist_list.append(env.p.clone())
            theta_hist_list.append(env.theta.clone())
            v_hist_list.append(env.v.clone()) 
            act_hist_list.append(action_cmd)
            vec_obs_hist_list.append(env.find_vec_to_nearest_pt())

        # === Loss Calculation ===
        p_stack = torch.stack(p_hist_list)          # (TimeStep, B, 3)
        theta_stack = torch.stack(theta_hist_list)  # (T, B, 3)
        v_stack = torch.stack(v_hist_list)          # (T, B, 1)
        act_stack = torch.stack(act_hist_list)      # (T, B, 2)
        vec_obs_stack = torch.stack(vec_obs_hist_list) # (T, B, 3)

        target_expanded = env.p_target.unsqueeze(0).expand(args.timesteps, -1, -1)
        
        vec_to_target_all = target_expanded[..., :2] - p_stack[..., :2]
        dist_all_steps = torch.norm(vec_to_target_all + 1e-8, dim=-1)
        dist_to_target_vec = dist_all_steps.unsqueeze(-1) 
        dir_to_target_all = F.normalize(vec_to_target_all, dim=-1)
        

        cur_yaw_all = theta_stack[..., 2]
        cur_dir_all = torch.stack([torch.cos(cur_yaw_all), torch.sin(cur_yaw_all)], dim=-1)

        # ---------------------------------------------------------------------
        # 1. Pos Loss 
        # ---------------------------------------------------------------------
        arrival_tolerance = 1.0 
        loss_pos = F.relu(dist_all_steps - arrival_tolerance).mean()  

        # ---------------------------------------------------------------------
        # 2. Heading Loss 
        # ---------------------------------------------------------------------
        mask_heading = (dist_all_steps > arrival_tolerance).float()
        
        raw_heading_loss = 1.0 - F.cosine_similarity(cur_dir_all, dir_to_target_all, dim=-1)
        loss_heading = (raw_heading_loss * mask_heading).sum() / (mask_heading.sum() + 1e-5)

	# ---------------------------------------------------------------------
        # 3. Velocity Vector Loss (修正后)
        # ---------------------------------------------------------------------
        # 计算基于距离的目标速度，并【必须加上 .detach() 截断梯度】
        target_speed_scalar = torch.clamp((dist_to_target_vec - arrival_tolerance) * 2.0, 0.0, 4.0).detach()
        
        current_speed_scalar = v_stack
        
        # 【NMI 顶刊优化】：不再进行逐帧的严苛束缚，而是比较整段轨迹的“平均目标速度”和“平均当前速度”
        # 允许小车在过程中的某些帧为了避障紧急刹车，只要整体宏观速度达标即可
        mean_target_speed = target_speed_scalar.mean(dim=0)  # [B, 1]
        mean_current_speed = current_speed_scalar.mean(dim=0) # [B, 1]
        
        loss_velocity_scalar = F.smooth_l1_loss(mean_current_speed, mean_target_speed)
        
        # 可选：如果你怕它瞬间加减速太夸张，可以保留一个很小权重的逐帧 Loss 兜底
        # loss_velocity_scalar += 0.1 * F.smooth_l1_loss(current_speed_scalar, target_speed_scalar)
        # ---------------------------------------------------------------------
        # 4. avoid loss 
        # ---------------------------------------------------------------------
        dist_obs = torch.norm(vec_obs_stack[..., :2], dim=-1)
        pad = dist_obs[0:1]
        v_approach = (-torch.diff(dist_obs, n=1, dim=0, prepend=pad) / AVG_DT).clamp_min(1)
        loss_avoid = barrier_loss(dist_obs, v_approach)
        
        # ---------------------------------------------------------------------
        # 5. collide loss 
        # ---------------------------------------------------------------------
        radius = 0.3
        dist_diff = radius - dist_obs
        loss_collide = F.softplus(dist_diff * 10.0).mean() * 2.0
        
        # ---------------------------------------------------------------------
        # 6.smooth loss 
        # ---------------------------------------------------------------------
        # loss_smooth = act_stack.diff(1, 0).pow(2).mean()
        
        loss_smooth = (act_stack.diff(1, 0) / AVG_DT).pow(2).mean()

        v_real_vec = torch.stack([
            v_stack[..., 0] * torch.cos(theta_stack[..., 2]),
            v_stack[..., 0] * torch.sin(theta_stack[..., 2])
        ], dim=-1)

        v_proj_val = (v_real_vec * dir_to_target_all).sum(dim=-1, keepdim=True)
        v_proj_vec = v_proj_val * dir_to_target_all
        loss_bias = F.mse_loss(v_real_vec, v_proj_vec)
        
        # ---------------------------------------------------------------------
        # 8.  loss energy
        # ---------------------------------------------------------------------
        loss_action_energy = act_stack.pow(2).mean()

        # === Total Loss ===
        total_loss = args.coef_pos * loss_pos + \
                     args.coef_heading * loss_heading + \
                     args.coef_v * loss_velocity_scalar + \
                     args.coef_obj_avoidance * loss_avoid + \
                     args.coef_collide * loss_collide + \
                     args.coef_smooth * loss_smooth + \
                     args.coef_bias * loss_bias + \
                    args.coef_energy * loss_action_energy

        if torch.isnan(total_loss):
            print(f"Warning: NaN detected at iter {i}. Skipping this step to protect training.")
            optim.zero_grad() 
            
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
                    
            continue 
            
        optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()
        
        with torch.no_grad():
            scaler_q['Loss/Total'].append(total_loss.item())
            scaler_q['Loss/Pos'].append(loss_pos.item())
            scaler_q['Loss/Heading'].append(loss_heading.item())
            scaler_q['Loss/Velocity'].append(loss_velocity_scalar.item()) 
            scaler_q['Loss/Collide'].append(loss_collide.item())
            scaler_q['Metric/FinalDist'].append(dist_all_steps[-1].mean().item())
            scaler_q['Loss/Avoid'].append(loss_avoid.item())
            scaler_q['Loss/Smooth'].append(loss_smooth.item())
            scaler_q['Loss/Bias'].append(loss_bias.item())
            scaler_q['loss/Energy'].append(loss_action_energy.item())
            # scaler_q['Loss/Omega'].append(loss_action_energy.item())

            if (i + 1) % 25 == 0:
                for k, v in scaler_q.items():
                    writer.add_scalar(k, sum(v) / len(v), i + 1)
                scaler_q.clear()
            
            pbar.set_description(f"L:{total_loss.item():.2f}|P:{loss_pos.item():.2f}|H:{loss_heading.item():.2f}")

            if (i + 1) % 1000 == 0:
                plot_trajectory(env, p_stack, env.p_target, i, writer)
                
            if (i + 1) % 5000 == 0:
                torch.save(model.state_dict(), f'checkpoint_base11_{i+1}.pth')

if __name__ == '__main__':
    main()
