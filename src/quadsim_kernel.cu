#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

namespace {

template <typename scalar_t>
__global__ void render_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> canvas,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> flow,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R_old,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos_old,
    float drone_radius,
    int n_drones_per_group,
    float fov_x_half_tan) {

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = canvas.size(0);
    const int H = canvas.size(1);
    const int W = canvas.size(2);
    if (c >= B * H * W) return;
    const int b = c / (H * W);
    const int u = (c % (H * W)) / W;
    const int v = c % W;
    const scalar_t fov_y_half_tan = fov_x_half_tan / W * H;
    const scalar_t fu = (2 * (u + 0.5) / H - 1) * fov_y_half_tan - 1e-5;
    const scalar_t fv = (2 * (v + 0.5) / W - 1) * fov_x_half_tan - 1e-5;
    scalar_t dx = R[b][0][0] - fu * R[b][0][2] - fv * R[b][0][1];
    scalar_t dy = R[b][1][0] - fu * R[b][1][2] - fv * R[b][1][1];
    scalar_t dz = R[b][2][0] - fu * R[b][2][2] - fv * R[b][2][1];
    const scalar_t ox = pos[b][0];
    const scalar_t oy = pos[b][1];
    const scalar_t oz = pos[b][2];

    scalar_t min_dist = 100;
    scalar_t  t = (-1 - oz) / dz;
    if (t > 0) min_dist = t;

    // others
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue;
        scalar_t cx = pos[i][0];
        scalar_t cy = pos[i][1];
        scalar_t cz = pos[i][2];
        scalar_t r = 0.15;
        // (ox + t dx)^2 + (oy + t dy)^2 + 4 (oz + t dz)^2 = r^2
        scalar_t a = dx * dx + dy * dy + 4 * dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + 4 * dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // balls
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0];
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];
        scalar_t a = dx * dx + dy * dy + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // cylinders
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        scalar_t h = cylinders[batch_base][i][3];
        scalar_t a = dx * dx + dy * dy;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) - r * r;
        scalar_t d = b * b - 4 * a * c;
        // Side wall: accept a quadratic root only when its hit point lies
        // between the ground plane and this cylinder's sampled top.
        if (a > 1e-12 && d >= 0) {
            scalar_t sqrt_d = sqrt(d);
            scalar_t t_side_1 = (-b - sqrt_d) / (2 * a);
            scalar_t z_side_1 = oz + t_side_1 * dz;
            if (t_side_1 > 1e-5 && z_side_1 >= 0 && z_side_1 <= h)
                min_dist = min(min_dist, t_side_1);
            scalar_t t_side_2 = (-b + sqrt_d) / (2 * a);
            scalar_t z_side_2 = oz + t_side_2 * dz;
            if (t_side_2 > 1e-5 && z_side_2 >= 0 && z_side_2 <= h)
                min_dist = min(min_dist, t_side_2);
        }
        // Closed circular caps at z=0 and z=h.
        if (abs(dz) > 1e-12) {
            scalar_t t_cap_bottom = -oz / dz;
            scalar_t x_bottom = ox + t_cap_bottom * dx - cx;
            scalar_t y_bottom = oy + t_cap_bottom * dy - cy;
            if (t_cap_bottom > 1e-5 &&
                    x_bottom * x_bottom + y_bottom * y_bottom <= r * r)
                min_dist = min(min_dist, t_cap_bottom);

            scalar_t t_cap_top = (h - oz) / dz;
            scalar_t x_top = ox + t_cap_top * dx - cx;
            scalar_t y_top = oy + t_cap_top * dy - cy;
            if (t_cap_top > 1e-5 &&
                    x_top * x_top + y_top * y_top <= r * r)
                min_dist = min(min_dist, t_cap_top);
            }
    }
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        scalar_t a = dx * dx + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // voxels
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t rx = voxels[batch_base][i][3];
        scalar_t ry = voxels[batch_base][i][4];
        scalar_t rz = voxels[batch_base][i][5];
        scalar_t tx1 = (cx - rx - ox) / dx;
        scalar_t tx2 = (cx + rx - ox) / dx;
        scalar_t tx_min = min(tx1, tx2);
        scalar_t tx_max = max(tx1, tx2);
        scalar_t ty1 = (cy - ry - oy) / dy;
        scalar_t ty2 = (cy + ry - oy) / dy;
        scalar_t ty_min = min(ty1, ty2);
        scalar_t ty_max = max(ty1, ty2);
        scalar_t tz1 = (cz - rz - oz) / dz;
        scalar_t tz2 = (cz + rz - oz) / dz;
        scalar_t tz_min = min(tz1, tz2);
        scalar_t tz_max = max(tz1, tz2);
        scalar_t t_min = max(max(tx_min, ty_min), tz_min);
        scalar_t t_max = min(min(tx_max, ty_max), tz_max);
        if (t_min < min_dist && t_min < t_max && t_min > 0)
            min_dist = t_min;
    }

    canvas[b][u][v] = min_dist;
}

template <typename scalar_t>
__global__ void nearest_pt_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> nearest_pt,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pos,
    float drone_radius,
    int n_drones_per_group) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = nearest_pt.size(1);
    const int j = idx / B;
    if (j >= nearest_pt.size(0)) return;
    const int b = idx % B;

    const scalar_t ox = pos[j][b][0];
    const scalar_t oy = pos[j][b][1];
    const scalar_t oz = pos[j][b][2];

    scalar_t min_dist = 10.0f; 
    
    scalar_t nearest_ptx = ox + 10.0f;
    scalar_t nearest_pty = oy;
    scalar_t nearest_ptz = oz;

    // others
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue;
        scalar_t cx = pos[j][i][0];
        scalar_t cy = pos[j][i][1];
        scalar_t cz = pos[j][i][2];
        scalar_t r = 0.15;
        scalar_t dx = cx - ox, dy = cy - oy, dz = cz - oz;
        scalar_t d_sq = dx*dx + dy*dy + 4*dz*dz;
        scalar_t d_val = sqrt(d_sq);
        scalar_t dist = d_val - r;
        if (dist < min_dist) {
            min_dist = dist;
            scalar_t inv_d = (d_val > 1e-6f) ? (1.0f / d_val) : 0.0f;
            nearest_ptx = ox + dist * (cx - ox) * inv_d;
            nearest_pty = oy + dist * (cy - oy) * inv_d;
            nearest_ptz = oz + dist * (cz - oz) * inv_d;
        }
    }

    // balls
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0];
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];
        scalar_t dist_sq = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz);
        scalar_t d_val = sqrt(dist_sq);
        scalar_t dist = d_val - r; 
        if (dist < min_dist) {
            min_dist = dist;
            scalar_t inv_d = (d_val > 1e-6f) ? (1.0f / d_val) : 0.0f;
            nearest_ptx = ox + dist * (cx - ox) * inv_d; 
            nearest_pty = oy + dist * (cy - oy) * inv_d;
            nearest_ptz = oz + dist * (cz - oz) * inv_d;
        }
    }

    // cylinders
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        scalar_t dist_sq = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy);
        scalar_t d_val = sqrt(dist_sq);
        scalar_t dist = d_val - r; 
        
        if (dist < min_dist) { 
            min_dist = dist;
            scalar_t inv_d = (d_val > 1e-6f) ? (1.0f / d_val) : 0.0f;
            nearest_ptx = ox + dist * (cx - ox) * inv_d;
            nearest_pty = oy + dist * (cy - oy) * inv_d;
            nearest_ptz = oz;
        }
    }
    
    // cylinders_h 
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        scalar_t dist = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz);
        scalar_t d_val = sqrt(dist);
        scalar_t dist_val = d_val - r; 
        if (dist_val < min_dist) {
            min_dist = dist_val;
            scalar_t inv_d = (d_val > 1e-6f) ? (1.0f / d_val) : 0.0f;
            nearest_ptx = ox + dist_val * (cx - ox) * inv_d;
            nearest_pty = oy;
            nearest_ptz = oz + dist_val * (cz - oz) * inv_d;
        }
    }

    // voxels
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t rx = voxels[batch_base][i][3];
        scalar_t ry = voxels[batch_base][i][4];
        scalar_t rz = voxels[batch_base][i][5];

        scalar_t dx = abs(ox - cx) - rx;
        scalar_t dy = abs(oy - cy) - ry;
        scalar_t dz = abs(oz - cz) - rz;

        scalar_t ext_dist = sqrt(max(dx, 0.f)*max(dx, 0.f) + 
                                max(dy, 0.f)*max(dy, 0.f) + 
                                max(dz, 0.f)*max(dz, 0.f));
        scalar_t int_dist = min(max(dx, max(dy, dz)), 0.f);
        
        scalar_t dist = ext_dist + int_dist;

        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = cx + max(-rx, min(rx, ox - cx));
            nearest_pty = cy + max(-ry, min(ry, oy - cy));
            nearest_ptz = cz + max(-rz, min(rz, oz - cz));
        }
    }
    
    nearest_pt[j][b][0] = nearest_ptx;
    nearest_pt[j][b][1] = nearest_pty;
    nearest_pt[j][b][2] = nearest_ptz;
}


template <typename scalar_t>
__global__ void nearest_pt_backward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> d_pos,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> d_nearest_pt,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pos,
    float drone_radius,
    int n_drones_per_group) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = pos.size(1);
    const int j = idx / B;
    if (j >= pos.size(0)) return;
    const int b = idx % B;

    const scalar_t ox = pos[j][b][0];
    const scalar_t oy = pos[j][b][1];
    const scalar_t oz = pos[j][b][2];

    // ---- 重新寻找最近障碍物 (与 forward 完全一致) ----
    scalar_t min_dist = 10.0f;
    int winner_type = -1;  // 0=others, 1=ball, 2=cyl, 3=cyl_h, 4=voxel
    // 障碍物参数缓存
    scalar_t w_cx = 0, w_cy = 0, w_cz = 0, w_r = 0;
    scalar_t w_rx = 0, w_ry = 0, w_rz = 0;  // for voxels
    scalar_t w_alpha = 0, w_d = 0;           // α = d - r, d = weighted distance
    scalar_t w_dx = 0, w_dy = 0, w_dz = 0;   // direction vector from pos to center
    scalar_t w_wx = 1, w_wy = 1, w_wz = 1;   // distance weights

    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;

    // --- others (ellipsoid: z-weight=4) ---
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue;
        scalar_t cx = pos[j][i][0];
        scalar_t cy = pos[j][i][1];
        scalar_t cz = pos[j][i][2];
        scalar_t r = 0.15;
        scalar_t dx = cx - ox, dy = cy - oy, dz = cz - oz;
        scalar_t d_sq = dx*dx + dy*dy + 4*dz*dz;
        scalar_t d_val = sqrt(d_sq);
        scalar_t dist = d_val - r;
        if (dist < min_dist) {
            min_dist = dist;
            winner_type = 0;
            w_cx = cx; w_cy = cy; w_cz = cz; w_r = r;
            w_alpha = dist; w_d = d_val;
            w_dx = dx; w_dy = dy; w_dz = dz;
            w_wx = 1; w_wy = 1; w_wz = 4;
        }
    }

    // --- balls ---
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0];
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];
        scalar_t dx = cx - ox, dy = cy - oy, dz = cz - oz;
        scalar_t d_sq = dx*dx + dy*dy + dz*dz;
        scalar_t d_val = sqrt(d_sq);
        scalar_t dist = d_val - r;
        if (dist < min_dist) {
            min_dist = dist;
            winner_type = 1;
            w_cx = cx; w_cy = cy; w_cz = cz; w_r = r;
            w_alpha = dist; w_d = d_val;
            w_dx = dx; w_dy = dy; w_dz = dz;
            w_wx = 1; w_wy = 1; w_wz = 1;
        }
    }

    // --- cylinders (vertical, xy-plane) ---
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        scalar_t dx = cx - ox, dy = cy - oy;
        scalar_t d_sq = dx*dx + dy*dy;
        scalar_t d_val = sqrt(d_sq);
        scalar_t dist = d_val - r;
        if (dist < min_dist) {
            min_dist = dist;
            winner_type = 2;
            w_cx = cx; w_cy = cy; w_cz = 0; w_r = r;
            w_alpha = dist; w_d = d_val;
            w_dx = dx; w_dy = dy; w_dz = 0;
            w_wx = 1; w_wy = 1; w_wz = 0;
        }
    }

    // --- cylinders_h (horizontal, xz-plane) ---
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        scalar_t dx = cx - ox, dz = cz - oz;
        scalar_t d_sq = dx*dx + dz*dz;
        scalar_t d_val = sqrt(d_sq);
        scalar_t dist = d_val - r;
        if (dist < min_dist) {
            min_dist = dist;
            winner_type = 3;
            w_cx = cx; w_cy = 0; w_cz = cz; w_r = r;
            w_alpha = dist; w_d = d_val;
            w_dx = dx; w_dy = 0; w_dz = dz;
            w_wx = 1; w_wy = 0; w_wz = 1;
        }
    }

    // --- voxels ---
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t rx = voxels[batch_base][i][3];
        scalar_t ry = voxels[batch_base][i][4];
        scalar_t rz = voxels[batch_base][i][5];

        scalar_t ax = abs(ox - cx) - rx;
        scalar_t ay = abs(oy - cy) - ry;
        scalar_t az = abs(oz - cz) - rz;

        scalar_t ext_dist = sqrt(max(ax, 0.f)*max(ax, 0.f) + 
                                max(ay, 0.f)*max(ay, 0.f) + 
                                max(az, 0.f)*max(az, 0.f));
        scalar_t int_dist = min(max(ax, max(ay, az)), 0.f);
        scalar_t dist = ext_dist + int_dist;

        if (dist < min_dist) {
            min_dist = dist;
            winner_type = 4;
            w_cx = cx; w_cy = cy; w_cz = cz;
            w_rx = rx; w_ry = ry; w_rz = rz;
        }
    }

    // ---- 根据 winner_type 计算梯度 ----
    scalar_t dnx = d_nearest_pt[j][b][0];
    scalar_t dny = d_nearest_pt[j][b][1];
    scalar_t dnz = d_nearest_pt[j][b][2];
    scalar_t dpx = 0, dpy = 0, dpz = 0;

    if (winner_type == 0 || winner_type == 1) {
        scalar_t inv_d = (w_d > 1e-8f) ? (1.0f / w_d) : 0.0f;
        scalar_t ux = w_dx * inv_d;
        scalar_t uy = w_dy * inv_d;
        scalar_t uz = w_dz * inv_d;
        scalar_t dot_dn_u = dnx * ux + dny * uy + dnz * uz;
        scalar_t scale = w_r * inv_d;
        dpx = scale * (dnx - ux * dot_dn_u);
        dpy = scale * (dny - uy * dot_dn_u);
        dpz = scale * (dnz - uz * dot_dn_u);
    } else if (winner_type == 2) {
        scalar_t inv_d = (w_d > 1e-8f) ? (1.0f / w_d) : 0.0f;
        scalar_t ux = w_dx * inv_d;
        scalar_t uy = w_dy * inv_d;
        scalar_t dot_dn_u = dnx * ux + dny * uy;
        scalar_t scale = w_r * inv_d;
        dpx = scale * (dnx - ux * dot_dn_u);
        dpy = scale * (dny - uy * dot_dn_u);
        dpz = dnz;
    } else if (winner_type == 3) {
        scalar_t inv_d = (w_d > 1e-8f) ? (1.0f / w_d) : 0.0f;
        scalar_t ux = w_dx * inv_d;
        scalar_t uz = w_dz * inv_d;
        scalar_t dot_dn_u = dnx * ux + dnz * uz;
        scalar_t scale = w_r * inv_d;
        dpx = scale * (dnx - ux * dot_dn_u);
        dpy = dny;
        dpz = scale * (dnz - uz * dot_dn_u);
    } else if (winner_type == 4) {
        // --- Voxel: ∂npx/∂ox = 1 if |ox-cx| < rx else 0 ---
        dpx = (abs(ox - w_cx) < w_rx) ? dnx : 0.0f;
        dpy = (abs(oy - w_cy) < w_ry) ? dny : 0.0f;
        dpz = (abs(oz - w_cz) < w_rz) ? dnz : 0.0f;
    }
    // winner_type == -1: 无障碍物，d_pos = 0 (初始值)

    d_pos[j][b][0] = dpx;
    d_pos[j][b][1] = dpy;
    d_pos[j][b][2] = dpz;
}


template <typename scalar_t>
__global__ void rerender_backward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> depth,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dddp,
    float fov_x_half_tan) {

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = dddp.size(0);
    const int H = dddp.size(2);
    const int W = dddp.size(3);
    if (c >= B * H * W) return;
    const int b = c / (H * W);
    const int u = (c % (H * W)) / W;
    const int v = c % W;

    const scalar_t unit = fov_x_half_tan / W;
    const scalar_t d = (depth[b][0][u*2][v*2] + depth[b][0][u*2+1][v*2] + depth[b][0][u*2][v*2+1] + depth[b][0][u*2+1][v*2+1]) / 4 * unit;
    const scalar_t dddy = (depth[b][0][u*2][v*2] + depth[b][0][u*2+1][v*2] - depth[b][0][u*2][v*2+1] - depth[b][0][u*2+1][v*2+1]) / 2 / d;
    const scalar_t dddz = (depth[b][0][u*2][v*2] - depth[b][0][u*2+1][v*2] + depth[b][0][u*2][v*2+1] - depth[b][0][u*2+1][v*2+1]) / 2 / d;

    const scalar_t dddp_norm = max(8., sqrt(1 + dddy * dddy + dddz * dddz));
    dddp[b][0][u][v] = -1. / dddp_norm;
    dddp[b][1][u][v] = dddy / dddp_norm;
    dddp[b][2][u][v] = dddz / dddp_norm;
}

} // namespace

void render_cuda(
    torch::Tensor canvas,
    torch::Tensor flow,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor R_old,
    torch::Tensor pos,
    torch::Tensor pos_old,
    float drone_radius,
    int n_drones_per_group,
    float fov_x_half_tan) {

    AT_DISPATCH_FLOATING_TYPES(canvas.scalar_type(), "render_cuda", ([&] {
        int minGridSize, blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, render_cuda_kernel<scalar_t>, 0, 0);
        

        size_t state_size = canvas.numel();

        const dim3 blocks((state_size + blockSize - 1) / blockSize);

        render_cuda_kernel<scalar_t><<<blocks, blockSize>>>(
            canvas.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            flow.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R_old.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pos_old.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            drone_radius,
            n_drones_per_group,
            fov_x_half_tan);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }));
}

void rerender_backward_cuda(
    torch::Tensor depth,
    torch::Tensor dddp,
    float fov_x_half_tan) {

    AT_DISPATCH_FLOATING_TYPES(depth.scalar_type(), "rerender_backward_cuda", ([&] {
        int minGridSize, blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, rerender_backward_cuda_kernel<scalar_t>, 0, 0);
        
        size_t state_size = dddp.numel();
        const dim3 blocks((state_size + blockSize - 1) / blockSize);

        rerender_backward_cuda_kernel<scalar_t><<<blocks, blockSize>>>(
            depth.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            dddp.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            fov_x_half_tan);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }));
}

void find_nearest_pt_cuda(
    torch::Tensor nearest_pt,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos,
    float drone_radius,
    int n_drones_per_group) {
    
    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "nearest_pt_cuda", ([&] {
        int minGridSize, blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, nearest_pt_cuda_kernel<scalar_t>, 0, 0);
        
        size_t state_size = pos.size(0) * pos.size(1);
        const dim3 blocks((state_size + blockSize - 1) / blockSize);
        
        nearest_pt_cuda_kernel<scalar_t><<<blocks, blockSize>>>(
            nearest_pt.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            drone_radius,
            n_drones_per_group);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }));
}

void find_nearest_pt_backward_cuda(
    torch::Tensor d_pos,
    torch::Tensor d_nearest_pt,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos,
    float drone_radius,
    int n_drones_per_group) {
    
    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "nearest_pt_backward_cuda", ([&] {
        int minGridSize, blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, nearest_pt_backward_cuda_kernel<scalar_t>, 0, 0);
        
        size_t state_size = pos.size(0) * pos.size(1);
        const dim3 blocks((state_size + blockSize - 1) / blockSize);
        
        nearest_pt_backward_cuda_kernel<scalar_t><<<blocks, blockSize>>>(
            d_pos.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            d_nearest_pt.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            drone_radius,
            n_drones_per_group);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }));
}
