#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath> 

namespace {

template <typename scalar_t>
__global__ void update_state_vec_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R_new,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> theta) {
    
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= R_new.size(0)) return;

    // theta format: [x, y, yaw]
    scalar_t yaw = theta[b][2]; 
    
    scalar_t c = cos(yaw);
    scalar_t s = sin(yaw);

    // Forward Vector (X-axis): [cos, sin, 0]
    R_new[b][0][0] = c;   
    R_new[b][1][0] = s;   
    R_new[b][2][0] = 0;   

    // Left Vector (Y-axis): [-sin, cos, 0]
    R_new[b][0][1] = -s;  
    R_new[b][1][1] = c;   
    R_new[b][2][1] = 0;   

    // Up Vector (Z-axis): [0, 0, 1]
    R_new[b][0][2] = 0;   
    R_new[b][1][2] = 0;   
    R_new[b][2][2] = 1;   
}

template <typename scalar_t>
__global__ void run_forward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> p,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> theta,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> omega,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> p_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> theta_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> omega_next,
    float ctl_dt) {
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.size(0)) return;

    scalar_t cos_theta = cos(theta[i][2]); 
    scalar_t sin_theta = sin(theta[i][2]);
    scalar_t v_curr = v[i][0]; 


    p_next[i][0] = p[i][0] + v_curr * cos_theta * ctl_dt;
    p_next[i][1] = p[i][1] + v_curr * sin_theta * ctl_dt;
    
    theta_next[i][0] = theta[i][0]; 
    theta_next[i][1] = theta[i][1];
    theta_next[i][2] = theta[i][2] + omega[i][2] * ctl_dt; 


    v_next[i][0] = v[i][0];
    
    omega_next[i][0] = omega[i][0];
    omega_next[i][1] = omega[i][1];
    omega_next[i][2] = omega[i][2];
}

template <typename scalar_t>
__global__ void run_backward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> theta,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_p,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_v,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_theta,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_omega, 
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_p_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_v_next,   
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_theta_next,
    float grad_decay,
    float ctl_dt) {
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= v.size(0)) return;

    scalar_t cos_theta = cos(theta[i][2]);
    scalar_t sin_theta = sin(theta[i][2]);
    scalar_t v_val = v[i][0];
    
    scalar_t decay = exp(-grad_decay * ctl_dt); 
    
    d_p[i][0] += d_p_next[i][0] * decay;
    d_p[i][1] += d_p_next[i][1] * decay;

    scalar_t grad_from_pos_x = d_p_next[i][0] * cos_theta * ctl_dt;
    scalar_t grad_from_pos_y = d_p_next[i][1] * sin_theta * ctl_dt;
    
    d_v[i][0] += grad_from_pos_x + grad_from_pos_y + d_v_next[i][0] * decay;



    d_theta[i][0] += d_theta_next[i][0] * decay; 
    d_theta[i][1] += d_theta_next[i][1] * decay;



    scalar_t grad_from_theta_next = d_theta_next[i][2] * decay;
    scalar_t grad_from_pos_x_th = d_p_next[i][0] * (-v_val * sin_theta * ctl_dt);
    scalar_t grad_from_pos_y_th = d_p_next[i][1] * ( v_val * cos_theta * ctl_dt);

    d_theta[i][2] += grad_from_theta_next + grad_from_pos_x_th + grad_from_pos_y_th;


    d_omega[i][2] += d_theta_next[i][2] * ctl_dt;
    
}

} 



std::vector<torch::Tensor> run_forward_cuda(
    torch::Tensor p,
    torch::Tensor v,
    torch::Tensor theta,
    torch::Tensor omega,
    float ctl_dt) {

    auto p_next = torch::empty_like(p);
    auto v_next = torch::empty_like(v);
    auto theta_next = torch::empty_like(theta);
    auto omega_next = torch::empty_like(omega);

    const int batch_size = p.size(0);
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "run_forward_cuda", ([&] {
        run_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            p.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            theta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            omega.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            p_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            theta_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            omega_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            ctl_dt);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }));

    return {p_next, v_next, theta_next, omega_next};
}

std::vector<torch::Tensor> run_backward_cuda(
    torch::Tensor v,             
    torch::Tensor theta,         
    torch::Tensor d_p_next,     
    torch::Tensor d_v_next,      
    torch::Tensor d_theta_next,
    float grad_decay,
    float ctl_dt) {


    auto d_p = torch::zeros_like(d_p_next);      
    auto d_v = torch::zeros_like(v);             
    auto d_theta = torch::zeros_like(theta);     
    auto d_omega = torch::zeros_like(theta);     

    const int batch_size = v.size(0);
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "run_backward_cuda", ([&] {
        run_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            theta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_p.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_theta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_omega.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_p_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_v_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_theta_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            grad_decay, 
            ctl_dt);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }));

    return {d_p, d_v, d_theta, d_omega};
}

torch::Tensor update_state_vec_cuda(torch::Tensor theta) {
    const int batch_size = theta.size(0);
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    auto R_new = torch::empty({batch_size, 3, 3}, theta.options());

    AT_DISPATCH_FLOATING_TYPES(theta.scalar_type(), "update_state_vec_cuda", ([&] {
        update_state_vec_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R_new.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            theta.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }));

    return R_new;
}

