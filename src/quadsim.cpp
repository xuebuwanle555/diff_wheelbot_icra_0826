#include <torch/extension.h>
#include <vector>

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
    float fov_x_half_tan);

void rerender_backward_cuda(
    torch::Tensor depth,
    torch::Tensor dddp,
    float fov_x_half_tan);

void find_nearest_pt_cuda(
    torch::Tensor nearest_pt,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos,
    float drone_radius,
    int n_drones_per_group);

void find_nearest_pt_backward_cuda(
    torch::Tensor d_pos,
    torch::Tensor d_nearest_pt,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos,
    float drone_radius,
    int n_drones_per_group);


torch::Tensor update_state_vec_cuda(
    torch::Tensor theta);

// forward： p, v, theta, omega
std::vector<torch::Tensor> run_forward_cuda(
    torch::Tensor p,
    torch::Tensor v,
    torch::Tensor theta,
    torch::Tensor omega,
    float ctl_dt);

// backward： v, theta 
std::vector<torch::Tensor> run_backward_cuda(
    torch::Tensor v,             
    torch::Tensor theta,         
    torch::Tensor d_p_next,     
    torch::Tensor d_v_next,      
    torch::Tensor d_theta_next,
    float grad_decay,
    float ctl_dt);




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("render", &render_cuda, "render (CUDA)");
  m.def("rerender_backward", &rerender_backward_cuda, "rerender_backward_cuda (CUDA)");
  m.def("find_nearest_pt", &find_nearest_pt_cuda, "find_nearest_pt (CUDA)");
  m.def("find_nearest_pt_backward", &find_nearest_pt_backward_cuda, "find_nearest_pt_backward (CUDA)");


  m.def("update_state_vec", &update_state_vec_cuda, "Update Rotation Matrix from Yaw (CUDA)");
  
  m.def("run_forward", &run_forward_cuda, "Differential Drive Forward Dynamics (CUDA)");
  
  m.def("run_backward", &run_backward_cuda, "Differential Drive Backward Dynamics (CUDA)");
}