import torch
import quadsim_cuda 

# 动力学梯度测试脚本


def run_forward_pytorch(p, v, theta, omega, ctl_dt):

    cos_theta = torch.cos(theta[:, 2])
    sin_theta = torch.sin(theta[:, 2])
    v_curr = v[:, 0]
    

    p_next = p.clone()
    p_next[:, 0] = p[:, 0] + v_curr * cos_theta * ctl_dt
    p_next[:, 1] = p[:, 1] + v_curr * sin_theta * ctl_dt
    
    theta_next = theta.clone()
    theta_next[:, 2] = theta[:, 2] + omega[:, 2] * ctl_dt
    
    v_next = v.clone() 
    omega_next = omega.clone()
    
    return p_next, v_next, theta_next, omega_next


device = 'cuda'
B = 64
ctl_dt = 1/15.0


p = torch.randn((B, 2), device=device, requires_grad=True)
v = torch.randn((B, 1), device=device, requires_grad=True) 
theta = torch.randn((B, 3), device=device, requires_grad=True) 
omega = torch.randn((B, 3), device=device, requires_grad=True)

p_next_cuda, v_next_cuda, theta_next_cuda, omega_next_cuda = quadsim_cuda.run_forward(
    p, v, theta, omega, ctl_dt
)

p_next_py, v_next_py, theta_next_py, omega_next_py = run_forward_pytorch(
    p, v, theta, omega, ctl_dt
)

print("Check Forward...")
assert torch.allclose(p_next_cuda, p_next_py, atol=1e-5), "Pos mismatch"
assert torch.allclose(theta_next_cuda, theta_next_py, atol=1e-5), "Theta mismatch"
print("Forward PASS ✅")


d_p_next = torch.randn_like(p_next_cuda)
d_v_next = torch.randn_like(v_next_cuda)
d_theta_next = torch.randn_like(theta_next_cuda)


if p.grad is not None: p.grad.zero_()
if v.grad is not None: v.grad.zero_()
if theta.grad is not None: theta.grad.zero_()

torch.autograd.backward(
    (p_next_py, v_next_py, theta_next_py),
    (d_p_next, d_v_next, d_theta_next),
    retain_graph=True
)

grad_p_py = p.grad.clone()
grad_v_py = v.grad.clone()
grad_theta_py = theta.grad.clone()



grad_decay = 0.0 # 必须为 0（无衰减）才能与上方无衰减的 PyTorch 参考实现对比
d_p_cuda, d_v_cuda, d_theta_cuda, d_omega_cuda = quadsim_cuda.run_backward(
    v, theta, d_p_next, d_v_next, d_theta_next, grad_decay, ctl_dt
)

print("Check Backward...")
assert torch.allclose(d_p_cuda, grad_p_py, atol=1e-5), "Grad P mismatch"
assert torch.allclose(d_v_cuda, grad_v_py, atol=1e-5), "Grad V mismatch"
assert torch.allclose(d_theta_cuda, grad_theta_py, atol=1e-5), "Grad Theta mismatch"
print("Backward PASS ✅")