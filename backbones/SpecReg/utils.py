"""Loss helpers retained from the original SpecReg implementation."""

import torch


def calculate_gradient_penalty(critic, x_src, x_tgt):
    x = torch.cat([x_src, x_tgt], dim=0).requires_grad_(True)
    x_out = critic(x)
    grad_out = torch.ones(x_out.shape, requires_grad=False).to(x_out.device)

    gradient = torch.autograd.grad(
        outputs=x_out,
        inputs=x,
        grad_outputs=grad_out,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    return torch.mean((gradient.norm(2, dim=1) - 1) ** 2)
