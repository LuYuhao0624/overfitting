import torch
import torch.nn.functional as F


class PGD:
    def __call__(self, x, model, epsilon, num_steps, step_size, target, random):
        if random:
            x_adv = x - epsilon + 2 * epsilon * torch.rand_like(x)
        else:
            x_adv = torch.clone(x)
        x_adv = torch.clamp(x_adv, 0, 1)
        for _ in range(num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(model(x_adv), target)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(
                grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv
