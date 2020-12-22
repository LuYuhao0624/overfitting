import torch
import torch.nn.functional as F
from attacks import PGD


class Loss:
    def __init__(self, args):
        self.args = args

    def __call__(self, model, x, y, random=True):
        model.eval()
        target = self.preprocess(model, x, y)
        advs = self.generate_adv(model, x, y, target, random)
        model.train()
        params, metric_dict = self.postprocess(model, x, y, advs)
        loss, loss_terms = self.compute_loss(model, params)
        metric_dict.update(loss_terms)
        return loss, metric_dict

    def preprocess(self, model, x, y):
        raise NotImplementedError("This function should be invoked in "
                                  "sub-classes only.")

    def generate_adv(self, model, x, y, target, random):
        x_adv = PGD()(
            x, model, self.args.train_epsilon, self.args.train_num_steps,
            self.args.train_step_size, target, random
            )
        return x_adv

    def postprocess(self, model, x, y, x_adv):
        x_logit = model(x)
        x_plabel = torch.max(x_logit, dim=1)[1]
        x_adv_logit = model(x_adv)
        x_adv_plabel = torch.max(x_adv_logit, dim=1)[1]
        accurate = (x_plabel == y).float().sum()
        stable = (x_adv_plabel == x_plabel).float().sum()
        robust = (x_adv_plabel == y).float().sum()
        return [x_logit, y, x_plabel, x_adv_logit], {
            "accurate": accurate,
            "stable": stable,
            "robust": robust
            }

    def compute_loss(self, model, params):
        raise NotImplementedError("This function should be invoked in "
                                  "sub-classes only.")


class AccurateStableLoss(Loss):
    def preprocess(self, model, x, y):
        # get target for generating perturbation in loss function
        x_logit = model(x)
        x_plabel = torch.max(x_logit, dim=1)[1]
        return x_plabel

    def compute_loss(self, model, params):
        x_logit, y, x_plabel, x_adv_logit = params
        loss_accurate = F.cross_entropy(x_logit, y)
        loss_stable = F.cross_entropy(x_adv_logit, x_plabel)
        loss = loss_accurate + self.args.beta * loss_stable
        return loss, {
            "l_a": loss_accurate,
            "l_s": loss_stable
            }


class RobustLoss(Loss):
    def preprocess(self, model, x, y):
        # get target for generating perturbation in loss function
        return y

    def compute_loss(self, model, params):
        x_logit, y, x_plabel, x_adv_logit = params
        loss_accurate = F.cross_entropy(x_logit, y)
        loss_stable = F.cross_entropy(x_adv_logit, x_plabel)
        loss = F.cross_entropy(x_adv_logit, y)
        return loss, {
            "l_a": loss_accurate,
            "l_s": loss_stable
            }


class MultiTaskLoss(Loss):
    def preprocess(self, model, x, y):
        target_acc = y
        x_logit = model(x)
        target_sta = torch.max(x_logit, dim=1)[1]
        return target_acc, target_sta

    def generate_adv(self, model, x, y, target, random):
        target_acc, target_sta = target
        x_acc = PGD()(
            x, model, self.args.train_epsilon, self.args.train_num_steps,
            self.args.train_step_size, target_acc, random
            )
        x_sta = PGD()(
            x, model, self.args.train_epsilon, self.args.train_num_steps,
            self.args.train_step_size, target_sta, random
            )
        x_logit = model(x)
        x_plabel = target_sta
        x_acc_logit = model(x_acc)
        x_sta_logit = model(x_sta)
        x_acc_loss = self._as_loss(x_logit, y, x_acc_logit, x_plabel)
        x_sta_loss = self._as_loss(x_logit, y, x_sta_logit, x_plabel)
        x_adv = torch.zeros_like(x_acc)
        gt = x_acc_loss > x_sta_loss
        x_adv[gt] = x_acc[gt]
        x_adv[~gt] = x_sta[~gt]
        return x_adv

    def compute_loss(self, model, params):
        x_logit, y, x_plabel, x_adv_logit = params
        loss_accurate = F.cross_entropy(x_logit, y)
        loss_stable = F.cross_entropy(x_adv_logit, x_plabel)
        loss = loss_accurate + self.args.beta * loss_stable
        return loss, {
            "l_a": loss_accurate,
            "l_s": loss_stable
            }

    def _as_loss(self, x_logit, y, x_adv_logit, x_plabel):
        loss_accurate = F.cross_entropy(x_logit, y, reduction="none")
        loss_stable = F.cross_entropy(x_adv_logit, x_plabel, reduction="none")
        return loss_accurate + self.args.beta * loss_stable




#
# def accurate_stable_loss(model, args, x_clean, label):
#     model.eval()
#     x_adv = x_clean + 0.001 * torch.randn(x_clean.shape).cuda()
#     x_adv = torch.clamp(x_adv, 0, 1)
#     x_clean_logit = model(x_clean)
#     x_clean_plabel = torch.max(x_clean_logit, dim=1)[1]
#     # generate adv
#     for _ in range(args.train_num_steps):
#         x_adv.requires_grad_()
#         with torch.enable_grad():
#             loss_adv = F.cross_entropy(model(x_adv), x_clean_plabel)
#         grad = torch.autograd.grad(loss_adv, [x_adv])[0]
#         x_adv = x_adv.detach() + args.train_step_size * torch.sign(
#             grad.detach())
#         x_adv = torch.min(
#             torch.max(x_adv, x_clean - args.train_epsilon),
#             x_clean + args.train_epsilon
#             )
#         x_adv = torch.clamp(x_adv, 0.0, 1.0)
#
#     model.train()
#
#     x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
#
#     x_clean_logit = model(x_clean)
#     x_clean_plabel = torch.max(x_clean_logit, dim=1)[1]
#     x_adv_logit = model(x_adv)
#     x_adv_plabel = torch.max(x_adv_logit, dim=1)[1]
#     accurate = (x_clean_plabel == label).float().sum()
#     stable = (x_adv_plabel == x_clean_plabel).float().sum()
#     robust = (x_adv_plabel == label).float().sum()
#     loss_accurate = F.cross_entropy(x_clean_logit, label)
#     loss_stable = F.cross_entropy(model(x_adv), x_clean_plabel)
#     loss = loss_accurate + args.beta * loss_stable
#     return loss, {
#         "accurate": accurate,
#         "stable": stable,
#         "robust": robust
#         }
#
#
# def robust_loss(model, args, x_clean, label):
#     model.eval()
#     x_adv = x_clean + 0.001 * torch.randn(x_clean.shape).cuda()
#     x_adv = torch.clamp(x_adv, 0, 1)
#     # generate adv
#     for _ in range(args.train_num_steps):
#         x_adv.requires_grad_()
#         with torch.enable_grad():
#             loss_adv = F.cross_entropy(model(x_adv), label)
#         grad = torch.autograd.grad(loss_adv, [x_adv])[0]
#         x_adv = x_adv.detach() + args.train_step_size * torch.sign(
#             grad.detach())
#         x_adv = torch.min(
#             torch.max(x_adv, x_clean - args.train_epsilon),
#             x_clean + args.train_epsilon
#             )
#         x_adv = torch.clamp(x_adv, 0.0, 1.0)
#
#     model.train()
#
#     x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
#
#     x_clean_logit = model(x_clean)
#     x_clean_plabel = torch.max(x_clean_logit, dim=1)[1]
#     x_adv_logit = model(x_adv)
#     x_adv_plabel = torch.max(x_adv_logit, dim=1)[1]
#     stable = (x_adv_plabel == x_clean_plabel).float().sum()
#     robust = (x_adv_plabel == label).float().sum()
#     loss_stable = F.cross_entropy(x_adv_logit, x_clean_plabel)
#     loss = F.cross_entropy(x_adv_logit, label)
#     return {
#         "loss": loss,
#         "x_eta": x_eta_ret,
#         "x_adv": x_adv_ret,
#         "l_a": 0,
#         "l_s": loss_stable.item(),
#         "stable": stable,
#         "robust": robust
#         }
