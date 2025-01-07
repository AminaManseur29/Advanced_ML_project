import torch
import math
from collections import defaultdict

class GradientPredictionModel:
    def __init__(self, alpha):
        self.alpha = alpha
        self.prev_grad = None

    def predict(self, grad):
        if self.prev_grad is None or self.prev_grad.size() != grad.size():
            self.prev_grad = torch.zeros_like(grad)
        predicted_grad = self.alpha * self.prev_grad + (1 - self.alpha) * grad
        self.prev_grad = predicted_grad.detach()
        return predicted_grad

def compute_gradient_change_rate(grad, prev_grad):
    if prev_grad is None or prev_grad.size() != grad.size():
        return 0
    change_rate = torch.norm(grad - prev_grad) / (torch.norm(prev_grad) + 1e-8)
    return change_rate.item()

def compute_dynamic_beta(change_rate, min_val, max_val):
    beta = min_val + (max_val - min_val) * (1 - change_rate)
    return min(max(beta, min_val), max_val)

class BGE_Adam():
    def __init__(self, params, lr=0.001, alpha=0.5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 entropy_weight=0.01, amsgrad=False, beta1_max=0.9, beta1_min=0.5, beta2_max=0.999,
                 beta2_min=0.9):
        self.defaults = dict(lr=lr, alpha=alpha, betas=betas, eps=eps, weight_decay=weight_decay,
                        entropy_weight=entropy_weight, amsgrad=amsgrad,
                        beta1_max=beta1_max, beta1_min=beta1_min,
                        beta2_max=beta2_max, beta2_min=beta2_min)
        self.gradient_prediction_model = {}
        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        for param_group in param_groups:
            param_group['params'] = list(param_group['params'])
            for name, default in self.defaults.items():
                param_group.setdefault(name, default)
            self.param_groups.append(param_group)

    def zero_grad(self, set_to_none: bool = False):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        else:
                            if p.grad.grad_fn is not None:
                                p.grad.detach_()
                            else:
                                p.grad.requires_grad_(False)
                            p.grad.zero_()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                amsgrad = group['amsgrad']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Compute beta1 and beta2 values dynamically based on gradient change rate
                prev_grad = state['exp_avg'] if 'exp_avg' in state else None
                gradient_change_rate = compute_gradient_change_rate(grad, prev_grad)
                beta1 = compute_dynamic_beta(gradient_change_rate, group['beta1_min'], group['beta1_max'])
                beta2 = compute_dynamic_beta(gradient_change_rate, group['beta2_min'], group['beta2_max'])

                # Update the moving averages of gradient and its square
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Prediction of next gradient (gradient_prediction_model)
                if p in self.gradient_prediction_model:
                    predicted_grad = self.gradient_prediction_model[p].predict(grad)
                else:
                    self.gradient_prediction_model[p] = GradientPredictionModel(group['alpha'])
                    predicted_grad = self.gradient_prediction_model[p].predict(grad)

                # Final parameter update with entropy adjustment
                entropy_adjustment = 1 + group['entropy_weight'] * torch.randn_like(p.data).mean()
                p.data.addcdiv_(predicted_grad, denom, value=-step_size * entropy_adjustment)

        return loss
        
def func(params):
    x, y = params
    term1 = torch.sin(x) * torch.cos(y)
    term2 = 0.1 * (x - 1) ** 2 + 0.2 * (y - 2) ** 2
    term3 = torch.sin(x + y) - 0.5 * torch.cos(2 * x + y)
    return term1 + term2 + term3

x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)

optimizer = BGE_Adam([x, y], lr=0.1)

for step in range(1000):
    optimizer.zero_grad()    
    loss = func([x, y])
    loss.backward()         
    optimizer.step() 

    if step % 10 == 0:
        print(f"Step {step}, x = {x.item()}, y = {y.item()}, f(x, y) = {loss.item()}")
        
print(f"Optimized x = {x.item()}, y = {y.item()}, minimum value = {func([x, y]).item()}")
