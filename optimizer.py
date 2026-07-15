from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """Adam with decoupled weight decay (AdamW).

    WHAT ADAM DOES
    --------------
    Adam is an adaptive first-order optimizer. For every parameter it keeps a running
    estimate of the gradient's first moment (the mean, `m`) and second moment (the
    uncentered variance, `v`). It then takes a step whose direction is `m` but whose
    per-parameter size is scaled down by `sqrt(v)`. Parameters with consistently large /
    noisy gradients get smaller effective steps; parameters with small, steady gradients
    get relatively larger ones. This gives each weight its own adaptive learning rate.

    THE STATE (stored per-parameter in `self.state[p]`)
    ---------------------------------------------------
      - `step` : int, how many updates this parameter has seen (the timestep `t`).
      - `exp_avg`    (m) : tensor, same shape as p. Exponential moving average of grad.
      - `exp_avg_sq` (v) : tensor, same shape as p. EMA of grad*grad (element-wise square).
    On the very first step this dict is empty, so you must initialize step=0 and
    m, v = zeros_like(p.data).

    THE HYPERPARAMETERS (read from the `group` dict)
    ------------------------------------------------
      - lr           (alpha) : base step size.
      - betas = (b1, b2)     : EMA decay rates for m and v (e.g. 0.9, 0.999).
      - eps                  : small constant added to the denominator for stability.
      - weight_decay         : strength of the decoupled weight-decay term.
      - correct_bias         : whether to apply bias correction (see below).

    ONE UPDATE (what `step()` must compute for each parameter)
    ----------------------------------------------------------
      t <- t + 1
      m <- b1 * m + (1 - b1) * g                 # updated first moment
      v <- b2 * v + (1 - b2) * g^2               # updated second moment
      # Bias correction: m and v start at 0, so they are biased toward 0 for small t.
      # This code uses the "efficient" form from Kingma & Ba: instead of forming
      # m_hat = m/(1-b1^t) and v_hat = v/(1-b2^t) explicitly, fold the correction into
      # the step size:   alpha_t = alpha * sqrt(1 - b2^t) / (1 - b1^t)
      p <- p - alpha_t * m / (sqrt(v) + eps)     # main Adam update on p.data

    DECOUPLED WEIGHT DECAY (the "W" in AdamW)
    -----------------------------------------
    Classic Adam adds L2 regularization into the gradient, which then gets rescaled by
    the adaptive `sqrt(v)` denominator (so it interacts badly with the moment estimates).
    AdamW instead applies weight decay *directly to the parameters*, separately from the
    gradient step:   p <- p - lr * weight_decay * p. Do this as its own update (do NOT
    add weight_decay * p into `grad`).
    """

    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                # First time we see `p`, state is an empty dict -> initialize it.
                state = self.state[p]
                if state is None:
                    state = dict()

                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                correct_bias = group["correct_bias"]
                weight_decay = group["weight_decay"]
                
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                
                if state["step"] % 100 == 0:
                    print(f"optimizer state value: {state}")
                
                # Update state variables: Steps, First moment and Second moment.
                state["step"] += 1
                state["exp_avg"] = beta1*state["exp_avg"] + (1-beta1)*grad
                state["exp_avg_sq"] = beta2*state["exp_avg_sq"] + (1-beta2)*grad*grad
                
                # Calculate bias terms if needed.
                if correct_bias == False:
                    alpha_t = alpha
                else:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    alpha_t = alpha * math.sqrt(bias_correction2) / bias_correction1
                
                # Apply gradient update to weights
                p.data = p.data - alpha_t * state["exp_avg"] / (torch.sqrt(state["exp_avg_sq"]) + eps)
                # Apply decoupled weight decay
                p.data = p.data - alpha * weight_decay * p.data

        return loss
