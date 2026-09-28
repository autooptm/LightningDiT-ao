import contextlib
import os

import numpy as np
import torch as th
import torch.nn as nn
from torchdiffeq import odeint
from functools import partial
from tqdm import tqdm

class sde:
    """SDE solver class"""
    def __init__(
        self, 
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t
        dw = w_cur * th.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x
    
    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(self.dt)
        t_cur = th.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (K1 + K2), xhat # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Smapler type not implemented.")
    
        return sampler

    def sample(self, init, model, **model_kwargs):
        """forward loop of sde"""
        x = init
        mean_x = init 
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)

        return samples

#
#
#
#

def _ao_enabled(name, default=True):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


class _AoEulerRunner:

    def __init__(self, t_grid, use_step):
        self.t = t_grid
        self.dt = t_grid[1:] - t_grid[:-1]
        self.n_steps = len(t_grid) - 1
        self.use_step = use_step
        self._steps = {}

    @staticmethod
    def _opt_ctx(model):
        owner = getattr(model, "__self__", None)
        dtype = None
        if owner is not None:
            try:
                dtype = next(owner.parameters()).dtype
            except (StopIteration, AttributeError):
                dtype = None
        if dtype in (th.float16, th.bfloat16):
            return th.autocast("cuda", dtype=dtype)
        return contextlib.nullcontext()

    def _step_for(self, x, model, model_kwargs):
        key = (tuple(x.shape), x.dtype)
        cached = self._steps.get(key)
        if cached is not None:
            return cached

        sx = th.zeros_like(x)
        st = th.zeros(x.shape[0], device=x.device, dtype=self.t.dtype)
        sdt = th.zeros((), device=x.device, dtype=self.dt.dtype)
        kws = dict(model_kwargs)
        static_y = None
        if "y" in kws and th.is_tensor(kws["y"]):
            static_y = th.zeros_like(kws["y"])
            kws["y"] = static_y

        side = th.cuda.Stream()
        side.wait_stream(th.cuda.current_stream())
        with th.cuda.stream(side):
            for _ in range(3):
                with self._opt_ctx(model):
                    sx.add_(sdt * model(sx, st, **kws))
        th.cuda.current_stream().wait_stream(side)
        th.cuda.synchronize()

        g = th.cuda.CUDAGraph()
        with th.cuda.graph(g):
            with self._opt_ctx(model):
                sx.add_(sdt * model(sx, st, **kws))

        probe = th.randn_like(sx)
        pt = self.t[0].expand(x.shape[0]).clone()
        pdt = self.dt[0].clone()
        if static_y is not None:
            static_y.copy_(model_kwargs["y"])
        sx.copy_(probe); st.copy_(pt); sdt.copy_(pdt)
        g.replay()
        got = sx.clone()
        sx.copy_(probe); st.copy_(pt); sdt.copy_(pdt)
        with th.no_grad(), self._opt_ctx(model):
            sx.add_(sdt * model(sx, st, **kws))
        err = (got - sx).abs().max().item()
        scale = sx.abs().max().item() or 1.0
        if not err <= 1e-3 * scale:
            raise RuntimeError(
                "AO_LDIT_OPT_1: optimized path disagrees with the stock path for %s "
                "(max|diff| %.3g against a state scale of %.3g). "
                "Set AO_LDIT_OPT_1=0 to fall back." % (key, err, scale))

        cached = (g, sx, st, sdt, static_y)
        self._steps[key] = cached
        return cached

    def run(self, x, model, **model_kwargs):
        if self.use_step and x.is_cuda:
            g, sx, st, sdt, sy = self._step_for(x, model, model_kwargs)
            sx.copy_(x)
            if sy is not None:
                sy.copy_(model_kwargs["y"])
            for i in range(self.n_steps):
                st.copy_(self.t[i].expand(x.shape[0]))
                sdt.copy_(self.dt[i])
                g.replay()
            return sx.clone()

        with self._opt_ctx(model):
            for i in range(self.n_steps):
                x = x + self.dt[i] * model(x, self.t[i].expand(x.shape[0]),
                                           **model_kwargs)
        return x


class ode:
    """ODE solver class"""
    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        timestep_shift,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        self.t = th.linspace(t0, t1, num_steps)

        if timestep_shift > 0:
            def compute_tm(t_n, timestep_shift):
                numerator = timestep_shift * t_n
                denominator = 1 + (timestep_shift - 1) * t_n
                return numerator / denominator
            self.t = th.tensor([compute_tm(t_n, timestep_shift) for t_n in self.t])

        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):

        if (self.sampler_type == "euler" and not isinstance(x, tuple)
                and _ao_enabled("AO_LDIT_FAST_EULER")):
            if getattr(self, "_ao_runner", None) is None:
                self._ao_runner = _AoEulerRunner(
                    self.t.to(x.device),
                    use_step=_ao_enabled("AO_LDIT_OPT_1") and x.is_cuda)
            return [self._ao_runner.run(x, model, **model_kwargs)]

        device = x[0].device if isinstance(x, tuple) else x.device
        def _fn(t, x):
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            model_output = self.drift(x, t, model, **model_kwargs)
            return model_output

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(
            _fn,
            x,
            t,
            method=self.sampler_type,
            atol=atol,
            rtol=rtol
        )
        return samples