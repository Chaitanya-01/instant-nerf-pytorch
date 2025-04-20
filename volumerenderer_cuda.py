import torch
from einops import rearrange
from torch_scatter import segment_csr

from torch.amp import custom_fwd, custom_bwd

import nerfngp_cuda



class VolumeRenderer(torch.autograd.Function):
    """
    Volume rendering with different number of samples per ray
    Used in training only

    Inputs:
        sigmas: (N)
        rgbs: (N, 3)
        deltas: (N)
        ts: (N)
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]
        T_threshold: float, stop the ray if the transmittance is below it

    Outputs:
        total_samples: int, total effective samples
        opacity: (N_rays)
        depth: (N_rays)
        rgb: (N_rays, 3)
        ws: (N) sample point weights
    """
    @staticmethod
    @custom_fwd(device_type='cuda',cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, deltas, ts, rays, T_threshold):
        total_samples, opacity, depth, rgb, ws = \
            nerfngp_cuda.composite_train_fw(sigmas, rgbs, deltas, ts,
                                    rays, T_threshold)
        ctx.save_for_backward(sigmas, rgbs, deltas, ts, rays,
                              opacity, depth, rgb, ws)
        ctx.T_threshold = T_threshold
        return total_samples.sum(), opacity, depth, rgb, ws

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dL_dtotal_samples, dL_dopacity, dL_ddepth, dL_drgb, dL_dws):
        sigmas, rgbs, deltas, ts, rays, \
        opacity, depth, rgb, ws = ctx.saved_tensors
        dL_dsigmas, dL_drgbs = \
            nerfngp_cuda.composite_train_bw(dL_dopacity, dL_ddepth, dL_drgb, dL_dws,
                                    sigmas, rgbs, ws, deltas, ts,
                                    rays,
                                    opacity, depth, rgb,
                                    ctx.T_threshold)
        return dL_dsigmas, dL_drgbs, None, None, None, None