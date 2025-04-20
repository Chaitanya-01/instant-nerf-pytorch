import torch
from einops import rearrange
from torch_scatter import segment_csr

from torch.amp import custom_fwd, custom_bwd

import nerfngp_cuda
# try:
#     import _raymarching_cuda as _backend
# except ImportError:
# from .backend import _backend

class RayAABBIntersector(torch.autograd.Function):
    """
    Computes the intersections of rays and axis-aligned voxels.

    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        centers: (N_voxels, 3) voxel centers
        half_sizes: (N_voxels, 3) voxel half sizes
        max_hits: maximum number of intersected voxels to keep for one ray
                  (for a cubic scene, this is at most 3*N_voxels^(1/3)-2)

    Outputs:
        hits_cnt: (N_rays) number of hits for each ray
        (followings are from near to far)
        hits_t: (N_rays, max_hits, 2) hit t's (-1 if no hit)
        hits_voxel_idx: (N_rays, max_hits) hit voxel indices (-1 if no hit)
    """
    @staticmethod
    @custom_fwd(device_type='cuda',cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, center, half_size, max_hits):
        return nerfngp_cuda.ray_aabb_intersect(rays_o, rays_d, center, half_size, max_hits)

class RayMarching(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda',cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, hits_t, grid_size, num_cascades, scale, density_bitfield, max_samples):
        # noise to perturb the first sample of each ray
        noise = torch.rand_like(rays_o[:, 0])

        rays, xyzs, dirs, deltas, ts, counter = nerfngp_cuda.raymarching_train(rays_o, rays_d, hits_t, grid_size, num_cascades, scale, density_bitfield, max_samples, noise)

        total_samples = counter[0]
        # remove the redundant output
        xyzs = xyzs[:total_samples]
        dirs = dirs[:total_samples]
        deltas = deltas[:total_samples]
        ts = ts[:total_samples]

        ctx.save_for_backward(rays, ts)

        return rays, xyzs, dirs, deltas, ts, total_samples
    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dL_drays, dL_dxyzs, dL_ddirs, dL_ddeltas, dL_dts, dL_dtotal_samples):
        rays, ts = ctx.saved_tensors
        
        segments = torch.cat([rays[:, 1], rays[-1:, 1] + rays[-1:, 2]])
        dL_drays_o = segment_csr(dL_dxyzs, segments)
        dL_drays_d = segment_csr(dL_dxyzs*rearrange(ts, 'n -> n 1') + dL_ddirs, segments)

        return dL_drays_o, dL_drays_d, None, None, None, None, None, None, None