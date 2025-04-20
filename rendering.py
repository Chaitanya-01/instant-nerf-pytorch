import torch
from einops import rearrange
import numpy as np
import time

from raymarching_cuda import RayMarching, RayAABBIntersector
from volumerenderer_cuda import VolumeRenderer
import nerfngp_cuda

NEAR_DISTANCE = 0.01

def render(
        model,
        rays_o,
        rays_d,
        test_time=False,
        T_threshold=1e-4,
        max_samples=1024
):
    """
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)
    Inputs:
        model: NGP
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
    Outputs:
        result: dictionary containing final rgb and depth
    """
    
    rays_o = rays_o.contiguous()
    rays_d = rays_d.contiguous()
    

    # # hits_t = ray_aabb_intersect(rays_o.contiguous(), rays_d.contiguous(), model.scale)
    # hits_t = ray_aabb_intersect(rays_o, rays_d, model.scale)
    center = (model.aabb_max + model.aabb_min)/2
    half_size = (model.aabb_max - model.aabb_min)/2
    
    _, hits_t, _ = RayAABBIntersector.apply(rays_o, rays_d, center.reshape(1, -1), half_size.reshape(1, -1), 1)
    hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE
    
    # hits_t = hits_t.contiguous()
    if test_time:
        return __render_rays_test(
            model,
            rays_o,
            rays_d,
            hits_t, #hits_t
            T_threshold
        )
    else:
        return __render_rays_train(
            model,
            rays_o,
            rays_d,
            hits_t, #hits_tsqueeze(1)
            T_threshold
        )

def __render_rays_test(model, rays_o, rays_d, hits_t, T_threshold=1e-4):
    """
    Render rays by
    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    results={}
    # output tensors to be filled in
    N_rays = len(rays_o)
    
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)

    samples = total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1

    max_samples = 1024

    while samples < max_samples:
        N_alive = len(alive_indices)
        if N_alive == 0:
            break
        
        # the number of samples to add on each ray
        N_samples = max(min(N_rays // N_alive, 64), min_samples)
        samples += N_samples

        
        xyzs, dirs, deltas, ts, N_eff_samples = nerfngp_cuda.raymarching_test(rays_o, rays_d, hits_t[:, 0], alive_indices,model.density_bitfield, model.num_cascades,model.scale,model.grid_size, max_samples, N_samples)
        total_samples += N_eff_samples.sum()
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs==0, dim=1)
        if valid_mask.sum()==0: break

        sigmas = torch.zeros(len(xyzs), device=device)
        rgbs = torch.zeros(len(xyzs), 3, device=device)
        sigmas[valid_mask], _rgbs = model(xyzs[valid_mask], dirs[valid_mask])
        rgbs[valid_mask] = _rgbs.float()
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        
        nerfngp_cuda.composite_test_fw(sigmas, rgbs, deltas, ts,hits_t[:, 0], alive_indices, T_threshold,
            N_eff_samples, opacity, depth, rgb)
        # remove converged rays
        alive_indices = alive_indices[alive_indices >= 0]
    results['opacity'] = opacity
    results['depth'] = depth
    results['rgb'] = rgb
    results['total_samples'] = total_samples # total samples for all rays

    rgb_bg = torch.ones(3, device=device)
    results['rgb'] += rgb_bg * rearrange(1 - opacity, 'n -> n 1')
    return results

def __render_rays_train(
    model,
    rays_o,
    rays_d,
    hits_t,
    T_threshold=1e-4,
):
    """
    Render rays by
    1. March the rays along their directions, querying @density_bitfield
       to skip empty space, and get the effective sample points (where
       there is object)
    2. Infer the NN at these positions and view directions to get properties
       (currently sigmas and rgbs)
    3. Use volume rendering to combine the result (front to back compositing
       and early stop the ray if its transmittance is below a threshold)
    """
    
    results = {}
    max_samples = 1024
    
    # rays, xyzs, dirs, results['deltas'], results['ts'], results['rm_samples'] = ray_marching_train(rays_o, rays_d, hits_t[:,0], hits_t[:, 1], model.grid_size, model.num_cascades, model.scale, model.density_bitfield, 1024)
    # nears = hits_t[:,0]
    # fars = hits_t[:, 1]
    # nears = hits_t[:,0, 0]
    # fars = hits_t[:, 0, 1]
    # nears = nears.contiguous()
    # fars = fars.contiguous()
    # ray_march_timer = time.time()
    
    
    (rays, xyzs, dirs, results['deltas'], results['ts'], results['rm_samples']) = RayMarching.apply(rays_o, rays_d, hits_t[:, 0], model.grid_size, model.num_cascades, model.scale, model.density_bitfield, max_samples)
    # (rays, xyzs, dirs, results['deltas'], results['ts'], results['rm_samples']) = RayMarching.apply(rays_o, rays_d, hits_t[:,0, 0], hits_t[:, 0, 1], model.grid_size, model.num_cascades, model.scale, model.density_bitfield, 1024)
    
    
    sigmas, rgbs = model(xyzs, dirs)
    rgbs = rgbs.contiguous()

    # results['vr_samples'],results['opacity'],results['depth'],results['rgb'],results['ws'] = model.render_func(sigmas, rgbs, results['deltas'], results['ts'], rays, T_threshold)
    results['vr_samples'],results['opacity'],results['depth'],results['rgb'],results['ws'] = VolumeRenderer.apply(sigmas, rgbs, results['deltas'], results['ts'], rays, T_threshold)
    # results['vr_samples'],results['opacity'],results['depth'],results['rgb'],results['ws'] = model.render_func(sigmas, rgbs, deltas, ts, rays, T_threshold)
    results['rays'] = rays

    rgb_bg = torch.ones(3, device=rays_o.device)
    results['rgb'] = results['rgb'] +rgb_bg*rearrange(1-results['opacity'], 'n -> n 1')

    return results
