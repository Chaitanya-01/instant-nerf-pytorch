# Load a dataset and generate raybundles for the data and some util functions for the ray bundles.
from typing import Tuple, Union
import torch
import numpy as np
from kornia import create_meshgrid
from einops import rearrange



@torch.amp.autocast(device_type='cuda',dtype=torch.float32)
def get_ray_directions(H, W, K, device='cpu', random=False, return_uv=False, flatten=True):
    """
    Get ray directions for all pixels in camera coordinate [right down front].
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics
        random: whether the ray passes randomly inside the pixel
        return_uv: whether to return uv image coordinates

    Outputs: (shape depends on @flatten)
        directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
        uv: (H, W, 2) or (H*W, 2) image coordinates
    """
    # xs = torch.linspace(0, W - 1, W, device=device)
    # ys = torch.linspace(0, H - 1, H, device=device)
    # grid = torch.stack(torch.meshgrid([xs, ys], indexing="ij"), dim=-1).to(device) # (H, W, 2)
    grid = create_meshgrid(H, W, False, device=device)[0] # (H, W, 2)
    u, v = grid.unbind(-1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    if random:
        directions = \
            torch.stack([(u-cx+torch.rand_like(u))/fx,
                         (v-cy+torch.rand_like(v))/fy,
                         torch.ones_like(u)], -1)
    else: # pass by center
        directions = torch.stack([(u-cx+0.5)/fx, (v-cy+0.5)/fy, torch.ones_like(u)], -1)
    
    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1, 2)
    
    if return_uv:
        return directions, grid
    return directions


@torch.amp.autocast(device_type='cuda',dtype=torch.float32)
def get_rays(directions, c2w): #, K, c2w):
    """
    Get ray origin and directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (N, 3) ray directions in camera coordinate
        c2w: (3, 4) or (N, 3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (N, 3), the origin of the rays in world coordinate
        rays_d: (N, 3), the direction of the rays in world coordinate
    """
    if c2w.ndim == 2:
        # Rotate ray directions from camera coordinate to the world coordinate
        rays_d = directions @ c2w[:, :3].T
    else:
        rays_d = rearrange(directions, 'n c -> n 1 c') @ rearrange(c2w[..., :3], 'n a b -> n b a')
        rays_d = rearrange(rays_d, 'n 1 c -> n c')

    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[..., 3].expand_as(rays_d)
    
    # if(rays_d.shape[1] == 4):
    #     rays_d = rays_d[:, :3]
    #     rays_o = rays_o[:, :3]

    return rays_o, rays_d



def ray_aabb_intersect(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    scale: float,
    near_distance: float = 0.01
    # near_plane: float = -float("inf"),
    # far_plane: float = float("inf"),
    # miss_value: float = float("inf"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ray-AABB intersection.

    Functionally the same with `ray_aabb_intersect()`, but slower with pure Pytorch.
    Args:
        rays_o: (n_rays, 3) Ray origins.
        rays_d: (n_rays, 3) Normalized ray directions.
        aabbs: (m, 6) Axis-aligned bounding boxes {xmin, ymin, zmin, xmax, ymax, zmax}. NOT USING THIS METHOD
        near_plane: Optional. Near plane. Default to -infinity.
        far_plane: Optional. Far plane. Default to infinity.
        miss_value: Optional. Value to use for tmin and tmax when there is no intersection.
            Default to infinity.

    Returns:
        A tuple of {Tensor, Tensor, BoolTensor}:

        - **t_mins**: (n_rays, m) tmin for each ray-AABB pair.
        - **t_maxs**: (n_rays, m) tmax for each ray-AABB pair.
        - **hits**: (n_rays, m) whether each ray-AABB pair intersects.
    """
    # Initialize output tensor with -1s
    hits_t = torch.full((rays_o.shape[0], 2), -1.0, device=rays_o.device, dtype=rays_o.dtype)

    # Compute the minimum and maximum bounds of the AABB
    aabb_max = torch.Tensor([scale, scale, scale]).to(device=rays_o.device)
    # aabb_max = aabb_max.to(device=rays_o.device)
    aabb_min = -aabb_max
    half_size = (aabb_max - aabb_min) / 2
    center = torch.Tensor([0.0, 0.0, 0.0]) #(aabb_max + aabb_min) / 2

    # Compute the intersection distances between the ray and each of the six AABB planes
    # t1 = (aabb_min[None, :, :] - rays_o[:, None, :]) / rays_d[:, None, :]
    # t2 = (aabb_max[None, :, :] - rays_o[:, None, :]) / rays_d[:, None, :]
    t1 = (aabb_min - rays_o) / rays_d
    
    t2 = (aabb_max - rays_o) / rays_d

    # Compute the maximum tmin and minimum tmax for each AABB
    t_mins = torch.max(torch.min(t1, t2), dim=-1)[0]
    t_maxs = torch.min(torch.max(t1, t2), dim=-1)[0]

    # Compute whether each ray-AABB pair intersects
    hits = (t_maxs > t_mins) & (t_maxs > 0)
    # Only set valid intersections
    hits_t[hits] = torch.stack([
        torch.maximum(t_mins[hits], torch.tensor(near_distance, device=rays_o.device)),
        t_maxs[hits]
    ], dim=1)

    # # Clip the tmin and tmax values to the near and far planes
    # t_mins = torch.clamp(t_mins, min=near_plane, max=far_plane)
    # t_maxs = torch.clamp(t_maxs, min=near_plane, max=far_plane)

    # # Set the tmin and tmax values to miss_value if there is no intersection
    # t_mins = torch.where(hits, t_mins, miss_value)
    # t_maxs = torch.where(hits, t_maxs, miss_value)
    return hits_t
    # return t_mins, t_maxs, hits



def main():
    print("\nTesting")
    


if __name__ == "__main__":
    main()