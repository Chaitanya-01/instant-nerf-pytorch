from typing import Callable, Optional, Set, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.amp import custom_fwd, custom_bwd
from einops import rearrange

from model.encoder import hash_encoding, sh_encoding
import tinycudann as tcnn

from kornia.utils.grid import create_meshgrid3d
import nerfngp_cuda
NEAR_DISTANCE = 0.01

# def _meshgrid3d(
#     res: torch.Tensor, device: Union[torch.device, str] = "cpu"
# ) -> torch.Tensor:
#     """Create 3D grid coordinates."""
#     assert len(res) == 3
#     res = res.tolist()
#     return torch.stack(
#         torch.meshgrid(
#             [
#                 torch.arange(res[0], dtype=torch.long),
#                 torch.arange(res[1], dtype=torch.long),
#                 torch.arange(res[2], dtype=torch.long),
#             ],
#             indexing="ij",
#         ),
#         dim=-1,
#     )#.to(device)
def _meshgrid3d(
    res: torch.Tensor, device: Union[torch.device, str] = "cpu"
) -> torch.Tensor:
    """Create 3D grid coordinates."""
    assert len(res) == 3
    res = res.tolist()
    return torch.stack(
        torch.meshgrid(
            [
                torch.arange(res[0], dtype=torch.int32),
                torch.arange(res[1], dtype=torch.int32),
                torch.arange(res[2], dtype=torch.int32),
            ],
            indexing="ij",
        ),
        dim=-1,
    )#.to(device)

class TruncExp(torch.autograd.Function):

    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))


class NGPNeRFNetwork(nn.Module):
    def __init__(
            self,
            num_levels: int = 16, # number of levels in hash table
            coarse_resolution: int = 16, # minimum resolution of  hash table
            fine_resolution: int = 2048, # maximum resolution of the hash table
            table_feature_vec_size: int = 2, # num_features_per_level: int = 2 # number of features per level
            log2_table_size: int = 19,
            aabb: torch.Tensor = torch.Tensor([[-1.0, -1.0, -1.0],[1.0, 1.0, 1.0]]), # Here Scenebox(aabb) where aabb = scale*torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
            sh_degrees: int = 4,
            sh_input_dim: int = 3,            
            scale: float=0.5, # How much to uniformly scale the cube

            density_mlp_num_layers: int = 2,
            density_mlp_layer_width: int = 64,
            density_mlp_output_dim: int = 16,
            color_mlp_num_layers: int = 3,
            color_mlp_layer_width: int = 64
    ):
        super().__init__()
        
        # Other setup use config file to fill multi res hash encoding variables and do something about using aabb in hash encoding
        # Scene bounding box
        self.scale = scale
        self.register_buffer('aabb', self.scale*aabb)
        self.register_buffer('aabb_min', self.aabb[0])
        self.register_buffer('aabb_max', self.aabb[1])

        self.num_cascades = max(1 + int(np.ceil(np.log2(2 * scale))), 1)
        self.grid_size = 128 #64
        self.register_buffer('density_bitfield', torch.zeros(self.num_cascades * self.grid_size**3 // 8, dtype=torch.uint8))
        self.register_buffer('density_grid', torch.zeros(self.num_cascades, self.grid_size**3))

        
        # self.register_buffer('grid_coords', _meshgrid3d(torch.Tensor([self.grid_size]*3)).reshape(-1, 3))
        self.register_buffer('grid_coords', create_meshgrid3d(self.grid_size, self.grid_size, self.grid_size, False, dtype=torch.int32).reshape(-1, 3))
        
        
        self.use_tcnn = True # Set false if you want to use pytorch implementation of MLP and encodings
        if self.use_tcnn:
            b = np.exp(np.log(2048*scale/coarse_resolution)/(num_levels-1))
            self.density_mlp_with_encoding = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=density_mlp_output_dim,
                                                                        encoding_config={
                                                                            "otype": "Grid",
                                                                            "type": "Hash",
                                                                            "n_levels":num_levels,
                                                                            "n_features_per_level":table_feature_vec_size,
                                                                            "base_resolution": coarse_resolution,
                                                                            "per_level_scale": b,
                                                                            "interpolation": "Linear"
                                                                        },
                                                                        network_config={
                                                                            #    "otype": "FullyFusedMLP",
                                                                            "otype": "CutlassMLP",
                                                                            "activation": "ReLU",
                                                                            "output_activation": "None",
                                                                            "n_neurons": 64,
                                                                            "n_hidden_layers": 1,
                                                                        })
            self.direction_encoding = \
                tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config={
                        "otype": "SphericalHarmonics",
                        "degree": 4,
                    },
                )
            
            self.color_mlp = tcnn.Network(n_input_dims=32, n_output_dims=3,
                                        network_config={
                                            #   "otype": "FullyFusedMLP",
                                            "otype": "CutlassMLP",
                                            "activation": "ReLU",
                                            "output_activation": "Sigmoid",
                                            "n_neurons": 64,
                                            "n_hidden_layers": 2,
                                        })
        else:
            self.density_mlp_output_dim = density_mlp_output_dim
            self.hash_encoding = hash_encoding(num_levels,log2_table_size, table_feature_vec_size, coarse_resolution, fine_resolution)
            self.direction_encoding = sh_encoding(sh_degrees, sh_input_dim)
            self.density_mlp = MLP(
                input_dim=self.hash_encoding.get_out_dim(),
                output_dim=self.density_mlp_output_dim,
                num_layers=density_mlp_num_layers,
                layer_width=density_mlp_layer_width,
                hidden_activation=nn.ReLU(),
                # hidden_activation=nn.Softmax(dim=1),
                output_activation=None
            )
            self.color_mlp = MLP(
                # input_dim=self.direction_encoding.output_dim + (self.density_mlp.output_dim - 1),
                input_dim=self.direction_encoding.output_dim + (self.density_mlp.output_dim),
                output_dim=3,
                num_layers=color_mlp_num_layers,
                layer_width=color_mlp_layer_width,
                hidden_activation=nn.ReLU(),
                output_activation=nn.Sigmoid()
            )

    
    def get_density(self, x):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature
        Outputs:
            sigmas: (N)
        """
        x = (x - self.aabb_min) / (self.aabb_max - self.aabb_min) # Required
        
        if self.use_tcnn:
            h = self.density_mlp_with_encoding(x)
        else:
            x = self.hash_encoding(x) # Bounding box already give as input
            h = self.density_mlp(x)


        # sigma = TruncExp.apply(h[..., 0])
        sigma = TruncExp.apply(h[:, 0])
        # sigma = nn.functional.softplus(h[:, 0])

        return sigma


    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        

        x = (x - self.aabb_min) / (self.aabb_max - self.aabb_min) # Required

        if self.use_tcnn:
            h = self.density_mlp_with_encoding(x)
        else:
            x = self.hash_encoding(x) # Bounding box already give as input
            h = self.density_mlp(x)
        

        # sigma = TruncExp.apply(h[..., 0]) # here trunc_exp is the activation function that we apply on the output of denisty_mlp
        sigma = TruncExp.apply(h[:, 0]) # here trunc_exp is the activation function that we apply on the output of denisty_mlp
        # sigma = nn.functional.softplus(h[:, 0])

        # geo_feat = h[..., 1:]
        # geo_feat = h[:, 1:]
        d = d / torch.norm(d, dim=1, keepdim=True) #directions = get_normalized_directions(ray_samples.frustums.directions)
        #directions_flat = directions.view(-1, 3) YES I GUESSSee if requires normalization ?????????
        d = self.direction_encoding((d+1)/2)
        
        # h = torch.cat([d, geo_feat], dim=-1) # (Ntotal_points, 15+16)
        h = torch.cat([d, h], 1) # (Ntotal_points, 15+16)
        
        color = self.color_mlp(h)
        return sigma, color
    
    # Two funcs below tell us how to sample when doing the periodic update of occ grid every 16 iteration steps 
    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = nerfngp_cuda.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)]*self.num_cascades
        return cells
    
    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_thresh):
        cells = []
        for c in range(self.num_cascades):
            # Uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32, device=self.density_grid.device)
            indices1 = nerfngp_cuda.morton3D(coords1).long()

            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c] > density_thresh)[:, 0]
            if len(indices2) > 0:
                rand_idx = torch.randint(len(indices2), (M,), device=self.density_grid.device)
                indices2 = indices2[rand_idx]
            # coords2 = morton3D_invert(indices2.int())
            coords2 = nerfngp_cuda.morton3D_invert(indices2.int())

            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))] 
        return cells
    
    # This func is called at the very start before training to initially markup the empty spaces of occ grid for that scene
    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=32**3):
        """
        mark the cells that aren't covered by the cameras with density -1
        only executed once before training starts
        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        N_cams = poses.shape[0]
        self.count_grid = torch.zeros_like(self.density_grid)
        w2c_R = rearrange(poses[:, :3, :3], 'n a b -> n b a')  # (N_cams, 3, 3)
        w2c_T = -w2c_R @ poses[:, :3, 3:]  # (N_cams, 3, 1)
        
        cells = self.get_all_cells()
        for c in range(self.num_cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i + chunk] / (self.grid_size - 1) * 2 - 1
                s = min(2**(c - 1), self.scale)
                half_grid_size = s / self.grid_size
                xyzs_w = (xyzs * (s - half_grid_size)).T  # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T  # (N_cams, 3, chunk)
                uvd = K @ xyzs_c  # (N_cams, 3, chunk)
                uv = uvd[:, :2] / uvd[:, 2:]  # (N_cams, 2, chunk)
                in_image = (uvd[:, 2]>=0)& \
                           (uv[:, 0]>=0)&(uv[:, 0]<img_wh[0])& \
                           (uv[:, 1]>=0)&(uv[:, 1]<img_wh[1])
                covered_by_cam = (uvd[:, 2] >=
                                  NEAR_DISTANCE) & in_image  # (N_cams, chunk)
                # if the cell is visible by at least one camera
                self.count_grid[c, indices[i:i+chunk]] = \
                    count = covered_by_cam.sum(0)/N_cams

                too_near_to_cam = (uvd[:, 2] <
                                   NEAR_DISTANCE) & in_image  # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count > 0) & (~too_near_to_any_cam)
                # This tells us that all locations are valid i.e. no invisible locations
                # if torch.any(~valid_mask):
                #     print("valid mask (is 0 for invisible -1 grid)", valid_mask)
                self.density_grid[c, indices[i:i+chunk]] = \
                    torch.where(valid_mask, 0., -1.)

    
    @torch.no_grad()
    def update_density_grid(self, density_thresh, warmup=False, decay=0.95, erode=False):
        density_grid_tmp = torch.zeros_like(self.density_grid)
        if warmup:  # during the first steps
            print("Using warmup for update")
            cells = self.get_all_cells()
        else:
            print("Not Using warmup for update")
            cells = self.sample_uniform_and_occupied_cells(
                self.grid_size**3 // 4, density_thresh)
        
        # get sigmas or densities
        for c in range(self.num_cascades):
            indices, coords = cells[c]
            s = min(2**(c - 1), self.scale)
            half_grid_size = s / self.grid_size
            xyzs_w = (coords /(self.grid_size - 1) * 2 - 1) * (s - half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w) * 2 - 1) * half_grid_size
            density_grid_tmp[c, indices] = self.get_density(xyzs_w) #????????????????????????
            # if density_grid_tmp[c, indices] != 0.0:
            # print("Density here is ", density_grid_tmp[c, indices])
        
        if erode:
            # decay more the cells that are visible to few cameras
            decay = torch.clamp(decay**(1 / self.count_grid), 0.1, 0.95)

        self.density_grid = \
            torch.where(self.density_grid<0,
                        self.density_grid,
                        torch.maximum(self.density_grid*decay, density_grid_tmp))
        mean_density = self.density_grid[self.density_grid > 0].mean().item()
    
        nerfngp_cuda.packbits(self.density_grid, min(mean_density, density_thresh), self.density_bitfield)   

        



# Basic MLP with option of skip connections
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_layers: int, # aka total layers
            layer_width: int,
            skip_connections: Optional[Tuple[int]] = None,
            hidden_activation: Optional[nn.Module] = nn.ReLU(),
            output_activation: Optional[nn.Module] = None
            # bias_enabled: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer_width = layer_width
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        # self.bias_enabled = bias_enabled
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.input_dim, self.output_dim))
        else:
            for i in range(self.num_layers - 1):
                
                if i == 0:
                    layers.append(nn.Linear(self.input_dim, self.layer_width))
                elif (i in self._skip_connections) and (i > 0): # Cause skip connection at layer 0 doesn't make sense
                    layers.append(nn.Linear(self.layer_width + self.input_dim, self.layer_width))
                else:
                    layers.append(nn.Linear(self.layer_width, self.layer_width))
                
            layers.append(nn.Linear(self.layer_width, self.output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, in_tensor):
        x = in_tensor
        for i, layer in enumerate(self.layers):
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)
            x = layer(x)
            if self.hidden_activation is not None and i < len(self.layers) - 1:
                x = self.hidden_activation(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x
