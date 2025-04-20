import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class BaseConfig:


    # alpha_thre: float = 0.01
    # """Threshold for opacity skipping."""

    # Grid parameters
    grid_size: int = 128
    num_cascades: int = 5

    # Rendering parameters
    min_near: float = 0.2
    max_far: float = 80.0
    max_samples: int = 1024

    # training params
    batch_size: int = 8192 # number of rays in a batch
    num_epochs: int = 20000 # number of epochs to train
    lr: float = 1e-2

    ckpt_path: str = None # pretrained checkpoint to load (including optimizers)

@dataclass
class NetworkConfig:
    # Multi resolution hash grid params
    num_levels: int = 16 # number of levels in hash table
    coarse_resolution: int = 16 # minimum resolution of  hash table
    fine_resolution: int = 2048 # maximum resolution of the hash table
    table_feature_vec_size: int = 2 # num_features_per_level: int = 2 # number of features per level
    log2_table_size: int = 19 # log2_hashmap_size: int = 19 # maximum number of entries per level 2^19
    aabb: torch.Tensor = torch.Tensor([[-1.0, -1.0, -1.0],[1.0, 1.0, 1.0]]) #**REMOVE THIS AND USE AABB CLASS IF NEEDED AT ALL
    
    # SH encoding
    sh_degrees: int = 4
    sh_input_dim: int = 3
    
    # Network params
    scale: float = 1.0#0.5 # scene scale (whole scene must lie in [-scale, scale]^3
    # # density MLP parameters
    density_mlp_num_layers: int = 2
    density_mlp_layer_width: int = 64
    density_mlp_output_dim: int = 16
    # # color MLP parameters
    color_mlp_num_layers: int = 3
    color_mlp_layer_width: int = 64