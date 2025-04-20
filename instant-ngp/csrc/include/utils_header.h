#pragma once

#include <stdint.h>
#include <torch/torch.h>
#include <ATen/ATen.h>

// function headers
at::Tensor morton3D_cu(const at::Tensor coords);
at::Tensor morton3D_invert_cu(const at::Tensor indices);

void packbits_cu(
    at::Tensor density_grid,
    const float density_threshold,
    at::Tensor density_bitfield
);

std::vector<at::Tensor> ray_aabb_intersect_cu(
    const at::Tensor rays_o,
    const at::Tensor rays_d,
    const at::Tensor centers,
    const at::Tensor half_sizes,
    const int max_hits
);

std::vector<at::Tensor> raymarching_train_cu(
    const at::Tensor rays_o,
    const at::Tensor rays_d,
    // const at::Tensor nears,
    // const at::Tensor fars,
    const at::Tensor hits_t,
    const int grid_size,
    const int num_cascades,
    const float scale,
    const at::Tensor density_bitfield,
    const int max_samples,
    const at::Tensor noise
);

std::vector<at::Tensor> raymarching_test_cu(
    const at::Tensor rays_o,
    const at::Tensor rays_d,
    at::Tensor hits_t,
    const at::Tensor alive_indices,
    const at::Tensor density_bitfield,
    const int num_cascades,
    const float scale,
    const int grid_size,
    const int max_samples,
    const int N_samples
);


std::vector<at::Tensor> composite_train_fw_cu(
    const at::Tensor sigmas,
    const at::Tensor rgbs,
    const at::Tensor deltas,
    const at::Tensor ts,
    const at::Tensor rays,
    const float T_threshold
);


std::vector<at::Tensor> composite_train_bw_cu(
    const at::Tensor dL_dopacity,
    const at::Tensor dL_ddepth,
    const at::Tensor dL_drgb,
    const at::Tensor dL_dws,
    const at::Tensor sigmas,
    const at::Tensor rgbs,
    const at::Tensor ws,
    const at::Tensor deltas,
    const at::Tensor ts,
    const at::Tensor rays,
    const at::Tensor opacity,
    const at::Tensor depth,
    const at::Tensor rgb,
    const float T_threshold
);

void composite_test_fw_cu(
    const at::Tensor sigmas,
    const at::Tensor rgbs,
    const at::Tensor deltas,
    const at::Tensor ts,
    const at::Tensor hits_t,
    const at::Tensor alive_indices,
    const float T_threshold,
    const at::Tensor N_eff_samples,
    at::Tensor opacity,
    at::Tensor depth,
    at::Tensor rgb
);
