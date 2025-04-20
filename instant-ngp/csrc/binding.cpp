#include "include/utils_header.h"

#include <torch/extension.h>

at::Tensor morton3D(const at::Tensor coords){
    return morton3D_cu(coords);
}


at::Tensor morton3D_invert(const at::Tensor indices){
    return morton3D_invert_cu(indices);
}

void packbits(
    at::Tensor density_grid,
    const float density_threshold,
    at::Tensor density_bitfield
){  
    return packbits_cu(density_grid, density_threshold, density_bitfield);
}

std::vector<at::Tensor> ray_aabb_intersect(
    const at::Tensor rays_o,
    const at::Tensor rays_d,
    const at::Tensor centers,
    const at::Tensor half_sizes,
    const int max_hits
){
    return ray_aabb_intersect_cu(rays_o, rays_d, centers, half_sizes, max_hits);
}

std::vector<at::Tensor> raymarching_train(
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
){
    return raymarching_train_cu(
        rays_o, rays_d, hits_t, grid_size, num_cascades,
        scale, density_bitfield, max_samples, noise);
}

std::vector<at::Tensor> raymarching_test(
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
){
    return raymarching_test_cu(
        rays_o, rays_d, hits_t, alive_indices, density_bitfield, num_cascades,
        scale, grid_size, max_samples, N_samples);
}

std::vector<at::Tensor> composite_train_fw(
    const at::Tensor sigmas,
    const at::Tensor rgbs,
    const at::Tensor deltas,
    const at::Tensor ts,
    const at::Tensor rays,
    const float opacity_threshold
){
    return composite_train_fw_cu(
                sigmas, rgbs, deltas, ts,
                rays, opacity_threshold);
}


std::vector<at::Tensor> composite_train_bw(
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
    const float opacity_threshold
){
    return composite_train_bw_cu(
                dL_dopacity, dL_ddepth, dL_drgb, dL_dws,
                sigmas, rgbs, ws, deltas, ts, rays,
                opacity, depth, rgb, opacity_threshold);
}

void composite_test_fw(
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
){
    composite_test_fw_cu(
        sigmas, rgbs, deltas, ts, hits_t, alive_indices,
        T_threshold, N_eff_samples,
        opacity, depth, rgb);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#define _REG_FUNC(funname) m.def(#funname, &funname)
    _REG_FUNC(morton3D);  // TODO: check this function.
    _REG_FUNC(morton3D_invert);
    _REG_FUNC(packbits);
    _REG_FUNC(ray_aabb_intersect);
    _REG_FUNC(raymarching_train);
    _REG_FUNC(raymarching_test);
    _REG_FUNC(composite_train_fw);
    _REG_FUNC(composite_train_bw);
    _REG_FUNC(composite_test_fw);
#undef _REG_FUNC
}