#include <ATen/ATen.h>
#include <torch/torch.h>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "include/utils_math.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/////////training/////////////////////////////////
template <typename scalar_t>
__global__ void composite_train_fw_kernel(
    const at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, size_t> sigmas,
    const at::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, size_t> rgbs,
    const at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, size_t> deltas,
    const at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, size_t> ts,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> rays,
    const scalar_t T_threshold,
    at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> total_samples,
    at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, size_t> opacity,
    at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, size_t> depth,
    at::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, size_t> rgb,
    at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, size_t> ws
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= opacity.size(0)) return;

    const int ray_idx = rays[n][0], start_idx = rays[n][1], N_samples = rays[n][2];

    // front to back compositing
    int samples = 0; scalar_t T = 1.0f;

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T; // weight of the sample point

        rgb[ray_idx][0] += w*rgbs[s][0];
        rgb[ray_idx][1] += w*rgbs[s][1];
        rgb[ray_idx][2] += w*rgbs[s][2];
        depth[ray_idx] += w*ts[s];
        opacity[ray_idx] += w;
        ws[s] = w;
        T *= 1.0f-a;

        if (T <= T_threshold) break; // ray has enough opacity
        samples++;
    }
    total_samples[ray_idx] = samples;
}


std::vector<at::Tensor> composite_train_fw_cu(
    const at::Tensor sigmas,
    const at::Tensor rgbs,
    const at::Tensor deltas,
    const at::Tensor ts,
    const at::Tensor rays,
    const float T_threshold
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays);
    const int N_rays = rays.size(0), N = sigmas.size(0);

    auto opacity = at::zeros({N_rays}, sigmas.options());
    auto depth = at::zeros({N_rays}, sigmas.options());
    auto rgb = at::zeros({N_rays, 3}, sigmas.options());
    auto ws = at::zeros({N}, sigmas.options());
    auto total_samples = at::zeros({N_rays}, at::dtype(at::kLong).device(sigmas.device()));

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.scalar_type(), "composite_train_fw_cu", 
    ([&] {
        composite_train_fw_kernel<scalar_t><<<blocks, threads>>>(
            sigmas.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, size_t>(),
            rgbs.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, size_t>(),
            rays.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
            T_threshold,
            total_samples.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            opacity.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, size_t>(),
            depth.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, size_t>(),
            rgb.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, size_t>(),
            ws.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, size_t>()
        );
    }));

    return {total_samples, opacity, depth, rgb, ws};
}

// backward

template <typename scalar_t>
__global__ void composite_train_bw_kernel(
    const at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, size_t> dL_dopacity,
    const at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, size_t> dL_ddepth,
    const at::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, size_t> dL_drgb,
    const at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, size_t> dL_dws,
    scalar_t* __restrict__ dL_dws_times_ws,
    const at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, size_t> sigmas,
    const at::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, size_t> rgbs,
    const at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, size_t> deltas,
    const at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, size_t> ts,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> rays,
    const at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, size_t> opacity,
    const at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, size_t> depth,
    const at::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, size_t> rgb,
    const scalar_t T_threshold,
    at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, size_t> dL_dsigmas,
    at::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, size_t> dL_drgbs
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= opacity.size(0)) return;

    const int ray_idx = rays[n][0], start_idx = rays[n][1], N_samples = rays[n][2];

    // front to back compositing
    int samples = 0;
    scalar_t R = rgb[ray_idx][0], G = rgb[ray_idx][1], B = rgb[ray_idx][2];
    scalar_t O = opacity[ray_idx], D = depth[ray_idx];
    scalar_t T = 1.0f, r = 0.0f, g = 0.0f, b = 0.0f, d = 0.0f;

    // compute prefix sum of dL_dws * ws
    // [a0, a1, a2, a3, ...] -> [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, ...]
    thrust::inclusive_scan(thrust::device,
                           dL_dws_times_ws+start_idx,
                           dL_dws_times_ws+start_idx+N_samples,
                           dL_dws_times_ws+start_idx);
    scalar_t dL_dws_times_ws_sum = dL_dws_times_ws[start_idx+N_samples-1];

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        r += w*rgbs[s][0]; g += w*rgbs[s][1]; b += w*rgbs[s][2];
        d += w*ts[s];
        T *= 1.0f-a;

        // compute gradients by math...
        dL_drgbs[s][0] = dL_drgb[ray_idx][0]*w;
        dL_drgbs[s][1] = dL_drgb[ray_idx][1]*w;
        dL_drgbs[s][2] = dL_drgb[ray_idx][2]*w;

        dL_dsigmas[s] = deltas[s] * (
            dL_drgb[ray_idx][0]*(rgbs[s][0]*T-(R-r)) + 
            dL_drgb[ray_idx][1]*(rgbs[s][1]*T-(G-g)) + 
            dL_drgb[ray_idx][2]*(rgbs[s][2]*T-(B-b)) + // gradients from rgb
            dL_dopacity[ray_idx]*(1-O) + // gradient from opacity
            dL_ddepth[ray_idx]*(ts[s]*T-(D-d)) + // gradient from depth
            T*dL_dws[s]-(dL_dws_times_ws_sum-dL_dws_times_ws[s]) // gradient from ws
        );

        if (T <= T_threshold) break; // ray has enough opacity
        samples++;
    }
}


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
){
    CHECK_INPUT(dL_dopacity);
    CHECK_INPUT(dL_ddepth);
    CHECK_INPUT(dL_drgb);
    CHECK_INPUT(dL_dws);
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(ws);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays);
    CHECK_INPUT(opacity);
    CHECK_INPUT(depth);
    CHECK_INPUT(rgb);
    const int N = sigmas.size(0), N_rays = rays.size(0);

    auto dL_dsigmas = at::zeros({N}, sigmas.options());
    auto dL_drgbs = at::zeros({N, 3}, sigmas.options());

    auto dL_dws_times_ws = dL_dws * ws; // auxiliary input

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.scalar_type(), "composite_train_bw_cu", 
    ([&] {
        composite_train_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dopacity.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, size_t>(),
            dL_ddepth.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, size_t>(),
            dL_drgb.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, size_t>(),
            dL_dws.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, size_t>(),
            dL_dws_times_ws.data_ptr<scalar_t>(),
            sigmas.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, size_t>(),
            rgbs.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, size_t>(),
            rays.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
            opacity.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, size_t>(),
            depth.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, size_t>(),
            rgb.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, size_t>(),
            T_threshold,
            dL_dsigmas.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, size_t>(),
            dL_drgbs.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, size_t>()
        );
    }));

    return {dL_dsigmas, dL_drgbs};
}

//////////testing

template <typename scalar_t>
__global__ void composite_test_fw_kernel(
    const at::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, size_t> sigmas,
    const at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, size_t> rgbs,
    const at::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, size_t> deltas,
    const at::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, size_t> ts,
    const at::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, size_t> hits_t,
    at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> alive_indices,
    const scalar_t T_threshold,
    const at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> N_eff_samples,
    at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, size_t> opacity,
    at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, size_t> depth,
    at::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, size_t> rgb
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= alive_indices.size(0)) return;

    if (N_eff_samples[n]==0){ // no hit
        alive_indices[n] = -1;
        return;
    }

    const size_t r = alive_indices[n]; // ray index

    // front to back compositing
    int s = 0; scalar_t T = 1-opacity[r];

    while (s < N_eff_samples[n]) {
        const scalar_t a = 1.0f - __expf(-sigmas[n][s]*deltas[n][s]);
        const scalar_t w = a * T;

        rgb[r][0] += w*rgbs[n][s][0];
        rgb[r][1] += w*rgbs[n][s][1];
        rgb[r][2] += w*rgbs[n][s][2];
        depth[r] += w*ts[n][s];
        opacity[r] += w;
        T *= 1.0f-a;

        if (T <= T_threshold){ // ray has enough opacity
            alive_indices[n] = -1;
            break;
        }
        s++;
    }
}


void composite_test_fw_cu(
    const at::Tensor sigmas,
    const at::Tensor rgbs,
    const at::Tensor deltas,
    const at::Tensor ts,
    const at::Tensor hits_t,
    at::Tensor alive_indices,
    const float T_threshold,
    const at::Tensor N_eff_samples,
    at::Tensor opacity,
    at::Tensor depth,
    at::Tensor rgb
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(alive_indices);
    CHECK_INPUT(N_eff_samples);
    CHECK_INPUT(opacity);
    CHECK_INPUT(depth);
    CHECK_INPUT(rgb);
    const int N_rays = alive_indices.size(0);

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.scalar_type(), "composite_test_fw_cu", 
    ([&] {
        composite_test_fw_kernel<scalar_t><<<blocks, threads>>>(
            sigmas.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, size_t>(),
            rgbs.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, size_t>(),
            hits_t.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, size_t>(),
            alive_indices.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            T_threshold,
            N_eff_samples.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
            opacity.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, size_t>(),
            depth.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, size_t>(),
            rgb.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, size_t>()
        );
    }));
}