#include <ATen/ATen.h>
#include <torch/torch.h>

#include "include/utils_math.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define SQRT3 1.73205080757f

///////////////////ray marching utils////////////////////////////////////////////////
inline __host__ __device__ float signf(const float x) { return copysignf(1.0f, x); }
// Example input range of |xyz| and return value of this function
// [0, 0.5) -> 0
// [0.5, 1) -> 1
// [1, 2) -> 2
inline __device__ int mip_from_pos(const float x, const float y, const float z, const int num_cascades) {
    const float mx = fmaxf(fabsf(x), fmaxf(fabsf(y), fabsf(z)));
    int exponent; frexpf(mx, &exponent);
    return min(num_cascades-1, max(0, exponent+1));
}

// Example input range of dt and return value of this function
// [0, 1/grid_size) -> 0
// [1/grid_size, 2/grid_size) -> 1
// [2/grid_size, 4/grid_size) -> 2
inline __device__ int mip_from_dt(float dt, int grid_size, int num_cascades) {
    int exponent; frexpf(dt*grid_size, &exponent);
    return min(num_cascades-1, max(0, exponent));
}

////////////////Occupancy grid utils//////////////////////////////////////////////////

inline __host__ __device__ uint32_t __expand_bits(uint32_t v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline __host__ __device__ uint32_t __morton3D(uint32_t x, uint32_t y, uint32_t z)
{
    uint32_t xx = __expand_bits(x);
    uint32_t yy = __expand_bits(y);
    uint32_t zz = __expand_bits(z);
    return xx | (yy << 1) | (zz << 2);
}

inline __host__ __device__ uint32_t __morton3D_invert(uint32_t x)
{
    x = x & 0x49249249;
    x = (x | (x >> 2)) & 0xc30c30c3;
    x = (x | (x >> 4)) & 0x0f00f00f;
    x = (x | (x >> 8)) & 0xff0000ff;
    x = (x | (x >> 16)) & 0x0000ffff;
    return x;
}

__global__ void morton3D_kernel(
    const at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> coords,
    at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> indices
){
    const int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= coords.size(0)) return;

    indices[n] = __morton3D(coords[n][0], coords[n][1], coords[n][2]);
}

at::Tensor morton3D_cu(const at::Tensor coords){
    CHECK_INPUT(coords);
    int N = coords.size(0);

    auto indices = at::zeros({N}, coords.options());

    const int threads = 256, blocks = (N+threads-1)/threads;

    AT_DISPATCH_INTEGRAL_TYPES(coords.type(), "morton3D_cu", // made change here scalar_
    ([&] {
        morton3D_kernel<<<blocks, threads>>>(
            coords.packed_accessor32<int, 2, at::RestrictPtrTraits>(),
            indices.packed_accessor32<int, 1, at::RestrictPtrTraits>()
        );
    }));

    return indices;
}

__global__ void morton3D_invert_kernel(
    const at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> indices,
    at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> coords
){
    const int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= coords.size(0)) return;

    const int ind = indices[n];
    coords[n][0] = __morton3D_invert(ind >> 0);
    coords[n][1] = __morton3D_invert(ind >> 1);
    coords[n][2] = __morton3D_invert(ind >> 2);
}

at::Tensor morton3D_invert_cu(const at::Tensor indices){
    CHECK_INPUT(indices);
    int N = indices.size(0);

    auto coords = at::zeros({N, 3}, indices.options());

    const int threads = 256, blocks = (N+threads-1)/threads;

    AT_DISPATCH_INTEGRAL_TYPES(indices.type(), "morton3D_invert_cu", //made change here scalar_
    ([&] {
        morton3D_invert_kernel<<<blocks, threads>>>(
            indices.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
            coords.packed_accessor32<int, 2, at::RestrictPtrTraits>()
        );
    }));

    return coords;
}

// packbits utils
template <typename scalar_t>
__global__ void packbits_kernel(
    const scalar_t* __restrict__ density_grid,
    const int N,
    const float density_threshold,
    uint8_t* __restrict__ density_bitfield
){
    // parallel per byte
    const int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    uint8_t bits = 0;

    #pragma unroll 8
    for (uint8_t i = 0; i < 8; i++) {
        bits |= (density_grid[8*n+i]>density_threshold) ? ((uint8_t)1<<i) : 0;
    }

    density_bitfield[n] = bits;
}

void packbits_cu(
    const at::Tensor density_grid,
    const float density_threshold,
    at::Tensor density_bitfield
){
    CHECK_INPUT(density_grid);
    CHECK_INPUT(density_bitfield);
    const int N = density_bitfield.size(0);

    const int threads = 256, blocks = (N+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(density_grid.scalar_type(), "packbits_cu", 
    ([&] {
        packbits_kernel<scalar_t><<<blocks, threads>>>(
            density_grid.data_ptr<scalar_t>(),
            N,
            density_threshold,
            density_bitfield.data_ptr<uint8_t>()
        );
    }));
}

/////////////////// ray aabb intersection ////////////////////////////////

__device__ __forceinline__ float2 _ray_aabb_intersect(
    const float3 ray_o,
    const float3 inv_d,
    const float3 center,
    const float3 half_size
){

    const float3 t_min = (center-half_size-ray_o)*inv_d;
    const float3 t_max = (center+half_size-ray_o)*inv_d;

    const float3 _t1 = fminf(t_min, t_max);
    const float3 _t2 = fmaxf(t_min, t_max);
    const float t1 = fmaxf(fmaxf(_t1.x, _t1.y), _t1.z);
    const float t2 = fminf(fminf(_t2.x, _t2.y), _t2.z);

    if (t1 > t2) return make_float2(-1.0f); // no intersection
    return make_float2(t1, t2);
}


__global__ void ray_aabb_intersect_kernel(
    const at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> rays_o,
    const at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> rays_d,
    const at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> centers,
    const at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> half_sizes,
    const int max_hits,
    int* __restrict__ hit_cnt,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> hits_t,
    at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> hits_voxel_idx
){
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (v>=centers.size(0) || r>=rays_o.size(0)) return;

    const float3 ray_o = make_float3(rays_o[r][0], rays_o[r][1], rays_o[r][2]);
    const float3 ray_d = make_float3(rays_d[r][0], rays_d[r][1], rays_d[r][2]);
    const float3 inv_d = 1.0f/ray_d;

    const float3 center = make_float3(centers[v][0], centers[v][1], centers[v][2]);
    const float3 half_size = make_float3(half_sizes[v][0], half_sizes[v][1], half_sizes[v][2]);
    const float2 t1t2 = _ray_aabb_intersect(ray_o, inv_d, center, half_size);

    if (t1t2.y > 0){ // if ray hits the voxel
        const int cnt = atomicAdd(&hit_cnt[r], 1);
        if (cnt < max_hits){
            hits_t[r][cnt][0] = fmaxf(t1t2.x, 0.0f);
            hits_t[r][cnt][1] = t1t2.y;
            hits_voxel_idx[r][cnt] = v;
        }
    }
}


std::vector<at::Tensor> ray_aabb_intersect_cu(
    const at::Tensor rays_o,
    const at::Tensor rays_d,
    const at::Tensor centers,
    const at::Tensor half_sizes,
    const int max_hits
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(centers);
    CHECK_INPUT(half_sizes);
    const int N_rays = rays_o.size(0), N_voxels = centers.size(0);
    auto hits_t = at::zeros({N_rays, max_hits, 2}, rays_o.options())-1;
    auto hits_voxel_idx = 
        at::zeros({N_rays, max_hits}, 
                     at::dtype(at::kLong).device(rays_o.device()))-1;
    // auto hit_cnt = 
    //     at::zeros({N_rays}, 
    //                  at::dtype(at::kInt32).device(rays_o.device()));
    auto hit_cnt = 
        at::zeros({N_rays}, 
                     at::dtype(at::kInt).device(rays_o.device()));

    const dim3 threads(256, 1);
    const dim3 blocks((N_rays+threads.x-1)/threads.x,
                      (N_voxels+threads.y-1)/threads.y);
    
    AT_DISPATCH_FLOATING_TYPES(rays_o.scalar_type(), "ray_aabb_intersect_cu", 
    ([&] {
        ray_aabb_intersect_kernel<<<blocks, threads>>>(
            rays_o.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
            rays_d.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
            centers.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
            half_sizes.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
            max_hits,
            hit_cnt.data_ptr<int>(),
            hits_t.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
            hits_voxel_idx.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>()
        );
    }));

    // sort intersections from near to far based on t1
    auto hits_order = std::get<1>(at::sort(hits_t.index({"...", 0})));
    hits_voxel_idx = at::gather(hits_voxel_idx, 1, hits_order);
    hits_t = at::gather(hits_t, 1, hits_order.unsqueeze(-1).tile({1, 1, 2}));

    return {hit_cnt, hits_t, hits_voxel_idx};
}

//////////////// ray marching ////////////////////////////////////////////
__global__ void raymarching_train_kernel(
    const at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> rays_o,
    const at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> rays_d,
    const at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> hits_t,
    // const at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> nears,
    // const at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> fars,
    const uint8_t* __restrict__ density_bitfield,
    const int num_cascades,
    const int grid_size,
    const float scale,
    const at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> noise,
    const int max_samples,
    int* __restrict__ counter,//int * counter,
    at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> rays,
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> xyzs,
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> dirs,
    at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> deltas,
    at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> ts
){
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rays_o.size(0)) return;

    const uint32_t grid_size3 = grid_size*grid_size*grid_size;
    const float grid_size_inv = 1.0f/grid_size;

    const float ox = rays_o[r][0], oy = rays_o[r][1], oz = rays_o[r][2];
    const float dx = rays_d[r][0], dy = rays_d[r][1], dz = rays_d[r][2];
    const float dx_inv = 1.0f/dx, dy_inv = 1.0f/dy, dz_inv = 1.0f/dz;

    // float t1 = nears[r];
    // float t2 = fars[r];
    float t1 = hits_t[r][0], t2 = hits_t[r][1];

    const float dt = SQRT3/max_samples;

    if (t1>=0){
        // const float dt = calc_dt(t1, exp_step_factor, max_samples, grid_size, scale);
        t1 += dt*noise[r];
    }

    // first pass to compute the number of samples on the ray
    float t = t1; int N_samples = 0;

    // if t1 < 0 (no hit) this loop will be skipped (N_samples will be 0)
    while (0<=t && t<t2 && N_samples<max_samples){
        const float x = ox + t*dx, y = oy + t*dy, z = oz + t*dz;
        // const float dt = calc_dt(t, exp_step_factor, max_samples, grid_size, scale);
        const int mip = max(mip_from_pos(x, y, z, num_cascades), mip_from_dt(dt, grid_size, num_cascades));
        const float mip_bound = fminf(scalbnf(1.0f, mip-1), scale);
        const float mip_bound_inv = 1/mip_bound;

        // round down to nearest grid position
        const int nx = clamp(0.5f*(x*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);
        const int ny = clamp(0.5f*(y*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);
        const int nz = clamp(0.5f*(z*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);

        const uint32_t idx = mip*grid_size3 + __morton3D(nx, ny, nz);
        const bool occ = density_bitfield[idx/8] & (1<<(idx%8));

        if (occ) {
            t += dt; N_samples++;
        } else { // skip until the next voxel
            const float tx = (((nx+0.5f+0.5f*signf(dx))*grid_size_inv*2-1)*mip_bound-x)*dx_inv;
            const float ty = (((ny+0.5f+0.5f*signf(dy))*grid_size_inv*2-1)*mip_bound-y)*dy_inv;
            const float tz = (((nz+0.5f+0.5f*signf(dz))*grid_size_inv*2-1)*mip_bound-z)*dz_inv;

            const float t_target = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            do {
                // t += calc_dt(t, exp_step_factor, max_samples, grid_size, scale);
                t += dt;
            } while (t < t_target);
        }
    }

    // second pass to write the output
    const int start_idx = atomicAdd(counter, N_samples);
    const int ray_count = atomicAdd(counter+1, 1);

    rays[ray_count][0] = r;
    rays[ray_count][1] = start_idx; rays[ray_count][2] = N_samples;

    t = t1; int samples = 0; float last_t = t1;

    while (t<t2 && samples<N_samples){
        const float x = ox + t*dx, y = oy + t*dy, z = oz + t*dz;
        // const float dt = calc_dt(t, exp_step_factor, max_samples, grid_size, scale);
        const int mip = max(mip_from_pos(x, y, z, num_cascades), mip_from_dt(dt, grid_size, num_cascades));
        const float mip_bound = fminf(scalbnf(1.0f, mip-1), scale);
        const float mip_bound_inv = 1/mip_bound;

        // round down to nearest grid position
        const int nx = clamp(0.5f*(x*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);
        const int ny = clamp(0.5f*(y*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);
        const int nz = clamp(0.5f*(z*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);

        const uint32_t idx = mip*grid_size3 + __morton3D(nx, ny, nz);
        const bool occ = density_bitfield[idx/8] & (1<<(idx%8));

        if (occ) {
            const int s = start_idx + samples;
            xyzs[s][0] = x; xyzs[s][1] = y; xyzs[s][2] = z;
            dirs[s][0] = dx; dirs[s][1] = dy; dirs[s][2] = dz;
            ts[s] = t; deltas[s] = t - last_t;//deltas[s] = dt;
            last_t = t;
            t += dt; samples++;
        } else { // skip until the next voxel
            const float tx = (((nx+0.5f+0.5f*signf(dx))*grid_size_inv*2-1)*mip_bound-x)*dx_inv;
            const float ty = (((ny+0.5f+0.5f*signf(dy))*grid_size_inv*2-1)*mip_bound-y)*dy_inv;
            const float tz = (((nz+0.5f+0.5f*signf(dz))*grid_size_inv*2-1)*mip_bound-z)*dz_inv;

            const float t_target = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            do {
                // t += calc_dt(t, exp_step_factor, max_samples, grid_size, scale);
                t += dt;
            } while (t < t_target);
        }
    }
}


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
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    // CHECK_INPUT(nears);
    // CHECK_INPUT(fars);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(density_bitfield);
    CHECK_INPUT(noise);
    const int N_rays = rays_o.size(0);

    // count the number of samples and the number of rays processed
    // auto counter = at::zeros({2}, at::dtype(at::kInt32).device(rays_o.device()));
    auto counter = at::zeros({2}, at::dtype(at::kInt).device(rays_o.device()));
    // ray attributes: ray_idx, start_idx, N_samples
    auto rays = at::zeros({N_rays, 3}, at::dtype(at::kLong).device(rays_o.device()));
    auto xyzs = at::zeros({N_rays*max_samples, 3}, rays_o.options());
    auto dirs = at::zeros({N_rays*max_samples, 3}, rays_o.options());
    auto deltas = at::zeros({N_rays*max_samples}, rays_o.options());
    auto ts = at::zeros({N_rays*max_samples}, rays_o.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads; // HOW???

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(rays_o.scalar_type(), "raymarching_train_cu", 
    ([&] {
        raymarching_train_kernel<<<blocks, threads>>>(
            rays_o.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
            rays_d.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
            // nears.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
            // fars.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
            hits_t.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
            density_bitfield.data_ptr<uint8_t>(),
            num_cascades,
            grid_size,
            scale,
            noise.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
            max_samples,
            counter.data_ptr<int>(),
            rays.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
            xyzs.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
            dirs.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
            deltas.packed_accessor32<float, 1, at::RestrictPtrTraits>(),
            ts.packed_accessor32<float, 1, at::RestrictPtrTraits>()
        );
    }));

    return {rays, xyzs, dirs, deltas, ts, counter};
}
// Test
__global__ void raymarching_test_kernel(
    const at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> rays_o,
    const at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> rays_d,
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> hits_t,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> alive_indices,
    const uint8_t* __restrict__ density_bitfield,
    const int num_cascades,
    const int grid_size,
    const float scale,
    const int N_samples,
    const int max_samples,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> xyzs,
    at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> dirs,
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> deltas,
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> ts,
    at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> N_eff_samples
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= alive_indices.size(0)) return;

    const size_t r = alive_indices[n]; // ray index
    const uint32_t grid_size3 = grid_size*grid_size*grid_size;
    const float grid_size_inv = 1.0f/grid_size;

    const float ox = rays_o[r][0], oy = rays_o[r][1], oz = rays_o[r][2];
    const float dx = rays_d[r][0], dy = rays_d[r][1], dz = rays_d[r][2];
    const float dx_inv = 1.0f/dx, dy_inv = 1.0f/dy, dz_inv = 1.0f/dz;

    float t = hits_t[r][0], t2 = hits_t[r][1];
    const float dt = SQRT3/max_samples;
    int s = 0;

    while (t<t2 && s<N_samples){
        const float x = ox+t*dx, y = oy+t*dy, z = oz+t*dz;

        // const float dt = calc_dt(t, exp_step_factor, max_samples, grid_size, cascades);
        const int mip = max(mip_from_pos(x, y, z, num_cascades),
                            mip_from_dt(dt, grid_size, num_cascades));

        const float mip_bound = fminf(scalbnf(1.0f, mip-1), scale);
        const float mip_bound_inv = 1/mip_bound;

        // round down to nearest grid position
        const int nx = clamp(0.5f*(x*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);
        const int ny = clamp(0.5f*(y*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);
        const int nz = clamp(0.5f*(z*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);

        const uint32_t idx = mip*grid_size3 + __morton3D(nx, ny, nz);
        const bool occ = density_bitfield[idx/8] & (1<<(idx%8));

        if (occ) {
            xyzs[n][s][0] = x; xyzs[n][s][1] = y; xyzs[n][s][2] = z;
            dirs[n][s][0] = dx; dirs[n][s][1] = dy; dirs[n][s][2] = dz;
            ts[n][s] = t; deltas[n][s] = dt;
            t += dt;
            hits_t[r][0] = t; // modify the starting point for the next marching
            s++;
        } else { // skip until the next voxel
            const float tx = (((nx+0.5f+0.5f*signf(dx))*grid_size_inv*2-1)*mip_bound-x)*dx_inv;
            const float ty = (((ny+0.5f+0.5f*signf(dy))*grid_size_inv*2-1)*mip_bound-y)*dy_inv;
            const float tz = (((nz+0.5f+0.5f*signf(dz))*grid_size_inv*2-1)*mip_bound-z)*dz_inv;

            const float t_target = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            do {
                // t += calc_dt(t, exp_step_factor, max_samples, grid_size, cascades);
                t += dt;
            } while (t < t_target);
        }
    }
    N_eff_samples[n] = s; // effective samples that hit occupied region (<=N_samples)
}


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
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(alive_indices);
    CHECK_INPUT(density_bitfield);
    const int N_rays = alive_indices.size(0);

    auto xyzs = at::zeros({N_rays, N_samples, 3}, rays_o.options());
    auto dirs = at::zeros({N_rays, N_samples, 3}, rays_o.options());
    auto deltas = at::zeros({N_rays, N_samples}, rays_o.options());
    auto ts = at::zeros({N_rays, N_samples}, rays_o.options());
    auto N_eff_samples = at::zeros({N_rays},
                            at::dtype(at::kInt).device(rays_o.device()));

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(rays_o.scalar_type(), "raymarching_test_cu", 
    ([&] {
        raymarching_test_kernel<<<blocks, threads>>>(
            rays_o.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
            rays_d.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
            hits_t.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
            alive_indices.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            density_bitfield.data_ptr<uint8_t>(),
            num_cascades,
            grid_size,
            scale,
            N_samples,
            max_samples,
            xyzs.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
            dirs.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
            deltas.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
            ts.packed_accessor32<float, 2, at::RestrictPtrTraits>(),
            N_eff_samples.packed_accessor32<int, 1, at::RestrictPtrTraits>()
        );
    }));

    return {xyzs, dirs, deltas, ts, N_eff_samples};
}