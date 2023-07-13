/*
NATTEN1D-AV TORCH EXTENSION (CPU)

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/
#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#if defined(AVX_INT)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif
#include "natten_cpu_commons.h"

namespace natten {

#define GRAIN_SIZE 0

template <int KS, int NS, int DILATION, typename scalar_t>
void natten1dav_cpu_forward_kernel(
    const at::TensorAccessor<scalar_t, 4> attn,
    const at::TensorAccessor<scalar_t, 4> value,
    at::TensorAccessor<scalar_t, 4> out,
    const int length, 
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int dim,
    const int batch_size) {
#if defined(AVX_INT)
    using Vec = at::vec::Vectorized<scalar_t>;
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    at::parallel_for(0, batch_size*heads*length, GRAIN_SIZE, [&](int start, int end) {
    for (int x = start; x < end; x++) {
        int indtmp1 = x/length;
        const int i = x - indtmp1 * length;
        int indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int ni = get_window_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        int attnOffset = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2);
        const int valueOffset = b * value.stride(0) + h * value.stride(1);
        const int outIndex = b*out.stride(0) + h*out.stride(1) + i*out.stride(2);
        scalar_t* _oaddr = out.data() + outIndex;
        for (int xi=ni; xi < ni + KERNEL_SIZE * dilation; xi+=dilation){
            const int valueIndex = valueOffset + xi * value.stride(2);
            scalar_t* _vaddr = value.data() + valueIndex;
            Vec a = Vec(attn.data()[attnOffset]);
            at::vec::map2([a](Vec& x, Vec& y) { return fmadd(a, x, y); }, _oaddr, _vaddr, _oaddr, dim);
            ++attnOffset;
        }
    }});
#else
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    for (int b = 0; b < batch_size; b++) {
        at::parallel_for(0, heads, GRAIN_SIZE, [&](int start, int end) {
        for (int h = start; h < end; h++) {
            for (int i = 0; i < length; i++) {
                const int ni = get_window_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                for (int d = 0; d < dim; d++) {
                    scalar_t updt = scalar_t(0);
                    int attnOffset = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2);
                    const int valueOffset = b * value.stride(0) + h * value.stride(1) + d;
                    for (int xi=ni; xi < ni + KERNEL_SIZE * dilation; xi+=dilation){
                        const int valueIndex = valueOffset + xi * value.stride(2);
                        updt += attn.data()[attnOffset] * value.data()[valueIndex];
                        ++attnOffset;
                    }
                    const int linearIndex = b*out.stride(0) + h*out.stride(1) + i*out.stride(2) + d;
                    out.data()[linearIndex] = updt;
                }
            }
        }});
    }
#endif
}

template <int KS, int NS, int DILATION, typename scalar_t>
void natten1da_cpu_backward_kernel(
    const at::TensorAccessor<scalar_t, 4> d_out,
    at::TensorAccessor<scalar_t, 4> d_attn,
    const at::TensorAccessor<scalar_t, 4> value,
    const int length,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int dim,
    const int batch_size) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    for (int b = 0; b < batch_size; b++) {
        at::parallel_for(0, heads, GRAIN_SIZE, [&](int start, int end) {
        for (int h = start; h < end; h++) {
            for (int i = 0; i < length; i++) {
                const int ni = get_window_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                for (int ki = 0; ki < KERNEL_SIZE; ki++) {
                    scalar_t updt = scalar_t(0);
                    const int batchHeadOffset = b * d_out.stride(0) + h * d_out.stride(1);
                    const int d_outOffset = batchHeadOffset + i * d_out.stride(2);
                    const int valueOffset = batchHeadOffset + (ki*dilation+ni) * value.stride(2);
                    for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                        updt += d_out.data()[d_outOffset+dimOffset] * value.data()[valueOffset+dimOffset];
                    const int index = b * d_attn.stride(0) + h * d_attn.stride(1) + i * d_attn.stride(2) + ki;
                    d_attn.data()[index] = updt;
                }
            }
        }});
    }
}

template <int KS, int NS, int DILATION, typename scalar_t>
void natten1dv_cpu_backward_kernel(
    const at::TensorAccessor<scalar_t, 4> d_out,
    at::TensorAccessor<scalar_t, 4> d_value,
    const at::TensorAccessor<scalar_t, 4> attn,
    const int length,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int dim,
    const int batch_size) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    for (int b = 0; b < batch_size; b++) {
        at::parallel_for(0, heads, GRAIN_SIZE, [&](int start, int end) {
        for (int h = start; h < end; h++) {
            for (int i = 0; i < length; i++) {
                const int ni = get_backward_window_start(i, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int ei = get_backward_window_end(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                for (int d = 0; d < dim; d++) {
                    const int attnOffset = b * attn.stride(0) + h * attn.stride(1);
                    const int outOffset = b * d_out.stride(0) + h * d_out.stride(1) + d;
                    scalar_t d_value_update = scalar_t(0);
                    for (int xi=ni; xi < ei; xi+=dilation){
                        const int oni = get_window_start(xi, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                        const int outIndex = outOffset + xi * d_out.stride(2);
                        const int attnIndex = attnOffset + xi * attn.stride(2) + int((i-oni)/dilation);
                        d_value_update += d_out.data()[outIndex] * attn.data()[attnIndex];
                    }
                    const int linearIndex = b*d_value.stride(0) + h*d_value.stride(1) + i*d_value.stride(2) + d;
                    d_value.data()[linearIndex] = d_value_update;
                }
            }
        }});
    }
}

torch::Tensor natten1dav_cpu_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int dilation) {
    AT_ASSERTM(attn.device().is_cpu(), "attn must be a CPU tensor");
    AT_ASSERTM(value.device().is_cpu(), "value must be a CPU tensor");
    int batch_size = value.size(0);
    int heads = value.size(1);
    int length = value.size(2);
    int dim = value.size(3);
    int kernel_size = attn.size(3);
    CHECK_SEQUENCE(length, kernel_size, dilation);

    auto out = torch::zeros_like(value);

    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "natten1dav_forward_cpu", ([&] {
        LAUNCH_DNA_KNS(kernel_size, dilation, natten1dav_cpu_forward_kernel, 
                attn.accessor<scalar_t, 4>(), value.accessor<scalar_t, 4>(), out.accessor<scalar_t, 4>(), 
                length, heads, kernel_size, dilation, dim, batch_size);
    }));
    return out;
}

std::vector<torch::Tensor> natten1dav_cpu_backward(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int dilation) {
    AT_ASSERTM(d_out.device().is_cpu(), "d_out must be a CPU tensor");
    AT_ASSERTM(attn.device().is_cpu(), "attn must be a CPU tensor");
    AT_ASSERTM(value.device().is_cpu(), "value must be a CPU tensor");
    int batch_size = value.size(0);
    int heads = value.size(1);
    int length = value.size(2);
    int dim = value.size(3);
    int kernel_size = attn.size(3);
    CHECK_SEQUENCE(length, kernel_size, dilation);

    auto d_attn = torch::zeros_like(attn);
    auto d_value = torch::zeros_like(value);

    AT_DISPATCH_FLOATING_TYPES(d_attn.scalar_type(), "natten1dav_backward_cpu", ([&] {
        const auto d_out_a = d_out.accessor<scalar_t, 4>();
        const auto attn_a = attn.accessor<scalar_t, 4>();
        const auto value_a = value.accessor<scalar_t, 4>();
        auto d_attn_a = d_attn.accessor<scalar_t, 4>();
        auto d_value_a = d_value.accessor<scalar_t, 4>();
        LAUNCH_DNA_KNS(kernel_size, dilation, natten1da_cpu_backward_kernel, 
                d_out_a, d_attn_a, value_a, length, heads, kernel_size, dilation, dim, batch_size);
        LAUNCH_DNA_KNS(kernel_size, dilation, natten1dv_cpu_backward_kernel, 
                d_out_a, d_value_a, attn_a, length, heads, kernel_size, dilation, dim, batch_size);
    }));
    return {d_attn, d_value};
}
} // namespace natten
