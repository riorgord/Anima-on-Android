// Copyright (c) 2024 Anima RT contributors.
#include "anima_tensor.h"
#include <cstdlib>
#include <cstring>

namespace anima {

Tensor Tensor::alloc(const int64_t* sh, int nd, DType dt) {
    size_t ne = 1;
    for (int i = 0; i < nd; i++) ne *= (size_t)sh[i];
    size_t nb = ne * dtype_size(dt);
    void* ptr = std::calloc(1, nb);
    Tensor t;
    t.data     = ptr;
    t.nbytes   = nb;
    t.nelems   = ne;
    t.ndim     = nd;
    t.dtype    = dt;
    t.owns_data = true;
    for (int i = 0; i < nd; i++) t.shape[i] = sh[i];
    compute_strides(t.strides, t.shape, nd);
    return t;
}

Tensor Tensor::alloc2d(int64_t d0, int64_t d1, DType dt) {
    int64_t sh[2] = {d0, d1};
    return alloc(sh, 2, dt);
}

Tensor Tensor::view(void* ptr, const int64_t* sh, int nd, DType dt) {
    size_t ne = 1;
    for (int i = 0; i < nd; i++) ne *= (size_t)sh[i];
    Tensor t;
    t.data   = ptr;
    t.nbytes = ne * dtype_size(dt);
    t.nelems = ne;
    t.ndim   = nd;
    t.dtype  = dt;
    t.owns_data = false;
    for (int i = 0; i < nd; i++) t.shape[i] = sh[i];
    compute_strides(t.strides, t.shape, nd);
    return t;
}

void Tensor::free_if_owned() {
    if (owns_data && data) {
        std::free(data);
        data = nullptr;
        owns_data = false;
    }
}

} // namespace anima
