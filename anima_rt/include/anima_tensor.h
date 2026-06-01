// Copyright (c) 2024 Anima RT contributors.
// Minimal tensor abstraction — no autograd, no IR, no threading.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace anima {

enum class DType : uint8_t {
    kFloat32  = 0,
    kBFloat16 = 1,
    kInt8     = 2
};

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::kFloat32:  return "fp32";
        case DType::kBFloat16: return "bf16";
        case DType::kInt8:     return "int8";
    }
    return "???";
}

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::kFloat32:  return 4;
        case DType::kBFloat16: return 2;
        case DType::kInt8:     return 1;
    }
    return 0;
}

struct Tensor {
    void*   data     = nullptr;
    size_t  nbytes   = 0;
    size_t  nelems   = 0;
    int     ndim     = 0;
    int64_t shape[8] = {};
    int64_t strides[8] = {};
    DType   dtype    = DType::kFloat32;
    bool    owns_data = false;

    // Allocate an owned tensor with given shape/dtype (zero-filled).
    static Tensor alloc(const int64_t* sh, int nd, DType dt);

    // Convenience: 2D alloc (row-major).
    static Tensor alloc2d(int64_t d0, int64_t d1, DType dt);

    // Create a non-owning view into existing data.
    static Tensor view(void* ptr, const int64_t* sh, int nd, DType dt);

    // Free underlying buffer if owns_data.
    void free_if_owned();

    // Typed pointer access.
    template<typename T> T*       ptr()       { return static_cast<T*>(data); }
    template<typename T> const T* ptr() const { return static_cast<const T*>(data); }

    // Total elements (产品 of shape[0..ndim-1]).
    size_t numel() const {
        size_t n = 1;
        for (int i = 0; i < ndim; i++) n *= (size_t)shape[i];
        return n;
    }
};

// Compute row-major contiguous strides from shape.
inline void compute_strides(int64_t* strides, const int64_t* shape, int ndim) {
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

} // namespace anima
