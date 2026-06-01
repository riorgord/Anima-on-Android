// Scalar fallback for PyTorch's vec::Vectorized<T> — identical API, size()=1.
// Every "vectorized" operation becomes a scalar loop, preserving the EXACT
// algorithm structure from PyTorch but using plain scalar arithmetic.
#pragma once
#include <cmath>
#include <cstring>
#include <utility>

namespace anima {
namespace vec {

template<typename T>
struct Vectorized {
    T val;
    static constexpr int size() { return 1; }

    Vectorized() : val(0) {}
    explicit Vectorized(T v) : val(v) {}

    static Vectorized loadu(const T* ptr)                  { return Vectorized(ptr[0]); }
    static Vectorized loadu(const T* ptr, int64_t /*count*/) { return Vectorized(ptr[0]); }
    void store(T* ptr) const                               { ptr[0] = val; }
    void store(T* ptr, int64_t /*count*/) const            { ptr[0] = val; }

    Vectorized operator+(const Vectorized& o) const { return Vectorized(val + o.val); }
    Vectorized operator-(const Vectorized& o) const { return Vectorized(val - o.val); }
    Vectorized operator*(const Vectorized& o) const { return Vectorized(val * o.val); }
    Vectorized operator/(const Vectorized& o) const { return Vectorized(val / o.val); }
    Vectorized operator-() const                    { return Vectorized(-val); }
    Vectorized& operator+=(const Vectorized& o)     { val += o.val; return *this; }

    Vectorized exp()  const { return Vectorized(std::exp(val)); }
    Vectorized sqrt() const { return Vectorized(std::sqrt(val)); }
    Vectorized tanh() const { return Vectorized(std::tanh(val)); }
    Vectorized erf()  const { return Vectorized(std::erf(val)); }
};

// ---- map / reduce helpers (mirror PT's vec::map / vec::reduce_all) ----

template<typename T, typename Func>
inline void map(Func f, T* dst, const T* src, int64_t n) {
    for (int64_t i = 0; i < n; i++)
        dst[i] = f(Vectorized<T>(src[i])).val;
}

template<typename T, typename Func>
inline void map2(Func f, T* dst, const T* src1, const T* src2, int64_t n) {
    for (int64_t i = 0; i < n; i++)
        dst[i] = f(Vectorized<T>(src1[i]), Vectorized<T>(src2[i])).val;
}

template<typename T, typename Func>
inline void map3(Func f, T* dst, const T* src1, const T* src2, const T* src3, int64_t n) {
    for (int64_t i = 0; i < n; i++)
        dst[i] = f(Vectorized<T>(src1[i]), Vectorized<T>(src2[i]), Vectorized<T>(src3[i])).val;
}

template<typename T, typename Func>
inline T reduce_all(Func f, const T* data, int64_t n) {
    Vectorized<T> acc(data[0]);
    for (int64_t i = 1; i < n; i++)
        acc = f(acc, Vectorized<T>(data[i]));
    return acc.val;
}

template<typename T>
inline Vectorized<T> maximum(Vectorized<T> a, Vectorized<T> b) {
    return Vectorized<T>(a.val > b.val ? a.val : b.val);
}

template<typename T>
inline Vectorized<T> fmadd(Vectorized<T> a, Vectorized<T> b, Vectorized<T> c) {
    return Vectorized<T>(a.val * b.val + c.val);
}

// ---- convert_to_float / convert_from_float ----
// For float: identity. For BF16/Half: expand to two float vectors.
// Since our Vectorized size is 1, convert_to_float returns {vec(float(x)), vec(0)}.

template<typename T>
inline std::pair<Vectorized<float>, Vectorized<float>> convert_to_float(Vectorized<T> x) {
    return {Vectorized<float>(static_cast<float>(x.val)), Vectorized<float>(0.0f)};
}

template<>
inline std::pair<Vectorized<float>, Vectorized<float>> convert_to_float<float>(Vectorized<float> x) {
    return {x, Vectorized<float>(0.0f)};
}

template<typename T>
inline T convert_from_float(Vectorized<float> a, Vectorized<float> /*b*/) {
    return static_cast<T>(a.val);
}

template<>
inline float convert_from_float<float>(Vectorized<float> a, Vectorized<float> /*b*/) {
    return a.val;
}

} // namespace vec
} // namespace anima
