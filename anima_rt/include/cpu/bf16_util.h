// BF16 <-> FP32 conversion utilities.
// BF16 = IEEE float32 with the lower 16 mantissa bits truncated.
#pragma once
#include <cstdint>
#include <cstring>

namespace anima {

inline float bf16_to_f32(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // Round to nearest even (add 0x7FFF + ((u >> 16) & 1) before truncation)
    u += 0x7FFF + ((u >> 16) & 1);
    return static_cast<uint16_t>(u >> 16);
}

} // namespace anima
