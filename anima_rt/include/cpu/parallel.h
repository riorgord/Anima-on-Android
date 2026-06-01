// Single-threaded stub for at::parallel_for.
// On mobile we run serially — no OpenMP overhead, deterministic results.
#pragma once
#include <cstdint>
#include <functional>

namespace anima {

inline void parallel_for(int64_t start, int64_t end, int64_t /*grain*/,
                         std::function<void(int64_t, int64_t)> fn) {
    fn(start, end);
}

} // namespace anima
