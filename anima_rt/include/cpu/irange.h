// Trivial replacement for c10::irange — used in PyTorch for-loops.
#pragma once
#include <cstdint>

namespace anima {

struct irange {
    int64_t start, end;
    explicit irange(int64_t e) : start(0), end(e) {}
    irange(int64_t s, int64_t e) : start(s), end(e) {}

    struct iterator {
        int64_t val;
        int64_t  operator*() const          { return val; }
        iterator& operator++()              { ++val; return *this; }
        bool      operator!=(const iterator& o) const { return val != o.val; }
    };
    iterator begin() const { return {start}; }
    iterator end()   const { return {end}; }
};

} // namespace anima
