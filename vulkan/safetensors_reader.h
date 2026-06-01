// Safetensors C++ Reader — header-only, no dependencies
//
// Parses the safetensors JSON header and provides pread-based data access.
// BF16 weights are read as-is (uint16_t[] with BF16 bit pattern),
// no conversion to FP16 — shaders unpack BF16→FP32 via uintBitsToFloat(x << 16).
//
// Usage:
//   SafetensorsReader reader;
//   reader.open("/sdcard/anima_on_android/models/diffusion.safetensors");
//   for (auto& key : reader.keys()) {
//       auto& info = reader.info(key);
//       // info.dtype == "BF16" / "F16" / "F32"
//       // info.shape = vector of dimensions
//       // read data: pread(reader.fd(), buf, info.data_len, reader.data_start() + info.data_offsets[0]);
//   }
//   reader.close();

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cerrno>

// ── Minimal JSON string-view parser (no allocation, no dependencies) ──
namespace safetensors_json {

static inline void skip_ws(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
}

static inline std::string parse_string(const char*& p, const char* end) {
    if (p >= end || *p != '"') throw std::runtime_error("Expected '\"'");
    p++;
    std::string s;
    while (p < end && *p != '"') {
        if (*p == '\\') { p++; if (p < end) { s += *p++; } }
        else { s += *p++; }
    }
    if (p < end) p++; // skip closing '"'
    return s;
}

static inline int64_t parse_int(const char*& p, const char* end) {
    bool neg = false;
    if (p < end && *p == '-') { neg = true; p++; }
    int64_t v = 0;
    while (p < end && *p >= '0' && *p <= '9') { v = v * 10 + (*p++ - '0'); }
    return neg ? -v : v;
}

static inline std::vector<int64_t> parse_int_array(const char*& p, const char* end) {
    std::vector<int64_t> arr;
    if (p >= end || *p != '[') throw std::runtime_error("Expected '['");
    p++;
    skip_ws(p, end);
    if (*p == ']') { p++; return arr; }
    for (;;) {
        skip_ws(p, end);
        arr.push_back(parse_int(p, end));
        skip_ws(p, end);
        if (*p == ']') { p++; break; }
        if (*p == ',') { p++; continue; }
        break;
    }
    return arr;
}

} // namespace safetensors_json

// ── Tensor metadata ──
struct TensorInfo {
    std::string dtype;            // "BF16", "F16", "F32"
    std::vector<uint32_t> shape;  // dimensions
    std::vector<size_t> data_offsets; // [start, end] relative to data_start
    size_t data_len;              // data_offsets[1] - data_offsets[0]
};

// ── Reader ──
class SafetensorsReader {
public:
    SafetensorsReader() = default;

    bool open(const char* path) {
        fd_ = ::open(path, O_RDONLY);
        if (fd_ < 0) {
            fprintf(stderr, "SafetensorsReader: cannot open %s (errno=%d)\n", path, errno);
            return false;
        }

        // Read 8-byte header length (little-endian uint64)
        uint64_t header_len;
        if (::pread(fd_, &header_len, 8, 0) != 8) {
            fprintf(stderr, "SafetensorsReader: cannot read header length\n");
            close(); return false;
        }

        // Read JSON header
        std::vector<char> json_buf(header_len + 1);
        if (::pread(fd_, json_buf.data(), header_len, 8) != (ssize_t)header_len) {
            fprintf(stderr, "SafetensorsReader: cannot read header\n");
            close(); return false;
        }
        json_buf[header_len] = '\0';
        data_start_ = 8 + header_len;

        // Parse JSON
        if (!parse_header(json_buf.data(), header_len)) {
            fprintf(stderr, "SafetensorsReader: JSON parse failed\n");
            close(); return false;
        }

        return true;
    }

    void close() {
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        header_.clear();
    }

    int fd() const { return fd_; }
    size_t data_start() const { return data_start_; }

    const std::vector<std::string>& keys() const { return keys_; }

    const TensorInfo& info(const std::string& key) const {
        static TensorInfo empty;
        auto it = header_.find(key);
        return it != header_.end() ? it->second : empty;
    }

    bool read_tensor(const std::string& key, void* buf, size_t buf_size) const {
        auto it = header_.find(key);
        if (it == header_.end()) return false;
        size_t off = data_start_ + it->second.data_offsets[0];
        size_t len = it->second.data_len;
        if (buf_size < len) return false;
        return ::pread(fd_, buf, len, (off_t)off) == (ssize_t)len;
    }

    // Detect and return prefix to strip (e.g. "net."), or "" if none.
    // Call after open().
    std::string detect_prefix() const {
        if (keys_.empty()) return "";
        // Skip __metadata__
        const std::string* first = nullptr;
        for (auto& k : keys_) {
            if (k != "__metadata__") { first = &k; break; }
        }
        if (!first) return "";
        auto dot = first->find('.');
        if (dot == std::string::npos) return "";
        std::string candidate = first->substr(0, dot + 1); // e.g. "net."
        // Verify at least half of non-block keys start with this prefix
        int match = 0, total = 0;
        for (auto& k : keys_) {
            if (k == "__metadata__") continue;
            total++;
            if (k.compare(0, candidate.size(), candidate) == 0) match++;
        }
        if (match * 2 >= total) return candidate;
        return "";
    }

private:
    int fd_ = -1;
    size_t data_start_ = 0;
    std::unordered_map<std::string, TensorInfo> header_;
    std::vector<std::string> keys_;

    bool parse_header(const char* json, size_t /*len*/) {
        using namespace safetensors_json;
        const char* p = json;
        const char* end = p + strlen(json);

        skip_ws(p, end);
        if (p >= end || *p != '{') return false;
        p++; // skip '{'

        while (p < end) {
            skip_ws(p, end);
            if (*p == '}') { p++; break; }
            if (*p == ',') { p++; skip_ws(p, end); }

            // Parse key string
            if (*p != '"') return false;
            std::string key = parse_string(p, end);
            skip_ws(p, end);
            if (*p != ':') return false;
            p++; // skip ':'
            skip_ws(p, end);

            if (*p == '{') {
                // Tensor info object
                p++; // skip '{'
                TensorInfo info;
                while (p < end) {
                    skip_ws(p, end);
                    if (*p == '}') { p++; break; }
                    if (*p == ',') { p++; skip_ws(p, end); }
                    if (*p != '"') return false;
                    std::string fname = parse_string(p, end);
                    skip_ws(p, end);
                    if (*p != ':') return false;
                    p++; // skip ':'
                    skip_ws(p, end);

                    if (fname == "dtype") {
                        info.dtype = parse_string(p, end);
                    } else if (fname == "shape") {
                        auto arr = parse_int_array(p, end);
                        for (auto v : arr) info.shape.push_back((uint32_t)v);
                    } else if (fname == "data_offsets") {
                        auto arr = parse_int_array(p, end);
                        for (auto v : arr) info.data_offsets.push_back((size_t)v);
                    } else {
                        // Unknown field — skip its value
                        if (*p == '"') { parse_string(p, end); }
                        else if (*p == '[') { parse_int_array(p, end); }
                        else if (*p == '{') { /* nested object — skip */ p++; int depth=1; while(p<end && depth){if(*p=='{')depth++;if(*p=='}')depth--;p++;} }
                    }
                }
                if (info.data_offsets.size() >= 2) {
                    info.data_len = info.data_offsets[1] - info.data_offsets[0];
                }
                header_[key] = std::move(info);
                keys_.push_back(key);
            } else if (*p == '"') {
                // Metadata string value — skip
                parse_string(p, end);
                // Don't add __metadata__ to keys_ unless there's actual data
                // Actually, let's skip it from keys_ but keep it in header_ for completeness
                if (key != "__metadata__") keys_.push_back(key);
            } else {
                // Unknown — skip
                return false;
            }
        }
        return true;
    }
};
