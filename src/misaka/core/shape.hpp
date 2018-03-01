#pragma once
#include <cassert>
#include <cstdint>

#include <array>
#include <string>
#include <utility>
#include <vector>

struct shape_t {
    template <typename... T>
    explicit shape_t(T... dims) : dims({static_cast<uint32_t>(dims)...})
    {
    }

    explicit shape_t(const std::vector<uint32_t> &dims) : dims(dims) {}

    uint8_t rank() const { return dims.size(); }

    uint32_t dim() const
    {
        // TODO: use std::reduce when it's available
        uint32_t d = 1;
        for (auto i : dims) {
            d *= i;
        }
        return d;
    }

    uint32_t len() const
    {
        if (dims.size() > 0) {
            return dims[0];
        }
        return 1;
    }

    const std::vector<uint32_t> dims;
};

struct shape_list_t {
    std::vector<shape_t> shapes;
    shape_t operator[](int i) const { return shapes[i]; }
};

template <typename T, size_t... i>
auto _index(const std::vector<T> &v, std::index_sequence<i...>)
{
    return std::array<T, sizeof...(i)>({v[i]...});
}

template <uint8_t rank, typename T> auto cast(const std::vector<T> &v)
{
    assert(v.size() == rank);
    return _index(v, std::make_index_sequence<rank>());
}

namespace std
{
inline string to_string(const shape_t &shape)
{
    string buf;
    for (auto d : shape.dims) {
        if (!buf.empty()) {
            buf += ",";
        }
        buf += to_string(d);
    }
    return "shape(" + buf + ")";
}
}
