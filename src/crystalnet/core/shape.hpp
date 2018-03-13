#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <crystalnet/core/error.hpp>
#include <crystalnet/core/gc.hpp>
#include <crystalnet/utility/cast.hpp>

struct shape_t {
    const std::vector<uint32_t> dims;

    template <typename... T>
    explicit shape_t(T... dims) : dims({static_cast<uint32_t>(dims)...})
    {
    }

    explicit shape_t(const std::vector<uint32_t> &dims) : dims(dims) {}

    uint8_t rank() const { return dims.size(); }

    uint32_t dim() const
    {
        return std::accumulate(dims.begin(), dims.end(), 1,
                               std::multiplies<uint32_t>());
    }

    uint32_t len() const
    {
        if (dims.size() > 0) {
            return dims[0];
        }
        return 1;
    }

    shape_t sub() const
    {
        if (dims.size() > 0) {
            return shape_t(std::vector<uint32_t>(dims.begin() + 1, dims.end()));
        }
        return *this;
    }

    shape_t batch(uint32_t n) const
    {
        std::vector<uint32_t> dims({n});
        dims.insert(dims.end(), this->dims.begin(), this->dims.end());
        return shape_t(dims);
    }
};

struct shape_list_t {
    const std::vector<shape_t> shapes;
    explicit shape_list_t(const std::vector<shape_t> &shapes) : shapes(shapes)
    {
    }
    shape_t operator[](int i) const { return shapes[i]; }
    uint8_t size() const { return shapes.size(); }
};

struct shape_ctx_t {
    GC<shape_t> gc0;
    GC<shape_list_t> gc1;
    const shape_t *make_shape(const std::vector<uint32_t> &dims)
    {
        return gc0(new shape_t(dims));
    }
    const shape_list_t *make_shape_list(const std::vector<shape_t> &shapes)
    {
        return gc1(new shape_list_t(shapes));
    }
};

template <uint8_t r> struct ranked_shape_t {
    const std::array<uint32_t, r> dims;

    explicit ranked_shape_t(const std::array<uint32_t, r> &dims) : dims(dims) {}

    template <typename... I> uint32_t idx(I... i) const
    {
        static_assert(sizeof...(i) == r);
        const std::array<uint32_t, r> offs{static_cast<uint32_t>(i)...};
        uint32_t off = 0;
        for (uint8_t i = 0; i < r; ++i) {
            off = off * dims[i] + offs[i];
        }
        return off;
    }
};

template <typename... Dim> ranked_shape_t<sizeof...(Dim)> r_shape(Dim... dim)
{
    constexpr uint8_t r = sizeof...(Dim);
    const std::array<uint32_t, r> dims({static_cast<uint32_t>(dim)...});
    return ranked_shape_t<r>(dims);
}

template <uint8_t r> ranked_shape_t<r> ranked(const shape_t &shape)
{
    check(shape.rank() == r);
    const auto dims = cast<r>(shape.dims);
    return ranked_shape_t<r>(dims);
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
