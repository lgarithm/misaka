#pragma once
#include <algorithm>
#include <limits>

#include <crystalnet/core/operator.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/ops/batch.hpp>
#include <crystalnet/utility/cast.hpp>
#include <crystalnet/utility/range.hpp>

struct pool2d_c {
    constexpr static uint8_t arity = 1;

    struct trait_t {
        ranked_shape_t<2> filter;
        ranked_shape_t<2> stride;

        trait_t() : filter(r_shape(2, 2)), stride(filter) {}

        trait_t(const ranked_shape_t<2> &filter)
            : filter(filter), stride(filter)
        {
        }

        trait_t(const ranked_shape_t<2> &filter,
                const ranked_shape_t<2> &stride)
            : filter(filter), stride(stride)
        {
        }
    };

    static uint32_t output_size(uint32_t input_size, uint32_t filter_size,
                                uint32_t stride)
    {
        check(input_size >= filter_size);
        check((input_size - filter_size) % stride == 0);
        return (input_size - filter_size) / stride + 1;
    }

    // [w, h, c] -> [w', h', c]
    static shape_t infer(const shape_list_t &shape_list,
                         const trait_t &t = trait_t())
    {
        const auto[p] = cast<1>(shape_list.shapes);
        const auto[h, w, c] = cast<3>(p.dims);
        const auto h_ = output_size(h, t.filter.dims[0], t.stride.dims[0]);
        const auto w_ = output_size(w, t.filter.dims[1], t.stride.dims[1]);
        return shape_t(h_, w_, c);
    }

    using T = float; // TODO: cast based on dtype

    struct forward : forward_ctx_t {
        void operator()(const trait_t &t) const
        {
            const auto x = ranked<3, T>(inputs[0]);
            const auto y = ranked<3, T>(output);
            const auto[h, w, c] = x.shape.dims;
            const auto[_u, v, _c] = y.shape.dims;
            const auto[r, s] = t.filter.dims;
            const auto[stride_r, stride_s] = t.stride.dims;

            for (auto k : range(c)) {
                for (auto i : range(h)) {
                    for (auto j : range(w)) {
                        T yy = std::numeric_limits<T>::min();
                        for (auto u : range(r)) {
                            for (auto v : range(s)) {
                                yy = std::max(yy, x.at(i * stride_r + u,
                                                       j * stride_s + v, k));
                            }
                        }
                        y.at(i, j, k) = yy;
                    }
                }
            }
        }
    };

    struct backward : backward_ctx_t {
        void operator()(const trait_t &t) const
        {
            const auto gx = ranked<3, T>(input_gradients[0]);
            const auto gy = ranked<3, T>(output_gradient);
            const auto[h, w, c] = gx.shape.dims;
            const auto[u, v, _c] = gy.shape.dims;
            const auto[r, s] = t.filter.dims;
            const auto[stride_r, stride_s] = t.stride.dims;
            r_tensor_ref_t<T>(output_gradient).fill(0); // gy.fill(0);
            // for (auto p : range(u)) {
            //     for (auto q : range(v)) {
            //         for (auto i : range(p * r, p * r + r)) {
            //             for (auto j : range(q * s, q * s + s)) {
            //                 for (auto k : range(c)) {
            //                     // TODO: only assign to max
            //                     gx.at(i, j, k) += gy.at(p, q, k);
            //                 }
            //             }
            //         }
            //     }
            // }
        }
    };
};

struct op_pool2d_impl_t {
    const pool2d_c::trait_t t;

    op_pool2d_impl_t(const pool2d_c::trait_t &t) : t(t) {}

    shape_t infer(const shape_list_t &shape_list) const
    {
        return pool2d_c::infer(shape_list, t);
    }

    void forward(const forward_ctx_t &ctx) const
    {
        call<pool2d_c::forward>(ctx, t);
    }

    void backward(const backward_ctx_t &ctx) const
    {
        call<pool2d_c::backward>(ctx, t);
    }
};
