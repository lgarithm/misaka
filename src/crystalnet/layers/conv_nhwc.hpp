#pragma once
#include <crystalnet.h>
#include <crystalnet/layers/layer.hpp>

struct conv_nhwc : s_layer_t {
    const uint32_t r;
    const uint32_t s;
    const uint32_t d;
    conv_nhwc(uint32_t r, uint32_t s, uint32_t d = 1) : r(r), s(s), d(d) {}

    s_node_t *operator()(s_model_ctx_t &ctx, s_node_t *x) const override
    {
        const auto input_rank = x->shape.rank();
        auto c = [&]() {
            if (input_rank == 3) {
                const auto[h, w, c] = cast<3>(x->shape.dims);
                x = ctx.wrap_node(shape_t(1, h, w, c), x);
            }
            assert(x->shape.rank() == 4);
            const auto[n, h, w, c] = cast<4>(x->shape.dims);
            return c;
        }();
        auto weight = ctx.make_parameter(shape_t(r, s, c, d));
        auto y = ctx.make_operator(*op_conv_nhwc, x, weight);
        const auto[n, u, v, d] = cast<4>(y->shape.dims);
        y = ctx.wrap_node(shape_t(n * u * v, d), y);
        auto bias = ctx.make_parameter(shape_t(d));
        y = ctx.make_operator(*op_add, y, bias);
        if (input_rank == 3) {
            y = ctx.wrap_node(shape_t(u, v, d), y);
        } else {
            y = ctx.wrap_node(shape_t(n, u, v, d), y);
        }
        return y;
    }
};