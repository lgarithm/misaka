#pragma once
#include <crystalnet.h>
#include <crystalnet/layers/layer.hpp>

struct dense : s_layer_t {
    const uint32_t n;
    explicit dense(uint32_t n) : n(n) {}

    s_node_t *operator()(s_model_ctx_t &ctx, s_node_t *x) const override
    {
        const uint32_t n0 = x->shape.dim();
        if (x->shape.rank() != 1) {
            x = ctx.wrap_node(shape_t(n0), x);
        }
        auto w = ctx.make_parameter(shape_t(n0, n));
        auto y = ctx.make_operator(*op_mul, x, w);
        auto b = ctx.make_parameter(shape_t(n));
        return ctx.make_operator(*op_add, y, b);
    }

    static s_layer_t *create(const shape_list_t *shape_list)
    {
        // TODO: extract n fron shape_list
        const auto n = 10;
        return new dense(10);
    }
};

const auto new_layer_dense = dense::create;
