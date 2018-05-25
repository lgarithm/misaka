#include "reorg_layer.h"

#include <crystalnet-internal.h>
#include <crystalnet/core/cast.hpp>
#include <crystalnet/core/operator.hpp>  // TODO: don't include private headers
#include <crystalnet/layers/layer.hpp>
#include <crystalnet/utility/range.hpp>

namespace darknet
{

struct reorg_op {
    constexpr static uint8_t arity = 1;

    shape_t infer(const shape_list_t &shape_list) const
    {
        const auto [p] = cast<arity>(shape_list.shapes, auto_hint);
        if (p.rank() == 3) {
            return _infer(shape_list_t({p.batch(1)}));
        } else {
            return _infer(shape_list);
        }
    }

    shape_t _infer(const shape_list_t &shape_list) const
    {
        const auto [p] = cast<arity>(shape_list.shapes, auto_hint);
        const auto [b, c, h, w] = cast<4>(p.dims, auto_hint);
        check(h % 2 == 0);
        check(w % 2 == 0);
        return shape_t(b, c * 4, h / 2, w / 2);
    }

    using T = float;  // TODO: cast based on dtype

    // [N, C, 2H, 2W] -> [N, 4C, H, W]
    void forward(const forward_ctx_t &ctx) const
    {
        const auto [x] = cast<arity>(ctx.inputs._args, auto_hint);
        const auto [n, c, _2h, _2w] = cast<4>(x.shape.dims, auto_hint);
        const auto [_n, _4c, h, w] = cast<4>(ctx.output.shape.dims, auto_hint);

        for (const auto b : range(n)) {
            const auto xx = x[b];
            const auto yy = ctx.output[b];
            // [C, 2H, 2W] -> [4C, H, W]
            // C times ([1, 2H, 2W] -> [4, H, W])
            // TODO: verify
            for (const auto l : range(c)) {
                const auto xxx = ranked<2, T>(xx[l]);
                const auto yyy = ranked<3, T>(yy.slice(l * 4, (l + 1) * 4));
                for (const auto k : range(4)) {
                    const auto gx = k / 2;
                    const auto gy = k % 2;
                    for (const auto i : range(h)) {
                        for (const auto j : range(w)) {
                            yyy.at(k, i, j) = xxx.at(i + gx * h, j + gy * w);
                        }
                    }
                }
            }
        }
    }

    void backward(const backward_ctx_t &ctx) const
    {
        throw std::logic_error("NOT IMPLEMENTED");
    }
};

struct reorg_layer : s_layer_t {
    std::unique_ptr<reorg_op> _op;
    const operator_t *op;

    reorg_layer()
        : _op(new reorg_op),
          op(_register_generic_bi_op("darknet::reorg", _op.get()))
    {
    }

    s_node_t *operator()(s_model_ctx_t &ctx, s_node_t *x) const override
    {
        return ctx.make_operator(*op, x);
    }
};
}  // namespace darknet

s_layer_t *make_reorg_layer() { return new darknet::reorg_layer(); }
