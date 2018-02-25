#pragma once

#include <cassert> // for assert
#include <cmath>   // for log
#include <utility> // for integer_sequence

#include "teavana/core/shape.hpp"  // for dim, operator==, shape, shape_t
#include "teavana/core/tensor.hpp" // for scalar
#include "teavana/operator.hpp"    // for in, gin, select_ctx
#include "teavana/range.hpp"       // for range

namespace tea
{
struct op_cross_entropy {
    static constexpr const char *name = "cross_entropy";
    using signature = ::std::index_sequence<0, 1, 1>;
    using ctx_types = select_ctx<signature>;

    static constexpr auto infer_shape(const shape_t<1> &s1,
                                      const shape_t<1> &s2)
    {
        assert(s1 == s2 && __FILE__);
        return shape();
    }

    template <typename R> struct default_impl {
        using eval_ctx_t = typename ctx_types::template eval_ctx_type<R>;
        using grad_ctx_t = typename ctx_types::template grad_ctx_type<R>;
        using temp_ctx_t = typename ctx_types::template temp_ctx_type<R>;

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &)
        {
            const auto x = in<0>(ctx);
            const auto y = in<1>(ctx);
            R z = 0;
            for (auto i : range(dim(x.shape))) {
                z += x.data[i] * log(y.data[i]);
            }
            scalar(ctx.output) = -z;
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &)
        {
            const auto y = in<1>(ctx);
            const auto gx = gin<0>(ctx);
            for (auto i : range(dim(y.shape))) {
                gx.data[i] = scalar(ctx.g_output) * -log(y.data[i]);
            }
        }

        static void grad_1(const grad_ctx_t &ctx, const temp_ctx_t &)
        {
            const auto x = in<0>(ctx);
            const auto y = in<1>(ctx);
            const auto gy = gin<1>(ctx);
            for (auto i : range(dim(y.shape))) {
                gy.data[i] = scalar(ctx.g_output) * (-x.data[i] / y.data[i]);
            }
        }
    };
};
}
