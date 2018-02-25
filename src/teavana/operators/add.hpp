#pragma once

#include <cassert> // for assert
#include <utility> // for integer_sequence

#include "teavana/core/shape.hpp"  // for operator==, dim, shape_t
#include "teavana/core/tensor.hpp" // for assign
#include "teavana/operator.hpp"    // for gin, in, select_ctx
#include "teavana/range.hpp"       // for range

namespace tea
{
struct op_add {
    static constexpr const char *name = "add";
    using signature = ::std::index_sequence<1, 1, 1>;
    using ctx_types = select_ctx<signature>;

    static constexpr auto infer_shape(const shape_t<1> &s1,
                                      const shape_t<1> &s2)
    {
        assert(s1 == s2 && __FILE__);
        return s1;
    }

    template <typename R> struct default_impl {
        using eval_ctx_t = typename ctx_types::template eval_ctx_type<R>;
        using grad_ctx_t = typename ctx_types::template grad_ctx_type<R>;
        using temp_ctx_t = typename ctx_types::template temp_ctx_type<R>;

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &)
        {
            const auto x = in<0>(ctx);
            const auto y = in<1>(ctx);
            for (auto i : range(dim(ctx.output.shape))) {
                ctx.output.data[i] = x.data[i] + y.data[i];
            }
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &)
        {
            assign(gin<0>(ctx), ctx.g_output);
        }

        static void grad_1(const grad_ctx_t &ctx, const temp_ctx_t &)
        {
            assign(gin<1>(ctx), ctx.g_output);
        }
    };
};
}
