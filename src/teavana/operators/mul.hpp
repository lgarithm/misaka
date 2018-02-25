#pragma once

#include <cassert> // for assert
#include <utility> // for integer_sequence

#include "teavana/core/shape.hpp" // for operator==, sub, shape_t
#include "teavana/matrix.hpp" // for mat_x_mat, as_col_matrix, as_row_matrix, etc
#include "teavana/operator.hpp" // for in, gin, select_ctx

namespace tea
{
struct op_mul {
    static constexpr const char *name = "mul_mv";
    using signature = ::std::index_sequence<1, 2, 1>;
    using ctx_types = select_ctx<signature>;

    static constexpr auto infer_shape(const shape_t<2> &s1,
                                      const shape_t<1> &s2)
    {
        assert(sub(s1) == s2 && __FILE__);
        return shape(len(s1));
    }

    template <typename R> struct default_impl {
        using eval_ctx_t = typename ctx_types::template eval_ctx_type<R>;
        using grad_ctx_t = typename ctx_types::template grad_ctx_type<R>;
        using temp_ctx_t = typename ctx_types::template temp_ctx_type<R>;

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &)
        {
            mat_x_vec(in<0>(ctx), in<1>(ctx), ctx.output);
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &)
        {
            mat_x_mat(as_col_matrix(ctx.g_output), as_row_matrix(in<1>(ctx)),
                      gin<0>(ctx));
        }

        static void grad_1(const grad_ctx_t &ctx, const temp_ctx_t &)
        {
            vec_x_mat(ctx.g_output, in<0>(ctx), gin<1>(ctx));
        }
    };
};

struct op_mul_vm {
    static constexpr const char *name = "mul_vm";
    using signature = ::std::index_sequence<1, 1, 2>;
    using ctx_types = select_ctx<signature>;

    static constexpr auto infer_shape(const shape_t<1> &s1,
                                      const shape_t<2> &s2)
    {
        assert(len(s1) == len(s2) && __FILE__);
        return shape(wid(s2));
    }

    template <typename R> struct default_impl {
        using eval_ctx_t = typename ctx_types::template eval_ctx_type<R>;
        using grad_ctx_t = typename ctx_types::template grad_ctx_type<R>;
        using temp_ctx_t = typename ctx_types::template temp_ctx_type<R>;

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &)
        {
            vec_x_mat(in<0>(ctx), in<1>(ctx), ctx.output);
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &)
        {
            mat_x_vec(in<1>(ctx), ctx.g_output, gin<0>(ctx));
        }

        static void grad_1(const grad_ctx_t &ctx, const temp_ctx_t &)
        {
            mat_x_mat(as_col_matrix(in<0>(ctx)), as_row_matrix(ctx.g_output),
                      gin<1>(ctx));
        }
    };
};

struct op_mul_mm {
    static constexpr const char *name = "mul_mm";
    using signature = ::std::index_sequence<2, 2, 2>;
    using ctx_types = select_ctx<signature>;

    static constexpr auto infer_shape(const shape_t<2> &s1,
                                      const shape_t<2> &s2)
    {
        assert(wid(s1) == len(s2) && __FILE__);
        return shape(len(s1), wid(s2));
    }

    template <typename R> struct default_impl {
        using eval_ctx_t = typename ctx_types::template eval_ctx_type<R>;
        using grad_ctx_t = typename ctx_types::template grad_ctx_type<R>;
        using temp_ctx_t = typename ctx_types::template temp_ctx_type<R>;

        static void eval(const eval_ctx_t &ctx, const temp_ctx_t &)
        {
            mat_x_mat(in<0>(ctx), in<1>(ctx), ctx.output);
        }

        static void grad_0(const grad_ctx_t &ctx, const temp_ctx_t &)
        {
            mat_x_mat_trans(ctx.g_output, in<1>(ctx), gin<0>(ctx));
        }

        static void grad_1(const grad_ctx_t &ctx, const temp_ctx_t &)
        {
            mat_trans_x_mat(in<0>(ctx), ctx.g_output, gin<1>(ctx));
        }
    };
};
}
