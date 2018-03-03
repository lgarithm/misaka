#pragma once
#include <misaka.h>
#include <misaka/model/operator.hpp>
#include <teavana/range.hpp>

using tea::range;

template <typename T> vector_ref_t<T> cast_to_v(const tensor_ref_t &tensor)
{
    r_tensor_ref_t<T> r(tensor);
    return vector_ref_t<T>(r.shape.dim(), r.data);
}

struct relu {
    constexpr static uint8_t arity = 1;

    static shape_t *infer(const shape_list_t *shape_list)
    {
        assert(shape_list->shapes.size() == arity);
        return new shape_t((*shape_list)[0]);
    }

    using T = float; // TODO: cast based on dtype

    template <typename T> inline static T _relu(T x) { return x > 0 ? x : 0; }

    struct forward : forward_ctx_t {
        void operator()() const
        {
            DEBUG(__FILE__);
            auto x = cast_to_v<T>(inputs[0]);
            auto y = cast_to_v<T>(output);
            auto n = equally(x.n, y.n);
            for (auto i : range(n)) {
                y.data[i] = _relu(x.data[i]);
            }
        }
    };

    template <typename T> inline static T _relu_grad(T x)
    {
        return x > 0 ? 1.0 : 0.0;
    }

    struct backward : backward_ctx_t {
        void operator()() const
        {
            auto x = cast_to_v<T>(inputs[0]);
            auto gx = cast_to_v<T>(input_gradients[0]);
            auto gy = cast_to_v<T>(output_gradient);
            auto n = x.n; // == gx.n == gy.n
            for (auto i : range(n)) {
                gx.data[i] = gy.data[i] * _relu_grad(x.data[i]);
            }
        }
    };
};

operator_t *op_relu = _register_bi_op<relu>("relu");
