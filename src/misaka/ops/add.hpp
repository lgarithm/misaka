#pragma once
#include <misaka.h>
#include <misaka/core/debug.hpp>
#include <misaka/core/shape.hpp>
#include <misaka/linag/linag.hpp>
#include <misaka/model/operator.hpp>

struct add {
    constexpr static uint8_t arity = 2;

    static shape_t *infer(const shape_list_t *shape_list)
    {
        assert(shape_list->shapes.size() == arity);
        return new shape_t((*shape_list)[0]);
    }

    using T = float; // TODO: cast based on dtype

    struct forward : forward_ctx_t {
        void operator()() const
        {
            assert(inputs.arity() == arity);
            linag<T>::vv(as_vector_ref<T>(inputs[0]),
                         as_vector_ref<T>(inputs[1]), as_vector_ref<T>(output));
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            auto g = r_tensor_ref_t<T>(output_gradient);
            r_tensor_ref_t<T>(input_gradients[0]).copy(g);
            r_tensor_ref_t<T>(input_gradients[1]).copy(g);
        }
    };
};

operator_t *op_add = _register_bi_op<add>("add");
