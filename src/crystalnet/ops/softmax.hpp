#pragma once
#include <cmath>

#include <crystalnet.h>
#include <crystalnet/core/debug.hpp>
#include <crystalnet/model/operator.hpp>
#include <teavana/range.hpp>
using tea::range;

template <typename T>
void softmax_eval_safe(const vector_ref_t<T> &input,
                       const vector_ref_t<T> &output)
{
    const auto n = input.n;
    for (auto i : range(n)) {
        T tot = 0;
        for (auto j : range(n)) {
            tot += exp(input.data[j] - input.data[i]);
        }
        output.data[i] = std::max((T)1e-6, (T)1.0 / tot);
    }
}

template <typename T>
void softmax_grad(const vector_ref_t<T> &output, const matrix_ref_t<T> &grad)
{
    const auto n = output.n;
    for (auto i : range(n)) {
        grad.data[i * n + i] = output.data[i] * (1 - output.data[i]);
    }
    for (auto i : range(n)) {
        for (auto j : range(i)) {
            grad.data[i * n + j] = grad.data[j * n + i] =
                -output.data[i] * output.data[j];
        }
    }
}

struct softmax {
    constexpr static uint8_t arity = 1;

    static shape_t *infer(const shape_list_t *shape_list)
    {
        assert(shape_list->shapes.size() == arity);
        return new shape_t((*shape_list)[0]);
    }

    using T = float;

    struct forward : forward_ctx_t {
        void operator()() const
        {
            assert(inputs.arity() == arity);
            softmax_eval_safe(as_vector_ref<T>(inputs[0]),
                              as_vector_ref<T>(output));
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            auto n = output.shape.dim();
            tensor_t tmp(shape_t(n, n), idx_type<T>::type);
            softmax_grad(as_vector_ref<T>(output), as_matrix_ref<T>(ref(tmp)));
            linag<T>::vm(as_vector_ref<T>(output_gradient),
                         as_matrix_ref<T>(ref(tmp)),
                         as_vector_ref<T>(input_gradients[0]));
        }
    };
};

operator_t *op_softmax = _register_bi_op<softmax>("softmax");