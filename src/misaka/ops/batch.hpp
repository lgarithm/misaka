#pragma once
#include <misaka/core/tensor.hpp>
#include <teavana/range.hpp>
using tea::range;

forward_ctx_t unbatch(uint8_t pos, uint8_t idx, const forward_ctx_t &ctx)
{
    std::vector<tensor_ref_t> inputs;
    for (auto i : range(ctx.inputs._args.size())) {
        if (i == pos) {
            inputs.push_back(ctx.inputs[i][idx]);
        } else {
            inputs.push_back(ctx.inputs[i]);
        }
    }
    return forward_ctx_t(tensor_ref_list_t(inputs), ctx.output[idx]);
}

backward_ctx_t unbatch(uint8_t pos, uint8_t idx, const backward_ctx_t &ctx)
{
    std::vector<tensor_ref_t> inputs;
    std::vector<tensor_ref_t> input_gradients;
    for (auto i : range(ctx.inputs._args.size())) {
        if (i == pos) {
            inputs.push_back(ctx.inputs[i][idx]);
            input_gradients.push_back(ctx.input_gradients[i][idx]);
        } else {
            inputs.push_back(ctx.inputs[i]);
            input_gradients.push_back(ctx.input_gradients[i]);
        }
    }
    return backward_ctx_t(tensor_ref_list_t(inputs), ctx.output[idx],
                          tensor_ref_list_t(input_gradients),
                          ctx.output_gradient[idx]);
}

template <typename O, uint8_t pos> struct batch {
    constexpr static uint8_t arity = O::arity;

    static shape_t n_batch(uint32_t n, const shape_t &shape)
    {
        std::vector<uint32_t> dims({n});
        dims.insert(dims.end(), shape.dims.begin(), shape.dims.end());
        return shape_t(dims);
    }

    static shape_t *infer(const shape_list_t *shapes)
    {
        static_assert(pos < arity);
        assert(shapes->size() == arity);
        const auto batched_shape = (*shapes)[pos];
        assert(batched_shape.rank() > 1);
        const auto batch_size = batched_shape.len();
        const auto original_shape = batched_shape.sub();
        shape_list_t new_shape_list;
        for (auto i = 0; i < arity; ++i) {
            if (i == pos) {
                new_shape_list.shapes.push_back(original_shape);
            } else {
                new_shape_list.shapes.push_back((*shapes)[i]);
            }
        }
        auto out_shape_ = std::unique_ptr<shape_t>(O::infer(&new_shape_list));
        auto out_shape = n_batch(batch_size, *out_shape_);
        return new shape_t(out_shape);
    }

    using T = float; // TODO: cast based on dtype

    struct forward : forward_ctx_t {
        void operator()() const
        {
            for (auto i : range(inputs[pos].shape.len())) {
                const auto ctx = unbatch(pos, i, *this);
                (*(typename O::forward *)&ctx)();
            }
        }
    };

    struct backward : backward_ctx_t {
        void operator()() const
        {
            for (auto i : range(inputs[pos].shape.len())) {
                const auto ctx = unbatch(pos, i, *this);
                (*(typename O::backward *)&ctx)();
            }
        }
    };
};
