#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

#include <crystalnet.h>
#include <crystalnet/core/tensor.hpp>

struct forward_ctx_t {
    tensor_ref_list_t inputs;
    tensor_ref_t output;

    forward_ctx_t(const tensor_ref_list_t &inputs, const tensor_ref_t &output)
        : inputs(inputs), output(output)
    {
    }
};

struct backward_ctx_t {
    tensor_ref_list_t inputs;
    tensor_ref_t output;
    tensor_ref_list_t input_gradients; // TODO: make items of it optional
    tensor_ref_t output_gradient;

    backward_ctx_t(const tensor_ref_list_t &inputs, const tensor_ref_t &output,
                   const tensor_ref_list_t &input_gradients,
                   const tensor_ref_t &output_gradient)
        : inputs(inputs), output(output), input_gradients(input_gradients),
          output_gradient(output_gradient)
    {
    }
};

struct shape_func_t {
    virtual shape_t operator()(const shape_list_t &) = 0;
    virtual ~shape_func_t() {}
};

struct forward_func_t {
    virtual void operator()(const forward_ctx_t &) = 0;
    virtual ~forward_func_t() {}
};

struct backward_func_t {
    virtual void operator()(const backward_ctx_t &) = 0;
    virtual ~backward_func_t() {}
};

struct operator_t {
    const uint8_t arity;
    operator_t(const char *const name, uint8_t arity, shape_func_t *infer,
               forward_func_t *eval, backward_func_t *feed)
        : name(name), arity(arity), infer(infer), forward(eval), backward(feed)
    {
    }

    const std::string name;

    std::unique_ptr<shape_func_t> infer;
    std::unique_ptr<forward_func_t> forward;
    std::unique_ptr<backward_func_t> backward;
};

struct initializer_t {
    virtual void operator()(const tensor_ref_t &) const = 0;
    virtual ~initializer_t() {}
};

struct simple_shape_func_t : shape_func_t {
    typedef shape_t *(shape_func_ptr_t)(const shape_list_t *);
    shape_func_ptr_t *fn_ptr;
    simple_shape_func_t(shape_func_ptr_t *fn_ptr) : fn_ptr(fn_ptr) {}
    shape_t operator()(const shape_list_t &shape_list) override
    {
        shape_t *p_shape = fn_ptr(&shape_list);
        shape_t shape(*p_shape);
        free_shape(p_shape);
        return shape;
    }
};

template <typename T, typename S> void call(S &ctx)
{
    static_assert(std::is_base_of<S, T>::value);
    (*(T *)&ctx)();
}

template <typename T> struct simple_forward_func_t : forward_func_t {
    void operator()(const forward_ctx_t &ctx) override
    {
        call<typename T::forward>(ctx);
    }
};

template <typename T> struct simple_backward_func_t : backward_func_t {
    void operator()(const backward_ctx_t &ctx) override
    {
        call<typename T::backward>(ctx);
    }
};

template <typename T> operator_t *_register_bi_op(const char *const name)
{
    shape_func_t *infer = new simple_shape_func_t(T::infer);
    forward_func_t *eval = new simple_forward_func_t<T>;
    backward_func_t *feed = new simple_backward_func_t<T>;
    return register_op(name, T::arity, infer, eval, feed);
}
