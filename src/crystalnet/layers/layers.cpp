#include <crystalnet.h>
#include <crystalnet/layers/conv_nhwc.hpp>
#include <crystalnet/layers/dense.hpp>
#include <crystalnet/layers/pool.hpp>
#include <crystalnet/layers/relu.hpp>
#include <crystalnet/layers/softmax.hpp>

const auto new_layer_dense = dense::create;
const auto new_layer_conv_nhwc = conv_nhwc::create;
const auto new_layer_pool_max = pool::create;
const auto new_layer_relu = relu_layer::create;
const auto new_layer_softmax = softmax_layer::create;

s_node_t *transform(s_model_ctx_t *ctx, const s_layer_t *l, s_node_t *x)
{
    return (*l)(*ctx, x);
}