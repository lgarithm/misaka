#include <crystalnet.h>
#include <crystalnet/core/gc.hpp>
#include <crystalnet/core/shape.hpp>
#include <crystalnet/ops/pool_generic.hpp>

operator_t *make_op_pool2d(uint32_t r, uint32_t s, //
                           uint32_t stride_r, uint32_t stride_s)
{
    static GC<op_pool2d_impl_t> gc;
    const auto op = gc(
        new op_pool2d_impl_t(pool2d_c::trait_t(r_shape(r, s), //
                                               r_shape(stride_r, stride_s))));
    return register_op("pool", 2,
                       new generic_shape_func_t<op_pool2d_impl_t>(*op),
                       new generic_forward_func_t<op_pool2d_impl_t>(*op),
                       new generic_backward_func_t<op_pool2d_impl_t>(*op));
}
