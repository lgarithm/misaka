#include <crystalnet-ext.h>
#include <crystalnet/data/dataset.hpp>
#include <crystalnet/symbol/model.hpp>

void s_model_info(const s_model_t *m)
{
    uint32_t tot = 0;
    for (const auto &n : m->ctx->gc.allocs) {
        tot += n->shape.dim();
        printf("[d] %-12u %s\n", n->shape.dim(),
               std::to_string(n->shape).c_str());
    }
    printf("[s] total dim: %u\n", tot);
}

dataset_t *new_fake_dataset(const shape_t *p_shape, uint32_t arity)
{
    const uint32_t batch_size = 4;
    const uint32_t n_batches = 4;

    const uint32_t n = n_batches * batch_size;

    const auto images = new tensor_t(p_shape->batch(n), idx_type<float>::type);
    const auto labels = new tensor_t(shape_t(n, arity), idx_type<float>::type);

    return new simple_dataset_t(images, labels);
}
