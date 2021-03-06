#include <crystalnet/core/gc.hpp>
#include <crystalnet/train/optimizer.hpp>

struct sgd_optimizer_t : optimizer_t {
    struct ctx : optimizer_ctx_t {
        using T = float;
        static constexpr T eta = 1e-3;  // TODO: make it configrable

        model_t *model;
        explicit ctx(model_t *model) : model(model) {}
        void operator()() override
        {
            for (auto p : model->ctx.params.items) {
                r_tensor_ref_t<T> x(p->value());
                r_tensor_ref_t<T> g(p->gradient());
                // TODO: sync parameter with other agents
                const auto n = x.shape.dim();
                // TODO: use axpy from blas
                for (auto i = 0; i < n; ++i) { x.data[i] -= g.data[i] * eta; }
            }
        }
    };

    optimizer_ctx_t *optimize(model_t *model) const override
    {
        return new ctx(model);
    }
};

const optimizer_t *opt_sgd = gc(new sgd_optimizer_t);
