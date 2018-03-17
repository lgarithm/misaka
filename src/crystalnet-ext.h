#pragma once

#include <crystalnet.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct trait_ctx_t trait_ctx_t;
typedef struct filter_trait_t filter_trait_t;
typedef struct padding_trait_t padding_trait_t;
typedef struct stride_trait_t stride_trait_t;

extern trait_ctx_t *new_trait_ctx();
extern void free_trait_ctx(trait_ctx_t *);
extern filter_trait_t *mk_filter(trait_ctx_t *, const shape_t *);
extern padding_trait_t *mk_padding(trait_ctx_t *, const shape_t *);
extern stride_trait_t *mk_stride(trait_ctx_t *, const shape_t *);

extern s_layer_t *const new_layer_pool2d(const filter_trait_t *,
                                         const stride_trait_t *);
extern s_layer_t *const new_layer_conv2d(const filter_trait_t *,
                                         const padding_trait_t *,
                                         const stride_trait_t *);

typedef s_layer_t const *p_layer_t;
extern s_node_t *transform_all(s_model_ctx_t *, p_layer_t layers[], s_node_t *);
// debug APIs
extern void s_model_info(const s_model_t *);
extern dataset_t *new_fake_dataset(const shape_t *, uint32_t);

#ifdef __cplusplus
}
#endif
