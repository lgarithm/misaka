#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dtypes_t dtypes_t;

struct dtypes_t {
    const uint8_t i8;
    const uint8_t u8;
    const uint8_t i16;
    const uint8_t i32;
    const uint8_t f32;
    const uint8_t f64;
};

extern const dtypes_t dtypes;

extern const char *version();

typedef struct shape_t shape_t;
typedef struct shape_list_t shape_list_t;
typedef struct shape_ctx_t shape_ctx_t;
typedef struct tensor_t tensor_t;

typedef struct node_t node_t;
typedef struct model_t model_t;
typedef struct model_ctx_t model_ctx_t;

// TODO: make it possible to add user defined operators
typedef struct forward_ctx_t forward_ctx_t;
typedef struct backward_ctx_t backward_ctx_t;
typedef struct operator_t operator_t;
typedef struct shape_func_t shape_func_t;
typedef struct forward_func_t forward_func_t;
typedef struct backward_func_t backward_func_t;

extern shape_t *new_shape(int, ...);
extern void del_shape(shape_t *);
extern uint32_t shape_dim(const shape_t *);
extern uint32_t shape_rank(const shape_t *);
extern shape_ctx_t *new_shape_ctx();
extern void del_shape_ctx(shape_ctx_t *);
extern const shape_t *mk_shape(shape_ctx_t *, int, ...);
extern const shape_list_t *mk_shape_list(shape_ctx_t *,
                                         const shape_t *const p_shapes[]);

extern tensor_t *new_tensor(shape_t *, uint8_t);
extern void del_tensor(tensor_t *);
extern const shape_t *tensor_shape(tensor_t *);

extern model_ctx_t *new_model_ctx();
// extern void del_model_ctx(model_ctx_t *);  // managed by GC

extern model_t *new_model(model_ctx_t *, node_t *, node_t *);
extern void del_model(model_t *);

extern node_t *make_placeholder(model_ctx_t *, const shape_t *);
extern node_t *make_parameter(model_ctx_t *, const shape_t *);

typedef node_t *pnode_list_t[];
extern node_t *make_operator(model_ctx_t *, operator_t *, pnode_list_t);
extern node_t *wrap_node(model_ctx_t *, shape_t *, node_t *);

// operators
extern operator_t *register_op(const char *const, uint8_t, shape_func_t *,
                               forward_func_t *, backward_func_t *);
extern operator_t *op_add;
extern operator_t *op_mul;
extern operator_t *op_relu;
extern operator_t *op_softmax;
extern operator_t *op_xentropy;
// unstable operators:
extern operator_t *op_conv_nhwc;
extern operator_t *op_pool2d_c_max;
extern operator_t *make_op_pool2d(uint32_t, uint32_t, uint32_t, uint32_t);
extern operator_t *make_op_conv2d(uint32_t, uint32_t, uint32_t, uint32_t);

// symbolic APIs
typedef struct s_node_t s_node_t;
typedef s_node_t *symbol;
typedef struct s_model_t s_model_t;
typedef struct s_model_ctx_t s_model_ctx_t;
extern s_model_ctx_t *new_s_model_ctx();
extern s_model_t *new_s_model(s_model_ctx_t *, s_node_t *, s_node_t *);
extern void del_s_model(s_model_t *);
extern s_node_t *var(s_model_ctx_t *, const shape_t *);
extern s_node_t *covar(s_model_ctx_t *, const shape_t *);
extern s_node_t *reshape(s_model_ctx_t *, const shape_t *, const s_node_t *);
extern s_node_t *apply(s_model_ctx_t *, const operator_t *, s_node_t *args[]);

// layer APIs
typedef struct s_layer_t s_layer_t;
typedef s_layer_t *(layer_func_t)(const shape_list_t *);
extern layer_func_t *const new_layer_dense;
extern layer_func_t *const new_layer_relu;
extern layer_func_t *const new_layer_softmax;
extern layer_func_t *const new_layer_conv_nhwc; // TODO: deprecate
extern layer_func_t *const new_layer_pool_max;  // TODO: deprecate
extern s_node_t *transform(s_model_ctx_t *, const s_layer_t *, s_node_t *);
extern void del_s_layer(s_layer_t *);

// training
typedef struct dataset_t dataset_t;
typedef struct trainer_t trainer_t;
typedef struct s_trainer_t s_trainer_t;

typedef struct optimizer_t optimizer_t;
extern optimizer_t *opt_sgd;
extern optimizer_t *opt_adam;

// dataset_t *new_dataset();
extern void del_dataset(dataset_t *);
extern const shape_t *ds_image_shape(dataset_t *);
extern const shape_t *ds_label_shape(dataset_t *);

extern trainer_t *new_trainer(model_t *, operator_t *, optimizer_t *);
extern void del_trainer(trainer_t *);
extern void run_trainer(trainer_t *, dataset_t *);
extern void test_trainer(trainer_t *, dataset_t *);

extern s_trainer_t *new_s_trainer(s_model_t *, operator_t *, optimizer_t *);
extern void del_s_trainer(s_trainer_t *);
extern void s_trainer_run(s_trainer_t *, dataset_t *);
extern void s_rtainer_test(s_trainer_t *, dataset_t *);

// dataset
extern dataset_t *load_mnist(const char *const); // train | t10k
extern dataset_t *load_cifar();

// unstable APIs
extern tensor_t *_load_idx_file(const char *filename);
extern void experiment(trainer_t *, dataset_t *, dataset_t *);
extern void s_experiment(s_trainer_t *, dataset_t *, dataset_t *, uint32_t);

// eager APIs
extern shape_t *infer(const operator_t *, const shape_list_t *);
extern void eval(const operator_t *, const forward_ctx_t *);
extern void grad(const operator_t *, const backward_ctx_t *);

#ifdef __cplusplus
}
#endif
