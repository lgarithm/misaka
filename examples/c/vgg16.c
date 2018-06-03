#include <stdio.h>

#include <crystalnet-ext.h>

typedef shape_t const *p_shape_t;

// https://www.cs.toronto.edu/~frossard/post/vgg16/
s_model_t *vgg16(const shape_t *image_shape, uint32_t arity)
{
    context_t *ctx = new_context();
    trait_ctx_t *tc = new_trait_ctx();

    s_layer_t *c1 = new_layer_conv2d(                //
        mk_filter(tc, mk_shape(ctx, 3, 3, 3, 64)),   //
        mk_padding(tc, mk_shape(ctx, 2, 1, 1)),      //
        NULL);                                       //
    s_layer_t *c2 = new_layer_conv2d(                //
        mk_filter(tc, mk_shape(ctx, 3, 3, 3, 128)),  //
        mk_padding(tc, mk_shape(ctx, 2, 1, 1)),      //
        NULL);                                       //
    s_layer_t *c3 = new_layer_conv2d(                //
        mk_filter(tc, mk_shape(ctx, 3, 3, 3, 256)),  //
        mk_padding(tc, mk_shape(ctx, 2, 1, 1)),      //
        NULL);                                       //
    s_layer_t *c4_5 = new_layer_conv2d(              //
        mk_filter(tc, mk_shape(ctx, 3, 3, 3, 512)),  //
        mk_padding(tc, mk_shape(ctx, 2, 1, 1)),      //
        NULL);                                       //
    s_layer_t *f4096 = new_layer_dense(4096);        //
    s_layer_t *f_out = new_layer_dense(arity);       //
    s_layer_t *pool = new_layer_pool2d(              //
        mk_filter(tc, mk_shape(ctx, 2, 2, 2)),       //
        mk_stride(tc, mk_shape(ctx, 2, 2, 2)));      //
    s_layer_t *relu = new_layer_relu();              //
    s_layer_t *out = new_layer_softmax();

    printf("[x] creating model\n");
    symbol x = var(ctx, image_shape);
    symbol y = transform_all(  //
        ctx,                   //
        (p_layer_t[]){
            c1,    relu, c1,    relu, pool,              //
            c2,    relu, c2,    relu, pool,              //
            c3,    relu, c3,    relu, c3,   relu, pool,  //
            c4_5,  relu, c4_5,  relu, c4_5, relu, pool,  //
            c4_5,  relu, c4_5,  relu, c4_5, relu, pool,  //
            f4096, relu, f4096, relu,                    //
            f_out, out,                                  //
            NULL,                                        //
        },
        x);
    del_s_layer(c1);
    del_s_layer(c2);
    del_s_layer(c3);
    del_s_layer(c4_5);
    del_s_layer(f4096);
    del_s_layer(f_out);
    del_s_layer(pool);
    del_s_layer(relu);
    del_s_layer(out);
    printf("[y] creating model\n");
    return new_s_model(ctx, x, y);
}

const uint32_t height = 224;
const uint32_t width = 224;
const uint32_t class_number = 1000;

dataset_t *fake_imagenet()
{
    const shape_t *image_shape = new_shape(3, height, width, 3);
    dataset_t *p_ds = new_fake_dataset(image_shape, class_number);
    del_shape(image_shape);
    return p_ds;
}

int main()
{
    const shape_t *image_shape = new_shape(3, height, width, 3);
    s_model_t *model = vgg16(image_shape, class_number);
    s_model_info(model);
    del_s_model(model);
    del_shape(image_shape);
    return 0;
}
