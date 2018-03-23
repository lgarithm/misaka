#include <stdio.h>

#include <crystalnet-ext.h>

#include "alexnet.h"

dataset_t *fake_imagenet()
{
    const shape_t *image_shape = new_shape(3, image_height, image_width, 3);
    dataset_t *p_ds = new_fake_dataset(image_shape, class_number);
    del_shape(image_shape);
    return p_ds;
}

int main()
{
    const shape_t *image_shape = new_shape(3, image_height, image_width, 3);
    s_model_t *model = alexnet(image_shape, class_number);
    s_model_info(model);
    s_trainer_t *trainer = new_s_trainer(model, op_xentropy, opt_adam);
    dataset_t *ds1 = fake_imagenet();
    // dataset_t *ds2 = fake_imagenet();
    const uint32_t batch_size = 2;
    s_experiment(trainer, ds1, NULL, batch_size);
    del_dataset(ds1);
    // del_dataset(ds2);
    del_s_trainer(trainer);
    del_s_model(model);
    del_shape(image_shape);
    return 0;
}
