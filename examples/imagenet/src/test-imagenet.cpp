#include <cstdio>
#include <experimental/filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <crystalnet-ext.h>

#include "alexnet.h"
#include "imagenet.h"

namespace fs = std::experimental::filesystem;

struct tester {
    const shape_t *input_shape;
    const classifier_t *classifier;

    tester()
        : input_shape(new_shape(3, image_height, image_width, 3)),
          classifier(new_classifier(alexnet, input_shape, class_number))
    {
    }

    ~tester()
    {
        del_shape(input_shape);
        del_classifier(classifier);
    }

    uint32_t operator()(const tensor_ref_t *image) const
    {
        return most_likely(classifier, image);
    }
};

const std::string home(std::getenv("HOME"));
const std::string data_dir = home + "/var/data/imagenet/ILSVRC/val";

int main()
{
    const shape_t *input_shape = new_shape(3, image_height, image_width, 3);
    const tensor_t *_input_image = new_tensor(input_shape, dtypes.f32);
    const tensor_ref_t *input_image = new_tensor_ref(_input_image);

    tester t;
    for (const auto &f : fs::directory_iterator(fs::path(data_dir))) {
        std::cout << f << std::endl;
        const auto image = cv::imread(f.path().c_str());
        const auto input = square_normalize(image, 227);
        const auto dim = to_hwc(input, tensor_data_ptr(input_image));
        assert(dim == shape_dim(tensor_shape(input_image)));
        t(input_image);
    }

    del_tensor_ref(input_image);
    del_tensor(_input_image);
    del_shape(input_shape);
    return 0;
}
