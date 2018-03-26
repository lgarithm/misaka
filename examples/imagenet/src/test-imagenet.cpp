#include <cstdio>
#include <experimental/filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <crystalnet-ext.h>

#include "alexnet.h"
#include "imagenet.h"
#include "vgg16.h"

namespace fs = std::experimental::filesystem;

const std::string home(std::getenv("HOME"));
const std::string data_dir = home + "/var/data/imagenet/ILSVRC/val";
const std::string model_dir = home + "/var/models/vgg16";
const std::string
    test_image_url("https://www.cs.toronto.edu/~frossard/vgg16/laska.png");
const std::string test_image_path = home + "/var/models/vgg16/laska.png";

bool endwith(const std::string &str, const std::string &suffix)
{
    if (str.size() < suffix.size()) {
        return false;
    }
    return std::string(str.end() - suffix.size(), str.end()) == suffix;
}

std::string basename(const fs::path &p)
{
    std::string name;
    for (const auto &it : p) {
        name = it;
    }
    return name;
}

void load_weight(classifier_t *classifier)
{
    std::vector<std::string> names({
        "conv1_1_W", "conv1_1_b", //
        "conv1_2_W", "conv1_2_b", //
        "conv2_1_W", "conv2_1_b", //
        "conv2_2_W", "conv2_2_b", //
        "conv3_1_W", "conv3_1_b", //
        "conv3_2_W", "conv3_2_b", //
        "conv3_3_W", "conv3_3_b", //
        "conv4_1_W", "conv4_1_b", //
        "conv4_2_W", "conv4_2_b", //
        "conv4_3_W", "conv4_3_b", //
        "conv5_1_W", "conv5_1_b", //
        "conv5_2_W", "conv5_2_b", //
        "conv5_3_W", "conv5_3_b", //
        "fc6_W",     "fc6_b",     //
        "fc7_W",     "fc7_b",     //
        "fc8_W",     "fc8_b",     //
    });

    std::map<std::string, const tensor_t *> weights;
    for (const auto &f : fs::directory_iterator(fs::path(model_dir))) {
        if (!endwith(f.path().c_str(), ".idx")) {
            continue;
        }
        std::cout << f << std::endl;
        _idx_file_info(f.path().c_str());
        weights[basename(f.path())] = _load_idx_file(f.path().c_str());
    }

    uint32_t i = 0;
    for (const auto &name : names) {
        const tensor_t *w = weights[name + ".idx"];
        if (w == nullptr) {
            fprintf(stderr, "[e] %s.idx not exist.", name.c_str());
            exit(1);
        }
        const std::string name1 = "param" + std::to_string(i++);
        classifier_load(classifier, name1.c_str(), tensor_ref(w));
        printf("[i] %s <- %s\n", name1.c_str(), name.c_str());
    }
    for (const auto[k, v] : weights) {
        del_tensor(v);
    }
}

void preprocess(const fs::path &p, const tensor_ref_t *input_image)
{
    const auto image = cv::imread(p.c_str());
    const auto input = square_normalize(image, vgg16_image_size);
    // cv::imwrite("input.bmp", input);
    const tensor_t *_tmp = new_tensor(tensor_shape(input_image), dtypes.u8);
    const auto dim = to_hwc(input, tensor_ref(_tmp));
    assert(dim == shape_dim(tensor_shape(input_image)));
    using T = float;
    T *data = (T *)tensor_data_ptr(input_image);
    uint8_t *tmp = (uint8_t *)tensor_data_ptr(tensor_ref(_tmp));
    for (auto i = 0; i < dim; ++i) {
        data[i] = tmp[i] / 255.0;
    }
    del_tensor(_tmp);
}

void run()
{
    const shape_t *input_shape =
        new_shape(3, vgg16_image_size, vgg16_image_size, 3);
    const tensor_t *input_image = new_tensor(input_shape, dtypes.f32);
    classifier_t *classifier =
        new_classifier(vgg16, input_shape, vgg16_class_number);
    load_weight(classifier);

    preprocess(fs::path(test_image_path), tensor_ref(input_image));
    const int k = 5;
    int results[k];
    top_likely(classifier, tensor_ref(input_image), k, results);
    for (int i = 0; i < k; ++i) {
        std::cout << results[i] << std::endl;
    }

    /*
    for (const auto &f : fs::directory_iterator(fs::path(data_dir))) {
        std::cout << f << std::endl;
        preprocess(f.path(), input_image);
        const uint32_t result = most_likely(classifier, input_image);
        std::cout << "result: " << result
                  << std::endl; // TODO: output class name
    }
    */

    del_classifier(classifier);
    del_tensor(input_image);
    del_shape(input_shape);
}

int main()
{
    run();
    return 0;
}
