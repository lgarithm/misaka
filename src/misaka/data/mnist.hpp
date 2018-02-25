#pragma once
#include <cstdio>
#include <cstdlib>

#include <misaka.h>
#include <misaka/core/tensor.hpp>
#include <misaka/data/dataset.hpp>

tensor_t *_load_mnist(const std::string &name)
{
    const auto filename =
        std::string(getenv("HOME")) + "/var/data/mnist/" + name;
    return _load_idx_file(filename.c_str());
}

template <typename T> tensor_t *make_onehot(const tensor_t &tensor, uint32_t k)
{
    auto dims = std::vector<uint32_t>(tensor.shape.dims);
    dims.push_back(k);
    auto distro_ = new tensor_t(shape_t(dims), idx_type<T>::type);
    r_tensor_ref_t<T> distro(*distro_);
    r_tensor_ref_t<uint8_t> r(tensor); // TODO: support other uint types
    auto n = tensor.shape.dim();
    for (auto i = 0; i < n; ++i) {
        auto off = r.data[i];
        if (0 <= off && off < k) {
            distro.data[i * k + off] = 1;
        } else {
            // TODO: print a warning msg
            assert(false);
        }
    }
    return distro_;
}

dataset_t *load_mnist_data(const std::string &name)
{
    DEBUG(__func__);
    tensor_t *images = _load_mnist(name + "-images-idx3-ubyte");
    tensor_t *images_ = cast_to<float>(r_tensor_ref_t<uint8_t>(*images));
    delete images;
    {
        r_tensor_ref_t<float> r(*images_);
        auto n = r.shape.dim();
        for (auto i = 0; i < n; ++i) {
            r.data[i] /= 255.0;
        }
    }
    tensor_t *labels = _load_mnist(name + "-labels-idx1-ubyte");
    tensor_t *labels_ = make_onehot<float>(*labels, 10);
    delete labels;
    return new simple_dataset_t(images_, labels_);
}

dataset_t *load_mnist(const char *const name) { return load_mnist_data(name); }
