#pragma once
#include <sstream>
#include <vector>

#include <crystalnet-ext.h>
#include <crystalnet/core/tensor.hpp>
#include <crystalnet/core/tracer.hpp>
#include <crystalnet/utility/range.hpp>

namespace darknet
{
struct bbox_t {
    using T = float;

    T cx;
    T cy;
    T h;
    T w;
};

struct detection_t {
    using T = float;

    bbox_t bbox;

    const int classes;
    tensor_t probs;

    // vector_t<T> mask; T[coords - 4], if coords > 4
    // float *mask;

    float objectness;
    int sort_class;

    int i, j;

    detection_t(int classes, int coords)
        : classes(classes), probs(shape_t(classes), idx_type<T>::type)
    {
    }
};
}  // namespace darknet

using detection_list_t = std::vector<std::unique_ptr<darknet::detection_t>>;
detection_list_t get_detections(const tensor_ref_t & /* results */,
                                const tensor_ref_t & /* anchor boxes */);

namespace std
{
inline string to_string(const darknet::bbox_t &bb)
{
    stringstream ss;
    ss << "bbox<"
       << "cx=" << bb.cx << ","
       << "cy=" << bb.cy << ","
       << "w=" << bb.w << ","
       << "h=" << bb.h << ">";
    return ss.str();
}
}  // namespace std
