#include <cmath>

#include <crystalnet-contrib/yolo/detection.hpp>
#include <crystalnet-contrib/yolo/input.hpp>

detection_list_t get_detections(const tensor_ref_t &t,
                                const tensor_ref_t &biases)
{
    TRACE(__func__);
    // const auto names = ();

    using T = float;

    const uint32_t n = 5;
    const uint32_t classes = 80;
    const uint32_t coords = 4;

    const uint32_t h = 13;
    const uint32_t w = 13;

    const auto r =
        ranked<4, T>(t.reshape(shape_t(n, coords + 1 + classes, h, w)));
    const auto e = ranked<2, T>(biases.reshape(shape_t(n, 2)));

    // [n, coords + 1 + classes, h, w]
    //
    const uint32_t detections = n * h * w;
    logf("%d detections", detections);
    std::vector<std::unique_ptr<darknet::detection_t>> dets;
    for (auto l : range(n)) {
        for (auto i : range(h)) {
            for (auto j : range(w)) {
                // uint32_t idx = 0;  // 0 <= idx <= coords + classes
                const T c0 = r.at(l, 0, i, j);
                const T c1 = r.at(l, 1, i, j);
                const T c2 = r.at(l, 2, i, j);
                const T c3 = r.at(l, 3, i, j);

                darknet::bbox_t bbox;
                bbox.cy = (j + c1) / h;
                bbox.cx = (i + c0) / w;
                bbox.h = e.at(n, 1) * std::exp(c3) / h;
                bbox.w = e.at(n, 0) * std::exp(c2) / w;

                dets.push_back(
                    std::make_unique<darknet::detection_t>(classes, coords));
                const auto &d = dets[dets.size() - 1];
                d->objectness = 0;
                d->bbox = bbox;
                d->i = i;
                d->j = j;
                {
                    const auto probs = ranked<1, T>(ref(d->probs));
                    // std::vector<float> probs(classes);
                    for (auto k : range(classes)) {
                        probs.at(k) = r.at(l, 5 + k, i, j);
                    }
                }
                //
            }
        }
    }
    return dets;
}
