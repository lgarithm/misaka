#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <crystalnet-ext.h>
#include <crystalnet-internal.h>

#include <crystalnet-contrib/vis/snapshot.hpp>
#include <crystalnet-contrib/vis/vis.hpp>
#include <crystalnet-contrib/yolo/detection.hpp>
#include <crystalnet-contrib/yolo/input.hpp>
#include <crystalnet-contrib/yolo/yolo.h>
#include <crystalnet-contrib/yolo/yolo.hpp>
#include <crystalnet/core/tracer.hpp>  // TODO: don't include private headers
#include <crystalnet/debug/debug.hpp>
#include <crystalnet/model/model.hpp>

namespace fs = std::experimental::filesystem;

std::vector<std::string> coco_names_80;

// using candidate_t = struct {
//     std::string name;
// };

void run(const options_t &opt)
{
    const auto names = load_name_list(opt.darknet_path / "data/coco.names");
    coco_names_80 = names;
    const auto input = [&]() {
        if (extension(opt.filename) == ".bmp" ||
            extension(opt.filename) == ".idx") {
            logf("using resized test image: %s", opt.filename.c_str());
            return load_resized_test_image(opt.filename.c_str());
        }
        return load_test_image(opt.filename.c_str());
    }();
    logf("input image: %s", std::to_string(input->shape).c_str());

    const s_model_t *s_model = []() {
        TRACE(__func__);
        return yolov2();
    }();

    {
        TRACE("main::print symbolic model");
        show_layers(*s_model);
        FILE *fp = fopen("graph.dot", "w");
        graphviz(*s_model, fp);
        fclose(fp);
        system("dot -Tsvg graph.dot -O");
    }

    const auto p_ctx = new_parameter_ctx();
    const model_t *p_model = [&]() {
        TRACE(__func__);
        return realize(p_ctx, s_model, 1);
    }();
    TRACE_IT(load_parameters(p_model, opt.model_dir));
    {
        TRACE("main::print parameters");
        debug<tensor_t>(*p_ctx, [](const tensor_t *value) {
            return summary(r_tensor_ref_t<float>(*value));
        });
    }

    {
        TRACE("main::inference");
        const auto r = ref(*input);
        TRACE_IT(p_model->input.bind(r.reshape(r.shape.batch(1))));
        TRACE_IT(p_model->forward());
    }
    {
        TRACE("main::print layers");
        using T = float;
        const auto brief = summary(r_tensor_ref_t<T>(*input));
        logf("input: %s", brief.c_str());
        show_layers(*p_model, *s_model);
    }
    {
        TRACE("main::save layers");
        save_layers(*p_model, *s_model);
    }
    const auto dets = [&]() {
        using T = float;
        const auto r = ranked<4, T>(p_model->output.value());
        const auto [n, c, h, w] = r.shape.dims;
        printf("%d, %d, %d, %d\n", n, c, h, w);

        const auto y = p_model->output.value().reshape(shape_t(c, h, w));
        const auto anchor_boxes = ref(*p_ctx->index.at("yolov2_31_anchors"));
        return get_detections(y, anchor_boxes);
    }();
    {
        TRACE("main::show detections");
        using T = float;
        for (const auto &d : dets) {
            // logf("%s", p_str(d->bbox));
            const auto probs = ranked<1, T>(ref(d->probs));
            const auto idx =
                std::max_element(probs.data, probs.data + 80) - probs.data;
            logf("(%3d, %3d): %s (%ld): %f",  //
                 d->i, d->j, names.at(idx).c_str(), idx, probs.at(idx));
            if (probs.at(idx) > 0.7) {
                fprintf(stderr, "(%3d, %3d): %-4ld %-20s: %f\n",  //
                        d->i, d->j, idx, names.at(idx).c_str(), probs.at(idx));
                // awk '{print $5, $7}' stderr.log
            }
        }
    }

    del_model(p_model);
    del_s_model(s_model);
    del_tensor(input);
}

int main(int argc, char *argv[])
{
    TRACE(__func__);
    const auto opt = parse_flags(argc, argv);
    try {
        run(opt);
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }
    return 0;
}
