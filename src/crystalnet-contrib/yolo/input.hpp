#pragma once
#include <crystalnet-contrib/vis/vis.hpp>
#include <crystalnet-contrib/yolo/detection.hpp>
#include <crystalnet-contrib/yolo/input.hpp>
#include <crystalnet-contrib/yolo/options.hpp>
#include <crystalnet-contrib/yolo/yolo.h>
#include <crystalnet-ext.h>
#include <crystalnet/core/tracer.hpp>  // TODO: don't include private headers
#include <crystalnet/debug/debug.hpp>
#include <crystalnet/model/model.hpp>

extern const tensor_t *load_test_image(const char * /* filename */);
extern const tensor_t *load_resized_test_image(const char * /* filename */);

extern void load_parameters(const model_t *,
                            const std::experimental::filesystem::path &);
extern std::vector<std::string>
load_name_list(const std::experimental::filesystem::path &);

extern std::string extension(const std::experimental::filesystem::path &);
