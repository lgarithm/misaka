#include <misaka.h>
#include <misaka/core/gc.hpp>
#include <misaka/model/layer.hpp>

static GC<layer_t> gc;

layer_t *layer_fc = gc(new fc_layer_t(1024)); // TODO: customize layer size
