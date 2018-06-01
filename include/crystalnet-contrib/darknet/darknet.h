// https://github.com/pjreddie/darknet.git
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

extern void reorg_cpu(float *x, int w, int h, int c, int batch, int stride,
                      int forward, float *out);
extern void forward_region_layer(float *input, int batch, float *output,
                                 int outputs, int _n, int w, int h);

#ifdef __cplusplus
}
#endif
