#ifndef __PPF_H
#define __PPF_H

#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__

#include <Eigen/Core>

#include "debug.h"

int hist_main( void );

int ply_load_main(char *point_path, char *norm_path, int N, int devUse);

Eigen::Matrix4f ply_load_main(float3 *scenePoints, float3 *sceneNormals, int sceneN,
                  float3 *objectPoints, float3 *objectNormals, int objectN,
                  int devUse);

#endif /* __PPF_H */
