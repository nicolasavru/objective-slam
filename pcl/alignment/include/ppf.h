#ifndef __PPF_H
#define __PPF_H

#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "debug.h"

int hist_main( void );

int ply_load_main(char *point_path, char *norm_path, int N, int devUse);
void ptr_test_cu(pcl::PointCloud<pcl::PointNormal> *scene_cloud_ptr);
void ptr_test_cu2(pcl::PointCloud<pcl::PointNormal> scene_cloud_ptr);
void ptr_test_cu3(pcl::PointCloud<pcl::PointNormal> &scene_cloud);
void ptr_test_cu4(const pcl::PointCloud<pcl::PointNormal> &scene_cloud);

Eigen::Matrix4f ply_load_main(pcl::PointCloud<pcl::PointNormal> *scene_cloud,
                              float3 *scenePoints, float3 *sceneNormals, int sceneN,
                              pcl::PointCloud<pcl::PointNormal> *object_cloud_ptr,
                              float3 *objectPoints, float3 *objectNormals, int objectN,
                              int devUse);

#endif /* __PPF_H */
