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

std::vector<std::vector<Eigen::Matrix4f>> ppf_registration(
    std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> scene_clouds,
    std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> model_clouds,
    std::vector<float> model_d_dists, unsigned int ref_point_downsample_factor,
    float vote_count_threshold, bool cpu_clustering,
    bool use_l1_norm, bool use_averaged_clusters,
    int devUse, float *model_weights);

#endif /* __PPF_H */
