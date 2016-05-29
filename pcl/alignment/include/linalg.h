#ifndef LINALG_H
#define LINALG_H

#include <cuda.h>
#include <Eigen/Geometry>

__host__ float2 ht_dist(Eigen::Matrix4f a, Eigen::Matrix4f b);


#endif /* LINALG_H */
