#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

int main (int argc, char** argv){
  pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2 ());
  pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2 ());

  // Fill in the cloud data
  pcl::PLYReader reader;
  reader.read(argv[1], *cloud);

  std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height
            << " data points (" << pcl::getFieldsList (*cloud) << ")." << std::endl;

  // Create the filtering object
  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(0.01f, 0.01f, 0.01f);
  sor.filter(*cloud_filtered);

  std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height 
            << " data points (" << pcl::getFieldsList (*cloud_filtered) << ")." << std::endl;

  pcl::PLYWriter writer;
  writer.write(argv[2], *cloud_filtered, 
               Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), false, false);

  return (0);
}
