#include <Eigen/Core>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/ppf.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ppf_registration.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "ppf.h"
//#include "my_ppf_registration.h"

// Types
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
// typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::PPFSignature FeatureT;
// typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PPFEstimation<PointNT, PointNT, FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

// Align a rigid object to a scene with clutter and occlusions
int main(int argc, char **argv){
    // Point clouds
    PointCloudT::Ptr object(new PointCloudT);
    PointCloudT::Ptr object_aligned(new PointCloudT);
    PointCloudT::Ptr scene(new PointCloudT);
    FeatureCloudT::Ptr object_features(new FeatureCloudT);
    FeatureCloudT::Ptr scene_features(new FeatureCloudT);

    // cuda compilation experimentation
    if (argc == 2){
        int N = atoi(argv[1]);
        ply_load_main("/tmp/points.txt", "/tmp/norms.txt", N, 0);
        return 2;
    } else if (argc == 3){
//        int N = atoi(argv[1]);
//        int devUse = atoi(argv[2]);
//        ply_load_main("/tmp/points.txt", "/tmp/norms.txt", N, devUse);
//        return 2;
    }

    // Get input object and scene
    if (argc != 3){
        pcl::console::print_error("Syntax is: %s object.ply scene.ply\n",
                argv[0]);
        return (1);
    }

    // Load object and scene
    pcl::console::print_highlight("Loading point clouds...\n");
    if (pcl::io::loadPLYFile<PointNT>(argv[1], *object) < 0
            || pcl::io::loadPLYFile<PointNT>(argv[2], *scene) < 0) {
        pcl::console::print_error("Error loading object/scene file!\n");
        return (1);
    }

    // Downsample
    pcl::console::print_highlight("Downsampling...\n");
    pcl::VoxelGrid<PointNT> grid;
    float leaf = 0.01f;
    grid.setLeafSize(leaf, leaf, leaf);
    std::cerr << "object before filtering: " << object->width * object->height
            << " data points (" << pcl::getFieldsList(*object) << ")."
            << std::endl;
    grid.setInputCloud(object);
    grid.filter(*object);
    std::cerr << "object after filtering: " << object->width * object->height
            << " data points (" << pcl::getFieldsList(*object) << ")."
            << std::endl;
    leaf = 0.02f;
    grid.setLeafSize(leaf, leaf, leaf);
    std::cerr << "scene before filtering: " << scene->width * scene->height
            << " data points (" << pcl::getFieldsList(*scene) << ")."
            << std::endl;
    grid.setInputCloud(scene);
    grid.filter(*scene);
    std::cerr << "scene after filtering: " << scene->width * scene->height
            << " data points (" << pcl::getFieldsList(*scene) << ")."
            << std::endl;

    // Estimate normals for object
    pcl::console::print_highlight("Estimating object normals...\n");
    pcl::NormalEstimationOMP<PointNT, PointNT> nest_obj;
    nest_obj.setRadiusSearch(0.1);
    nest_obj.setInputCloud(object);
    nest_obj.compute(*object);

    // Estimate normals for scene
    pcl::console::print_highlight("Estimating scene normals...\n");
    pcl::NormalEstimationOMP<PointNT, PointNT> nest_scene;
    nest_scene.setRadiusSearch(0.1);
    nest_scene.setInputCloud(scene);
    nest_scene.compute(*scene);

    // TODO: Check for malloc errors
    float3 *scene_points = (float3 *) malloc(scene->points.size()*sizeof(float3));
    float3 *scene_normals = (float3 *) malloc(scene->points.size()*sizeof(float3));
    float3 *object_points = (float3 *) malloc(object->points.size()*sizeof(float3));
    float3 *object_normals = (float3 *) malloc(object->points.size()*sizeof(float3));
    for (int i=0; i<scene->points.size(); i++){
        scene_points->x = scene->points[i].x;
        scene_points->y = scene->points[i].y;
        scene_points->z = scene->points[i].z;
        scene_normals->x = scene->points[i].normal_x;
        scene_normals->y = scene->points[i].normal_y;
        scene_normals->z = scene->points[i].normal_z;
    }

    for (int i=0; i<object->points.size(); i++){
        object_points->x = object->points[i].x;
        object_points->y = object->points[i].y;
        object_points->z = object->points[i].z;
        object_normals->x = object->points[i].normal_x;
        object_normals->y = object->points[i].normal_y;
        object_normals->z = object->points[i].normal_z;
    }
    ply_load_main(scene_points, scene_normals, scene->points.size(), object_points,
                  object_normals, object->points.size(), 0);

//    // // Estimate features
//    // pcl::console::print_highlight ("Estimating features...\n");
//    // FeatureEstimationT fest;
//    // fest.setRadiusSearch (0.025);
//    // fest.setInputCloud (object);
//    // fest.setInputNormals (object);
//    // fest.compute (*object_features);
//    // fest.setInputCloud (scene);
//    // fest.setInputNormals (scene);
//    // fest.compute (*scene_features);
//
//    // Estimate features
//    FeatureEstimationT fest;
//    // fest.setRadiusSearch (0.025);
//    pcl::console::print_highlight("Estimating object features...\n");
//    fest.setInputCloud(object);
//    fest.setInputNormals(object);
//    fest.compute(*object_features);
//    // pcl::console::print_highlight ("Estimating scene features...\n");
//    // fest.setInputCloud (scene);
//    // fest.setInputNormals (scene);
//    // fest.compute (*scene_features);
//
//    // // Perform alignment
//    // pcl::console::print_highlight ("Starting alignment...\n");
//    // pcl::SampleConsensusPrerejective<PointNT,PointNT,FeatureT> align;
//    // align.setInputSource (object);
//    // align.setSourceFeatures (object_features);
//    // align.setInputTarget (scene);
//    // align.setTargetFeatures (scene_features);
//    // align.setNumberOfSamples (3); // Number of points to sample for generating/prerejecting a pose
//    // align.setCorrespondenceRandomness (2); // Number of nearest features to use
//    // align.setSimilarityThreshold (0.6f); // Polygonal edge length similarity threshold
//    // align.setMaxCorrespondenceDistance (1.5f * leaf); // Set inlier threshold
//    // align.setInlierFraction (0.25f); // Set required inlier fraction
//    // align.align (*object_aligned);
//
//    pcl::PPFHashMapSearch::Ptr searcher(new pcl::PPFHashMapSearch);
//    searcher->setInputFeatureCloud(object_features);
//
//    // Perform alignment
//    pcl::console::print_highlight("Starting alignment...\n");
//    pcl::PPFRegistration<PointNT, PointNT> align;
//    align.setInputSource(object);
//    // align.setSourceFeatures (object_features);
//    align.setInputTarget(scene);
//    align.setSearchMethod(searcher);
//    // align.setTargetFeatures (scene_features);
//    // align.setNumberOfSamples (3); // Number of points to sample for generating/prerejecting a pose
//    // align.setCorrespondenceRandomness (2); // Number of nearest features to use
//    // align.setSimilarityThreshold (0.6f); // Polygonal edge length similarity threshold
//    // align.setMaxCorrespondenceDistance (1.5f * leaf); // Set inlier threshold
//    // align.setInlierFraction (0.25f); // Set required inlier fraction
//    align.align(*object_aligned);
//
//    if (align.hasConverged()) {
//        // Print results
//        Eigen::Matrix4f transformation = align.getFinalTransformation();
//        pcl::console::print_info("    | %6.3f %6.3f %6.3f | \n",
//                transformation(0, 0), transformation(0, 1),
//                transformation(0, 2));
//        pcl::console::print_info("R = | %6.3f %6.3f %6.3f | \n",
//                transformation(1, 0), transformation(1, 1),
//                transformation(1, 2));
//        pcl::console::print_info("    | %6.3f %6.3f %6.3f | \n",
//                transformation(2, 0), transformation(2, 1),
//                transformation(2, 2));
//        pcl::console::print_info("\n");
//        pcl::console::print_info("t = < %0.3f, %0.3f, %0.3f >\n",
//                transformation(0, 3), transformation(1, 3),
//                transformation(2, 3));
//        pcl::console::print_info("\n");
//        // pcl::console::print_info ("Inliers: %i/%i\n", align.getInliers ().size (), object->size ());
//
//        // Show alignment
//        pcl::visualization::PCLVisualizer visu("Alignment");
//        visu.addPointCloud(scene, ColorHandlerT(scene, 0.0, 255.0, 0.0),
//                "scene");
//        visu.addPointCloud(object_aligned,
//                ColorHandlerT(object_aligned, 0.0, 0.0, 255.0),
//                "object_aligned");
//        visu.spin();
//    } else {
//        pcl::console::print_error("Alignment failed!\n");
//        return (1);
//    }

    return (0);
}
