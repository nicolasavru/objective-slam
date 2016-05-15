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
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ppf_registration.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/random_sample.h>

#include "ppf.h"
#include "vector_ops.h"
#include "impl/scene_generation.hpp"
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

template <typename Point>
typename pcl::PointCloud<Point>::Ptr randomDownsample(typename pcl::PointCloud<Point>::Ptr cloud, int n){
    pcl::RandomSample<Point> rs;
    typename pcl::PointCloud<Point>::Ptr filtered_cloud(new pcl::PointCloud<Point>);
    rs.setSample(1.0/n * cloud->size());
    rs.setInputCloud(cloud);
    rs.filter(*filtered_cloud);
    return filtered_cloud;
}

template <typename Point>
typename pcl::PointCloud<Point>::Ptr sequentialDownsample(typename pcl::PointCloud<Point>::Ptr cloud, int n){
    typename pcl::PointCloud<Point>::Ptr filtered_cloud(new pcl::PointCloud<Point>);
    int i = 0;
    for (typename pcl::PointCloud<Point>::iterator it = cloud->begin(); it != cloud->end(); ++it, ++i){
        if(i % n == 0){
            filtered_cloud->push_back(*it);
        }
    }
    return filtered_cloud;
}

template <typename Point>
typename pcl::PointCloud<Point>::Ptr voxelGridDownsample(typename pcl::PointCloud<Point>::Ptr cloud,
                                                         float leaf){
    typename pcl::PointCloud<Point>::Ptr filtered_cloud(new pcl::PointCloud<Point>);
    pcl::VoxelGrid<Point> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(leaf, leaf, leaf);
    sor.filter(*filtered_cloud);
    return filtered_cloud;
}


void ptr_test(pcl::PointCloud<pcl::PointNormal> *scene_cloud_ptr){
    /* DEBUG */
    fprintf(stderr, "foo-1: %p, %d, %d\n", scene_cloud_ptr, scene_cloud_ptr->points.size(), scene_cloud_ptr->size());
}

// Align a rigid object to a scene with clutter and occlusions
int main(int argc, char **argv){
    srand(time(0));

    // Point clouds
    PointCloudT::Ptr object(new PointCloudT);
    PointCloudT::Ptr object_aligned(new PointCloudT);
    PointCloudT::Ptr scene(new PointCloudT);
    PointCloudT::Ptr empty_scene(new PointCloudT);
    FeatureCloudT::Ptr object_features(new FeatureCloudT);
    FeatureCloudT::Ptr scene_features(new FeatureCloudT);

    // Get input object and scene
    if (argc != 4){
        pcl::console::print_error("Syntax is: %s object.ply scene.ply empty_scene.ply\n",
                argv[0]);
        return (1);
    }

    // Load object and scene
    // MATLAB drost.m:5-39
    pcl::console::print_highlight("Loading object point cloud...\n");
    if (pcl::io::loadPLYFile<PointNT>(argv[1], *object) < 0) {
        pcl::console::print_error("Error loading object file!\n");
        return (1);
    }
    pcl::console::print_highlight("Loading scene point cloud...\n");
    if (pcl::io::loadPLYFile<PointNT>(argv[2], *scene) < 0) {
        pcl::console::print_error("Error loading scene file!\n");
        return (1);
    }
    pcl::console::print_highlight("Loading empty scene point cloud...\n");
    if (pcl::io::loadPLYFile<PointNT>(argv[3], *empty_scene) < 0) {
        pcl::console::print_error("Error loading scene file!\n");
        return (1);
    }


    // Downsample
    pcl::console::print_info("Downsampling...\n");
    int object_n = 2000;
    int scene_n = 1000;
    int empty_scene_n = 1000;
    pcl::console::print_info("Object size before filtering: %u (%u x %u)\n",
                             object->size(), object-> width, object->height);
    object = sequentialDownsample<PointNT>(object, object_n);
    // object = voxelGridDownsample<PointNT>(object, 0.03);
    pcl::console::print_info("Object size after filtering: %u (%u x %u)\n",
                             object->size(), object-> width, object->height);

    pcl::console::print_info("Scene size before filtering: %u (%u x %u)\n",
                             scene->size(), scene-> width, scene->height);
    scene = sequentialDownsample<PointNT>(scene, scene_n);
    // scene = voxelGridDownsample<PointNT>(scene, 0.075);
    pcl::console::print_info("Scene size after filtering: %u (%u x %u)\n",
                             scene->size(), scene-> width, scene->height);

    pcl::console::print_info("Empty scene size before filtering: %u (%u x %u)\n",
                             empty_scene->size(), empty_scene-> width, empty_scene->height);
    empty_scene = sequentialDownsample<PointNT>(empty_scene, empty_scene_n);
    // empty_scene = voxelGridDownsample<PointNT>(empty_scene, 0.075);
    pcl::console::print_info("Empty scene size after filtering: %u (%u x %u)\n",
                             empty_scene->size(), empty_scene-> width, empty_scene->height);

    // // Estimate normals for object
    // pcl::console::print_highlight("Estimating object normals...\n");
    // // pcl::NormalEstimationOMP<PointNT, PointNT> nest_obj;
    // pcl::NormalEstimation<PointNT, PointNT> nest_obj;
    // nest_obj.setRadiusSearch(0.1);
    // // pcl::search::KdTree<PointNT>::Ptr tree (new pcl::search::KdTree<PointNT>);
    // // nest_obj.setSearchMethod (tree);
    // // nest_obj.setKSearch(15);
    // nest_obj.setInputCloud(object);
    // nest_obj.compute(*object);

    // // Estimate normals for scene
    // pcl::console::print_highlight("Estimating scene normals...\n");
    // // pcl::NormalEstimationOMP<PointNT, PointNT> nest_scene;
    // pcl::NormalEstimation<PointNT, PointNT> nest_scene;
    // nest_scene.setRadiusSearch(0.3);
    // // pcl::search::KdTree<PointNT>::Ptr tree_scene (new pcl::search::KdTree<PointNT>);
    // // nest_obj.setSearchMethod (tree_scene);
    // // nest_scene.setKSearch(15);
    // nest_scene.setInputCloud(scene);
    // nest_scene.compute(*scene);

    // CenterScene(*scene);

    // Convert point clouds to arrays of float3.
    // TODO: Check for malloc errors
    float3 *object_points = (float3 *) malloc(object->size()*sizeof(float3));
    float3 *object_normals = (float3 *) malloc(object->size()*sizeof(float3));

    for (int i = 0; i < object->size(); i++){
        (object_points+i)->x = object->points[i].x;
        (object_points+i)->y = object->points[i].y;
        (object_points+i)->z = object->points[i].z;
        (object_normals+i)->x = object->points[i].normal_x;
        (object_normals+i)->y = object->points[i].normal_y;
        (object_normals+i)->z = object->points[i].normal_z;
    }

    // MATLAB drost.m 59-63 model_description() and voting_scheme()
    // pass in object and scene, get back transformation matching object to scene
    pcl::PointCloud<pcl::PointNormal> test_cloud = pcl::PointCloud<pcl::PointNormal>(*scene);
    pcl::PointCloud<pcl::PointNormal> *test_cloud2 = new pcl::PointCloud<pcl::PointNormal>(*scene);
    /* DEBUG */
    fprintf(stderr, "foo0: %p, %d, %d\n", scene.get(), scene.get()->points.size(), scene.get()->size());
    fprintf(stderr, "foo0: %p, %d, %d\n", &test_cloud, (&test_cloud)->points.size(), (&test_cloud)->size());
    // fprintf(stderr, "foo0: %p, %d, %d\n", test_cloud2, test_cloud2->points.size(), test_cloud2->size());
    ptr_test(test_cloud2);
    ptr_test_cu(test_cloud2);
    ptr_test_cu2(*test_cloud2);
    ptr_test_cu3(*test_cloud2);
    ptr_test_cu4(*test_cloud2);
    /* DEBUG */
    float *model_weights = (float *)malloc(object->size()*sizeof(float));
    Eigen::Matrix4f T = ply_load_main(scene.get(),
                                      object.get(), object_points, object_normals, object->size(),
                                      empty_scene.get(), 1, model_weights);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr color_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::copyPointCloud(*object, *color_cloud);
    for(int i = 0; i < object->size(); i++){
        float weight = model_weights[i]/8;
        uint8_t r = (uint8_t) (255*weight);
        uint8_t g = (uint8_t) (165*weight);
        uint8_t b = (uint8_t) (0*weight);
        uint32_t rgb =
            static_cast<uint32_t>(r) << 16 |
            static_cast<uint32_t>(g) << 8  |
            static_cast<uint32_t>(b);
        (*color_cloud)[i].rgb = *reinterpret_cast<float *>(&rgb);
        /* DEBUG */
        fprintf(stderr, "rgb: %x, %u, %u, %u\n", rgb, r, g, b);
        /* DEBUG */
    }
    float3 t = {0, 0, 0};
    float4 r = {0, 0, 0, 0};
    // GenerateSceneWithModel(*object, *scene, t, r);

    // MATLAB drost.m:80-108
    cout << T << endl;
    pcl::transformPointCloudWithNormals(*object, *object_aligned, T);
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    viewer->addPointCloud<PointNT>(scene, "scene");
    viewer->addPointCloudNormals<PointNT, PointNT>(scene, scene, 5, 0.05, "scene_normals");

    // ColorHandlerT red_color(object, 255, 0, 0);
    // viewer->addPointCloud<PointNT>(object, red_color, "object");
    // viewer->addPointCloudNormals<PointNT, PointNT>(object, object, 5, 0.05, "object_normals");
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "object");

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb_color(color_cloud);
    viewer->addPointCloud<pcl::PointXYZRGBNormal>(color_cloud, rgb_color, "color_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "color_cloud");

    ColorHandlerT green_color(object_aligned, 0, 255, 0);
    viewer->addPointCloud<PointNT>(object_aligned, green_color, "object_aligned");
    viewer->addPointCloudNormals<PointNT, PointNT>(object_aligned, object_aligned,
                                                   5, 0.05, "object_aligned_normals");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "object_aligned");

    viewer->addCoordinateSystem (1.0, "foo", 0);
    viewer->initCameraParameters ();

    while (!viewer->wasStopped ()){
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

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
