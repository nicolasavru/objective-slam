#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>

#include <boost/format.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
// #include <boost/log/expressions.hpp>
// #include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
// #include <boost/log/sources/severity_logger.hpp>
// #include <boost/log/sources/record_ostream.hpp>
#include <boost/program_options.hpp>
#include <Eigen/Core>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/angles.h>
#include <pcl/common/distances.h>
#include <pcl/common/geometry.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/ppf.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/voxel_grid.h>
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
#include <vector>

#include "ppf.h"
#include "vector_ops.h"
#include "impl/cycle_iterator.hpp"
#include "impl/scene_generation.hpp"
#include "impl/util.hpp"
#include "linalg.h"
#include "kernel.h"

// Types
// typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::PPFSignature FeatureT;
// typedef pcl::FPFHEstimationOMP<pcl::PointNormal,pcl::PointNormal,FeatureT> FeatureEstimationT;
typedef pcl::PPFEstimation<pcl::PointNormal, pcl::PointNormal, FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;

void ptr_test(pcl::PointCloud<pcl::PointNormal> *scene_cloud_ptr){
    BOOST_LOG_TRIVIAL(debug) << boost::format("foo-1: %p, %lu, %lu") %
        scene_cloud_ptr % scene_cloud_ptr->points.size() % scene_cloud_ptr->size();
}

std::vector<uchar3> colors = {
    uchar3{255,   0, 0},   // red
    uchar3{  0, 255, 0},   // green
    uchar3{  0,   0, 255}, // blue
    uchar3{  0, 255, 255}, // cyan
    uchar3{255,   0, 255}, // magenta
    uchar3{255, 255, 0},   // yellow
};


namespace po = boost::program_options;


// Workaround for a bug in pcl::geometry::distance.
template <typename PointT>
inline float distance (const PointT& p1, const PointT& p2){
    Eigen::Vector3f diff = p1.getVector3fMap() - p2.getVector3fMap();
    return (diff.norm ());
}


template <typename Point>
typename pcl::PointCloud<Point>::Ptr randomDownsample(typename pcl::PointCloud<Point>::Ptr cloud, float p){
    pcl::RandomSample<Point> rs;
    typename pcl::PointCloud<Point>::Ptr filtered_cloud(new pcl::PointCloud<Point>);
    rs.setSample(p * cloud->size());
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


// http://stackoverflow.com/questions/26389297/how-to-parse-comma-separated-values-with-boostprogram-options
// http://stackoverflow.com/questions/18378798/use-boost-program-options-to-parse-an-arbitrary-string
class CommaSeparatedVector{
  public:
    std::vector<std::string> values;

    static void tokenize(const std::string& input, std::vector<std::string>& output,
                  const std::string& delimiters = ","){
        typedef boost::escaped_list_separator<char> separator_type;
        separator_type separator("\\",    // The escape characters.
                                 ",",    // The separator characters.
                                 "\"\'"); // The quote characters.

        // Tokenize the intput.
        boost::tokenizer<separator_type> tokens(input, separator);

        // Copy non-empty tokens from the tokenizer into the result.
        copy_if(tokens.begin(), tokens.end(), std::back_inserter(output),
                !boost::bind(&std::string::empty, _1));
    }

    friend std::istream& operator>>(std::istream& in, CommaSeparatedVector &value){
        std::string token;
        in >> token;
        tokenize(token, value.values);
        return in;
    }
};

po::variables_map configure_options(int argc, char **argv){
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        // runstate parameters
        ("dev", po::value<int>()->default_value(1), "CUDA device to use")
        ("logfile", po::value<std::string>(), "log file")
        ("loglevel", po::value<boost::log::trivial::severity_level>()->
         default_value(boost::log::trivial::info), "log level")

        // algorithm parameters
        ("tau_d", po::value<CommaSeparatedVector>()->multitoken()->required(), "voxel grid factors")
        ("scene_leaf_size", po::value<float>()->default_value(10.0), "voxel grid factor")
        ("ref_point_df", po::value<unsigned int>()->default_value(1),
         "scene reference point downsample factor")
        ("vote_count_threshold", po::value<float>()->default_value(0.4),
         "percentile of vote counts which are discarded")
        ("cpu_clustering", po::value<bool>()->default_value(false), "whether to cluster on the cpu")
        ("validation_translation_threshold", po::value<float>()->default_value(0.1),
         "validation_translation_threshold")
        ("validation_rotation_threshold", po::value<float>()->default_value(12),
         "validation_rotation_threshold")

        // input files
        ("scene_files", po::value<CommaSeparatedVector>()->multitoken()->required(),
         "ply files to find models in")
        ("model_files", po::value<CommaSeparatedVector>()->multitoken()->required(),
         "ply files to find in scenes")
        ("training_files", po::value<CommaSeparatedVector>()->multitoken(),
         "ply files to generate training scenes with")
        ("validation_files", po::value<CommaSeparatedVector>()->multitoken(),
         "file with ground truth transformations for models in scenes")

        // output parameters
        ("show_normals", po::value<bool>()->default_value(true), "whether to display normals")
        ("visualize", po::value<bool>()->default_value(true),
         "whether to visualize the scenes and models or only return text output")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if(vm.count("help")){
        cout << desc << "\n";
        exit(1);
    }

    // http://stackoverflow.com/questions/5395503/required-and-optional-arguments-using-boost-library-program-options
    po::notify(vm);

    return vm;
}

void init_logging(po::variables_map vm){
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

    if(vm.count("logfile")){
        boost::log::add_file_log(
            boost::log::keywords::file_name = vm["logfile"].as<std::string>(),
            boost::log::keywords::format = "[%TimeStamp%]: %Message%",
            boost::log::keywords::auto_flush = true);
        boost::log::add_common_attributes();

        boost::log::trivial::severity_level loglevel =
            vm["loglevel"].as<boost::log::trivial::severity_level>();
        boost::log::core::get()->set_filter(
            boost::log::trivial::severity >= loglevel);
    }
};

// Align a rigid object to a scene with clutter and occlusions
int main(int argc, char **argv){
    srand(time(0));
    po::variables_map vm = configure_options(argc, argv);
    init_logging(vm);

    // Load model and scene
    // MATLAB drost.m:5-39
    float scene_leaf_size = vm["scene_leaf_size"].as<float>();
    std::vector<float> scene_d_dists;
    std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> scene_clouds;
    CommaSeparatedVector scene_files = vm["scene_files"].as<CommaSeparatedVector>();
    for(std::string scene_file: scene_files.values){
        scene_clouds.push_back(pcl::PointCloud<pcl::PointNormal>::Ptr(
                                   new pcl::PointCloud<pcl::PointNormal>));
        BOOST_LOG_TRIVIAL(info) <<
            boost::format("Loading scene point cloud: %s") % scene_file.c_str();
        if(pcl::io::loadPLYFile<pcl::PointNormal>(scene_file, *scene_clouds.back()) < 0){
            BOOST_LOG_TRIVIAL(error) << "Error loading scene file!";
            exit(1);
        }

        scene_d_dists.push_back(scene_leaf_size);
    }

    std::vector<float> tau_d;
    CommaSeparatedVector tau_d_strs = vm["tau_d"].as<CommaSeparatedVector>();
    for(std::string tau_d_str: tau_d_strs.values){
        tau_d.push_back(std::atof(tau_d_str.c_str()));
    }

    std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> model_clouds;
    std::vector<float> model_d_dists;
    CommaSeparatedVector model_files = vm["model_files"].as<CommaSeparatedVector>();

    if(tau_d.size() != model_files.values.size()){
        BOOST_LOG_TRIVIAL(error) << "Each model must have an associated tau_d.";
        exit(1);
    }
    for(int i = 0; i < model_files.values.size(); i++){
        std::string model_file = model_files.values[i];
        model_clouds.push_back(pcl::PointCloud<pcl::PointNormal>::Ptr(
                                   new pcl::PointCloud<pcl::PointNormal>));

        BOOST_LOG_TRIVIAL(info) <<
            boost::format("Loading model point cloud: %s") % model_file.c_str();
        if(pcl::io::loadPLYFile<pcl::PointNormal>(model_file, *model_clouds.back()) < 0){
            BOOST_LOG_TRIVIAL(error) << "Error loading model file!";
            exit(1);
        }

        pcl::PointNormal minPt, maxPt;
        // Finding the max pairwise distance is epxensive, to
        // approximate it with the max difference between coords.
        pcl::getMinMax3D(*model_clouds.back(), minPt, maxPt);
        float3 model_diam = {maxPt.x - minPt.x,
                             maxPt.y - minPt.y,
                             maxPt.z - minPt.z};
        model_d_dists.push_back(tau_d[i]*max(model_diam));
        BOOST_LOG_TRIVIAL(debug) <<
            boost::format("model_diam, d_dist: (%f, %f, %f), %f") %
            model_diam.x % model_diam.y % model_diam.z % model_d_dists.back();
    }

    std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> training_clouds;
    if(vm.count("training_files")){
        CommaSeparatedVector training_files = vm["training_files"].as<CommaSeparatedVector>();
        for(std::string training_file: training_files.values){
            training_clouds.push_back(pcl::PointCloud<pcl::PointNormal>::Ptr(
                                          new pcl::PointCloud<pcl::PointNormal>));
            BOOST_LOG_TRIVIAL(info) <<
                boost::format("Loading training point cloud: %s") % training_file.c_str();
            if(pcl::io::loadPLYFile<pcl::PointNormal>(training_file, *training_clouds.back()) < 0){
                BOOST_LOG_TRIVIAL(error) << "Error loading scene file!";
                exit(1);
            }
        }
    }

    float d_dist3 = 15;

    // Downsample
    BOOST_LOG_TRIVIAL(info) << "Downsampling...";
    int model_n = 50;
    int scene_n = 50;
    // int model_n = 1000;
    // int scene_n = 500;
    int empty_scene_n = 1000;

    // shared_ptr &operator=(shared_ptr const &r) is equivalent to shared_ptr(r).swap(*this),
    // so, in teach loop, the original cloud gets de-allocated when new_* goes out of scope.
    // http://www.boost.org/doc/libs/1_60_0/libs/smart_ptr/shared_ptr.htm#assignment
    for(pcl::PointCloud<pcl::PointNormal>::Ptr& scene: scene_clouds){
        BOOST_LOG_TRIVIAL(info) <<
            boost::format("Scene size before filtering: %u (%u x %u)") %
            scene->size() % scene->width % scene->height;
        // scene = sequentialDownsample<pcl::PointNormal>(scene, scene_n);
        // scene = randomDownsample<pcl::PointNormal>(scene, 2500.0/scene->size());
        pcl::PointCloud<pcl::PointNormal>::Ptr new_scene =
            voxelGridDownsample<pcl::PointNormal>(scene, scene_leaf_size);
        scene = new_scene;
        BOOST_LOG_TRIVIAL(info) <<
            boost::format("Scene size after filtering: %u (%u x %u)") %
            scene->size() % scene-> width % scene->height;
    }

    for(int i = 0; i < model_clouds.size(); i++){
        pcl::PointCloud<pcl::PointNormal>::Ptr model = model_clouds[i];
        BOOST_LOG_TRIVIAL(info) <<
            boost::format("Model size before filtering: %u (%u x %u)") %
            model->size() % model->width % model->height;
        // model = sequentialDownsample<pcl::PointNormal>(model, model_n);
        // model = randomDownsample<pcl::PointNormal>(model, 2500.0/model->size());
        pcl::PointCloud<pcl::PointNormal>::Ptr new_model =
            voxelGridDownsample<pcl::PointNormal>(model, model_d_dists[i]);
        BOOST_LOG_TRIVIAL(info) <<
            boost::format("Model size after filtering: %u (%u x %u)") %
            new_model->size() % new_model-> width % new_model->height;
        model_clouds[i] = new_model;
    }

    for(pcl::PointCloud<pcl::PointNormal>::Ptr& training_cloud: training_clouds){
        BOOST_LOG_TRIVIAL(info) <<
            boost::format("Training cloud size before filtering: %u (%u x %u)") %
            training_cloud->size() % training_cloud->width % training_cloud->height;
        // training_cloud = sequentialDownsample<pcl::PointNormal>(training_cloud, training_cloud_n);
        // training_cloud = randomDownsample<pcl::PointNormal>(training_cloud, 2500.0/training_cloud->size());
        pcl::PointCloud<pcl::PointNormal>::Ptr new_training_cloud =
            voxelGridDownsample<pcl::PointNormal>(training_cloud, d_dist3);
        training_cloud = new_training_cloud;
        BOOST_LOG_TRIVIAL(info) <<
            boost::format("Training cloud size after filtering: %u (%u x %u)") %
            training_cloud->size() % training_cloud-> width % training_cloud->height;
    }

    // // Estimate normals for object
    // pcl::console::print_highlight("Estimating object normals...\n");
    // // pcl::NormalEstimationOMP<pcl::PointNormal, pcl::PointNormal> nest_obj;
    // pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> nest_obj;
    // nest_obj.setRadiusSearch(0.1);
    // // pcl::search::KdTree<pcl::PointNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointNormal>);
    // // nest_obj.setSearchMethod (tree);
    // // nest_obj.setKSearch(15);
    // nest_obj.setInputCloud(object);
    // nest_obj.compute(*object);

    // // Estimate normals for scene
    // pcl::console::print_highlight("Estimating scene normals...\n");
    // // pcl::NormalEstimationOMP<pcl::PointNormal, pcl::PointNormal> nest_scene;
    // pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> nest_scene;
    // nest_scene.setRadiusSearch(0.3);
    // // pcl::search::KdTree<pcl::PointNormal>::Ptr tree_scene (new pcl::search::KdTree<pcl::PointNormal>);
    // // nest_obj.setSearchMethod (tree_scene);
    // // nest_scene.setKSearch(15);
    // nest_scene.setInputCloud(scene);
    // nest_scene.compute(*scene);

    // CenterScene(*scene);

    // MATLAB drost.m 59-63 model_description() and voting_scheme()
    // pass in object and scene, get back transformation matching object to scene
    // pcl::PointCloud<pcl::PointNormal> test_cloud = pcl::PointCloud<pcl::PointNormal>(*scene);
    // pcl::PointCloud<pcl::PointNormal> *test_cloud2 = new pcl::PointCloud<pcl::PointNormal>(
    //     *scene_clouds[0]);
    // /* DEBUG */
    // fprintf(stderr, "foo0: %p, %lu, %lu\n", scene.get(), scene.get()->points.size(), scene.get()->size());
    // fprintf(stderr, "foo0: %p, %lu, %lu\n", &test_cloud, (&test_cloud)->points.size(), (&test_cloud)->size());
    // // fprintf(stderr, "foo0: %p, %d, %d\n", test_cloud2, test_cloud2->points.size(), test_cloud2->size());
    // ptr_test(test_cloud2);
    // ptr_test_cu(test_cloud2);
    // ptr_test_cu2(*test_cloud2);
    // ptr_test_cu3(*test_cloud2);
    // ptr_test_cu4(*test_cloud2);
    /* DEBUG */
    // float *model_weights = (float *)malloc(model->size()*sizeof(float));
    std::vector<std::vector<Eigen::Matrix4f>> results =
        ppf_registration(scene_clouds, model_clouds, training_clouds,
                         model_d_dists, vm["ref_point_df"].as<unsigned int>(),
                         vm["vote_count_threshold"].as<float>(),
                         vm["cpu_clustering"].as<bool>(), vm["dev"].as<int>(),
                         NULL);

    // pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr color_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    // pcl::copyPointCloud(*model, *color_cloud);
    // for(int i = 0; i < model->size(); i++){
    //     float weight = model_weights[i]/8;
    //     uint8_t r = (uint8_t) (255*weight);
    //     uint8_t g = (uint8_t) (165*weight);
    //     uint8_t b = (uint8_t) (0*weight);
    //     uint32_t rgb =
    //         static_cast<uint32_t>(r) << 16 |
    //         static_cast<uint32_t>(g) << 8  |
    //         static_cast<uint32_t>(b);
    //     (*color_cloud)[i].rgb = *reinterpret_cast<float *>(&rgb);
    // }

    if(vm.count("validation_files")){
        CommaSeparatedVector validation_files = vm["validation_files"].as<CommaSeparatedVector>();
        for(int i = 0; i < scene_clouds.size(); i++){
            for(int j = 0; j < model_clouds.size(); j++){
                ifstream validation_file_stream;
                validation_file_stream.open(validation_files.values[i*model_clouds.size() + j]);
                Eigen::Matrix4f truth;
                validation_file_stream >> truth;
                BOOST_LOG_TRIVIAL(info) <<
                    boost::format("Transformations for %s in %s:") %
                    model_files.values[j] % scene_files.values[i];
                validation_file_stream.close();
                BOOST_LOG_TRIVIAL(info) << "Estimated transformation:";
                BOOST_LOG_TRIVIAL(info) << results[i][j];
                BOOST_LOG_TRIVIAL(info) << "Ground truth:";
                BOOST_LOG_TRIVIAL(info) << truth;

                float model_diam = model_d_dists[j] / tau_d[j];
                float2 dist = ht_dist(results[i][j], truth);
                float trans_thresh = vm["validation_translation_threshold"].as<float>()*model_diam;
                float rot_thresh = pcl::deg2rad(vm["validation_rotation_threshold"].as<float>());
                bool trans_match = dist.x < trans_thresh;
                bool rot_match = dist.y < rot_thresh;
                bool match = trans_match && rot_match;

                BOOST_LOG_TRIVIAL(info) <<
                    boost::format("Distance (trans, rot): %f, %f") % dist.x % dist.y;
                BOOST_LOG_TRIVIAL(info) <<
                    boost::format("Threshold (validation_rotation_threshold*model_diam , 12 deg): %f, %f") %
                    trans_thresh % rot_thresh;
                BOOST_LOG_TRIVIAL(info) <<
                    boost::format("Match (trans, rot): %d, %d") % trans_match % rot_match;
                cout << boost::format("%d") % match << std::endl;
            }
        }
    }
    // MATLAB drost.m:80-108
    // cout << T << endl;
    if(vm["visualize"].as<bool>()){
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
            new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer->setBackgroundColor(0, 0, 0);
        viewer->addCoordinateSystem (1.0, "coords", 0);
        viewer->initCameraParameters();

        // TODO: gather the cloud IDs and color handlers into vectors to de-allocate them later.
        for(int i = 0; i < scene_clouds.size(); i++){
            std::string *cloud_id = new std::string(str(boost::format("scene_%d") % i));
            std::string *cloud_normals_id = new std::string(str(boost::format("scene_normals_%d") % i));
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> *white_color =
                new pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal>(
                    scene_clouds[i], 255, 255, 255);

            viewer->addPointCloud<pcl::PointNormal>(scene_clouds[i], *white_color, *cloud_id);
            viewer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, *cloud_id);

            if(vm["show_normals"].as<bool>()){
                viewer->addPointCloudNormals<pcl::PointNormal, pcl::PointNormal>(
                    scene_clouds[i], scene_clouds[i], 1, 3, *cloud_normals_id);
            }
        }

        // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> red_color(model, 255, 0, 0);
        // viewer->addPointCloud<pcl::PointNormal>(model, red_color, "model");
        // viewer->addPointCloudNormals<pcl::PointNormal, pcl::PointNormal>(model, model, 5, 0.05, "model_normals");
        // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "model");

        // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb_color(color_cloud);
        // viewer->addPointCloud<pcl::PointXYZRGBNormal>(color_cloud, rgb_color, "color_cloud");
        // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "color_cloud");

        cycle_iterator<std::vector<uchar3>::iterator> color_it(colors.begin(), colors.end());
        for(int i = 0; i < scene_clouds.size(); i++){
            for(int j = 0; j < model_clouds.size(); j++){
                pcl::PointCloud<pcl::PointNormal>::Ptr model_aligned(new pcl::PointCloud<pcl::PointNormal>);
                pcl::transformPointCloudWithNormals(*model_clouds[j], *model_aligned, results[i][j]);
                uchar3 color = *color_it++;

                std::string *cloud_id = new std::string(
                    str(boost::format("model_aligned_%d_%d") % i % j));
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> *color_h =
                    new pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal>(
                        model_aligned, color.x, color.y, color.z);

                viewer->addPointCloud<pcl::PointNormal>(model_aligned, *color_h, *cloud_id);
                viewer->setPointCloudRenderingProperties(
                    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, *cloud_id);

                if(vm["show_normals"].as<bool>()){
                    std::string *cloud_normals_id =
                        new std::string(str(boost::format("model_aligned_normals_%d_%d") % i % j));
                    viewer->addPointCloudNormals<pcl::PointNormal, pcl::PointNormal>(
                        model_aligned, model_aligned, 1, 3, *cloud_normals_id);
                    viewer->setPointCloudRenderingProperties(
                        pcl::visualization::PCL_VISUALIZER_COLOR,
                        color.x/255.0, color.y/255.0, color.z/255.0,
                        *cloud_normals_id);
                }
            }
        }

        while (!viewer->wasStopped ()){
            viewer->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }
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
