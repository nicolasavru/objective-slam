#include <algorithm>
#include <iterator>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>

#include "model.h"
#include "train_weights.h"
#include "impl/ppf_utils.hpp"
#include "impl/util.hpp"
#include "impl/parallel_hash_array.hpp"
#include "impl/scene_generation.hpp"
#include "kernel.h"
#include "book.h"


// This must be here, as opposed to in train_weights.cpp, due to linking issues.
Model *MODEL_OBJ;
Scene *SCENE_OBJ;
Eigen::Matrix4f TRUTH;

struct high_32_bits : public thrust::unary_function<unsigned long,unsigned int>{
    __host__ __device__
    unsigned int operator()(unsigned long i) const {
        return (unsigned int) (i >> 32);
    }
};


struct low_32_bits : public thrust::unary_function<unsigned long,unsigned int>{
    __host__ __device__
    unsigned int operator()(unsigned long i) const {
        return (unsigned int) (i & (-1ul >> 32));
    }
};

struct float16 {
    float f[16];
};

Model::Model(pcl::PointCloud<pcl::PointNormal> *cloud,
             thrust::host_vector<float3> *points, thrust::host_vector<float3> *normals, int n){
    this->cloud_ptr = cloud;
    this->initPPFs(points, normals, cloud->size());
    this->modelPointVoteWeights = thrust::device_vector<float>(n, 1);

    // For each element of data, compute a 32-bit hash,
    thrust::device_vector<unsigned int> nonunique_hashkeys =
        thrust::device_vector<unsigned int>(this->modelPPFs->size());
    int blocks = std::min(((int)(this->modelPPFs->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    ppf_hash_kernel<<<blocks,BLOCK_SIZE>>>(RAW_PTR(this->modelPPFs),
                                           thrust::raw_pointer_cast(nonunique_hashkeys.data()),
                                           this->modelPPFs->size());
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    this->search_array = ParallelHashArray<unsigned int>(nonunique_hashkeys);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
}

Model::~Model(){
    // TODO
}

void Model::SetModelPointVoteWeights(thrust::device_vector<float> modelPointVoteWeights){
    this->modelPointVoteWeights = modelPointVoteWeights;
}

void Model::ComputeUniqueVotes(Scene *scene){
    thrust::device_vector<unsigned long> *nonunique_votes = new thrust::device_vector<unsigned long>(scene->getModelPPFs()->size()*this->modelPoints->size(),0);
    thrust::device_vector<std::size_t> *sceneIndices =
        this->search_array.GetIndices(*(scene->getHashKeys()));

    int blocks = std::min(((int)(scene->getHashKeys()->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    ppf_vote_kernel<<<blocks,BLOCK_SIZE>>>
        (RAW_PTR(scene->getHashKeys()), RAW_PTR(sceneIndices),
         RAW_PTR(this->search_array.GetHashkeys()), RAW_PTR(this->search_array.GetCounts()),
         RAW_PTR(this->search_array.GetFirstHashkeyIndices()),
         RAW_PTR(this->search_array.GetHashkeyToDataMap()),
         RAW_PTR(this->modelPoints), RAW_PTR(this->modelNormals),
         this->n, RAW_PTR(scene->getModelPoints()),
         RAW_PTR(scene->getModelNormals()), scene->numPoints(),
         RAW_PTR(nonunique_votes),
         scene->getHashKeys()->size());

    thrust::sort(nonunique_votes->begin(), nonunique_votes->end());
    this->votes = new thrust::device_vector<unsigned long>();
    this->voteCounts = new thrust::device_vector<unsigned int>();
    histogram_destructive(*nonunique_votes, *(this->votes), *(this->voteCounts));
}

thrust::device_vector<float>
Model::ComputeWeightedVoteCounts(thrust::device_vector<unsigned long> votes,
                                 thrust::device_vector<unsigned int> vote_counts,
                                 thrust::device_vector<float> modelpoint_vote_weights){
    thrust::device_vector<float> weighted_vote_counts =
        thrust::device_vector<float>(votes.size());

    int blocks = std::min(((int)(votes.size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    vote_weight_kernel<<<blocks,BLOCK_SIZE>>>
        (thrust::raw_pointer_cast(votes.data()),
         thrust::raw_pointer_cast(vote_counts.data()),
         thrust::raw_pointer_cast(modelpoint_vote_weights.data()),
         thrust::raw_pointer_cast(weighted_vote_counts.data()),
         votes.size());

    return weighted_vote_counts;
}

void Model::ComputeTransformations(Scene *scene){
    this->transformations = thrust::device_vector<float>(this->votes->size()*16);
    int blocks = std::min(((int)(this->votes->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    trans_calc_kernel2<<<blocks,BLOCK_SIZE>>>
        (RAW_PTR(votes),
         RAW_PTR(this->modelPoints), RAW_PTR(this->modelNormals),
         RAW_PTR(scene->getModelPoints()), RAW_PTR(scene->getModelNormals()),
         thrust::raw_pointer_cast(this->transformations.data()),
         this->votes->size());
}

thrust::device_vector<float> *Model::ClusterTransformations(){
    thrust::device_vector<float3> transformation_trans =
        thrust::device_vector<float3>(this->votes->size());
    thrust::device_vector<float4> transformation_rots =
        thrust::device_vector<float4>(this->votes->size());
    int blocks = std::min(((int)(this->votes->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    mat2transquat_kernel<<<blocks,BLOCK_SIZE>>>
        (thrust::raw_pointer_cast(this->transformations.data()),
         thrust::raw_pointer_cast(transformation_trans.data()),
         thrust::raw_pointer_cast(transformation_rots.data()),
         this->votes->size());

    thrust::device_vector<unsigned int> nonunique_trans_hash =
        thrust::device_vector<unsigned int>(this->votes->size());
    thrust::device_vector<unsigned int> adjacent_trans_hash =
        thrust::device_vector<unsigned int>(27*this->votes->size());
    trans2idx_kernel<<<blocks,BLOCK_SIZE>>>
    (thrust::raw_pointer_cast(transformation_trans.data()),
     thrust::raw_pointer_cast(nonunique_trans_hash.data()),
     thrust::raw_pointer_cast(adjacent_trans_hash.data()),
     this->votes->size());

    ParallelHashArray<unsigned int> trans_search_array =
        ParallelHashArray<unsigned int>(nonunique_trans_hash);

    // write_device_vector("adjacent_trans_hash", adjacent_trans_hash);
    // write_device_vector("transCount", transCount);

    thrust::device_vector<std::size_t> *transIndices =
        trans_search_array.GetIndices(adjacent_trans_hash);

    thrust::device_vector<float> *vote_counts_out =
        new thrust::device_vector<float>(this->weightedVoteCounts.size());
    blocks = std::min(((int)(this->votes->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    rot_clustering_kernel<<<blocks,BLOCK_SIZE>>>
        (thrust::raw_pointer_cast(transformation_trans.data()),
         thrust::raw_pointer_cast(transformation_rots.data()),
         thrust::raw_pointer_cast(this->weightedVoteCounts.data()),
         thrust::raw_pointer_cast(adjacent_trans_hash.data()),
         RAW_PTR(transIndices), RAW_PTR(trans_search_array.GetHashkeys()),
         RAW_PTR(trans_search_array.GetCounts()),
         RAW_PTR(trans_search_array.GetFirstHashkeyIndices()),
         RAW_PTR(trans_search_array.GetHashkeyToDataMap()),
         thrust::raw_pointer_cast(vote_counts_out->data()),
         this->votes->size());
    delete transIndices;

    return vote_counts_out;
}

float Model::ScorePose(const float *weights, Eigen::Matrix4f truth,
                       pcl::PointCloud<pcl::PointNormal> scene){
    thrust::device_ptr<const float> weights_dev_ptr =
        thrust::device_pointer_cast<const float>(weights);
    thrust::device_vector<float> weight_vec =
        thrust::device_vector<float>(weights_dev_ptr, weights_dev_ptr + this->numPoints());

    this->weightedVoteCounts =
    // thrust::device_vector<float> weighted_vote_counts =
        ComputeWeightedVoteCounts(*(this->votes),
                                  *(this->voteCounts),
                                  weight_vec);

    // TODO: don't re-cluster transformations each time
    thrust::device_vector<float> *dev_vote_counts_out = ClusterTransformations();
    thrust::host_vector<float> vote_counts_out(*dev_vote_counts_out);
    int max_idx = std::distance(vote_counts_out.begin(),
                                std::max_element(vote_counts_out.begin(),
                                                 vote_counts_out.end()));
    // // TODO: replace sort with find max
    // thrust::sort_by_key(vote_counts_out->begin(),
    //                     vote_counts_out->end(),
    //                     thrust::device_ptr<struct float16>((struct float16 *) thrust::raw_pointer_cast(this->transformations.data())),
    //                     thrust::greater<unsigned int>());
    delete dev_vote_counts_out;

    thrust::host_vector<float> transformations =
        thrust::host_vector<float>(this->getTransformations());

    Eigen::Matrix4f T;
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++){
            T(i,j) = transformations[16*max_idx+i*4+j];
        }
    }

    float3 T_trans = {T(0,3), T(1,3), T(2,3)};
    float3 truth_trans = {truth(0,3), truth(1,3), truth(2,3)};
    float3 trans_diff = truth_trans - T_trans;

    Eigen::AngleAxisf T_rot, truth_rot;
    T_rot.fromRotationMatrix(T.block<3,3>(0,0));
    truth_rot.fromRotationMatrix(truth.block<3,3>(0,0));
    Eigen::AngleAxisf rotation_diff_mat(T_rot.inverse() * truth_rot);
    /* DEBUG */
    fprintf(stderr, "trans, rot: %f, %f\n", norm(trans_diff), fabsf(rotation_diff_mat.angle()));
    /* DEBUG */
    float score = norm(trans_diff)/D_DIST + fabsf(rotation_diff_mat.angle())/D_ANGLE0;
    // float score = 1 + fabsf(rotation_diff_mat.angle());
    return score;
}

thrust::device_vector<float> Model::OptimizeWeights(Scene *empty_scene,
                                                    int num_iterations){
    MODEL_OBJ = this;

    float3 t;
    float4 r;

    thrust::host_vector<float> avg_weight_vec(this->cloud_ptr->size());

    for(int i = 0; i < num_iterations; i++){
        t = (float3){0, 0, 0};
        r = (float4){0, 0, 0, 0};
        pcl::PointCloud<pcl::PointNormal> *new_scene_cloud =
            new pcl::PointCloud<pcl::PointNormal>();
        TRUTH = GenerateSceneWithModel(*this->cloud_ptr, *empty_scene->cloud_ptr, t, r,
                                       *new_scene_cloud);
        Scene *new_scene = new Scene(new_scene_cloud);
        SCENE_OBJ = new_scene;

        ComputeUniqueVotes(new_scene);
        ComputeTransformations(new_scene);

        thrust::host_vector<float> weights(optimize_weights(this->cloud_ptr->size()));
        for(int j = 0; j < this->cloud_ptr->size(); j++){
            fprintf(stderr, "weights[%d]: %f\n", j, weights[j]);
            avg_weight_vec[j] = (avg_weight_vec[j]*(i) + weights[j])/(i+1);
            fprintf(stderr, "avg_weights[%d]: %f\n", j, avg_weight_vec[j]);
        }
        // thrust::device_ptr<float> weights_dev_ptr =
        //     thrust::device_pointer_cast<float>(weights);
        // thrust::device_vector<float> weight_vec =
        //     thrust::device_vector<float>(weights,
        //                                  weights+this->cloud_ptr->size());

        delete new_scene_cloud;
        delete new_scene;
    }
    for(int k = 0; k < this->cloud_ptr->size(); k++){
        fprintf(stderr, "avg_weights[%d]: %f\n", k, avg_weight_vec[k]);
        fprintf(stdout, "avg_weights[%d]: %f\n", k, avg_weight_vec[k]);
    }
    return avg_weight_vec;
}

void Model::ppf_lookup(Scene *scene){

    #ifdef DEBUG
        cudaEvent_t start, stop;
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start, 0));
    #endif

    // Steps 1-3
    // launch voting kernel instance for each scene reference point
    ComputeUniqueVotes(scene);
    ComputeTransformations(scene);
    /* DEBUG */
    fprintf(stderr, "votes_size: %d\n", this->votes->size());
    /* DEBUG */

    this->weightedVoteCounts =
        ComputeWeightedVoteCounts(*(this->votes),
                                  *(this->voteCounts),
                                  this->modelPointVoteWeights);

    this->vote_counts_out = ClusterTransformations();

    thrust::sort_by_key(this->vote_counts_out->begin(),
                        this->vote_counts_out->end(),
                        thrust::device_ptr<struct float16>((struct float16 *) thrust::raw_pointer_cast(this->transformations.data())),
                        thrust::greater<unsigned int>());


    // thrust::device_vector<unsigned int> *uniqueSceneRefPts =
    //     new thrust::device_vector<unsigned int>(this->votes->size());
    // this->maxval = new thrust::device_vector<unsigned int>(this->votes->size());
    // thrust::device_vector<unsigned int> *maxModelAngleCode =
    //     new thrust::device_vector<unsigned int>(this->votes->size());

    // thrust::reduce_by_key
    //     (// key input: step function that increments for every row
    //      thrust::make_transform_iterator(votes->begin()+1, high_32_bits()),
    //      thrust::make_transform_iterator(votes->end(), high_32_bits()),
    //      // value input: (value, index) tuple
    //      thrust::make_zip_iterator(thrust::make_tuple(voteCounts->begin()+1,
    //                                                   thrust::make_transform_iterator(votes->begin()+1,
    //                                                                                   low_32_bits()))),
    //      uniqueSceneRefPts->begin(),
    //      thrust::make_zip_iterator(thrust::make_tuple(this->maxval->begin(),
    //                                                   maxModelAngleCode->begin())),
    //      thrust::equal_to<unsigned int>(),
    //      // compare by first element of tuple
    //      thrust::maximum<thrust::tuple<unsigned int, unsigned int> >());



    // Step 8, 9
    // call trans_calc_kernel
    // this->transformations = new thrust::device_vector<float>(this->votes->size()*16);

    // blocks = std::min(((int)(this->votes->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    // trans_calc_kernel<<<blocks,BLOCK_SIZE>>>
    //     (RAW_PTR(uniqueSceneRefPts), RAW_PTR(maxModelAngleCode),
    //      RAW_PTR(this->modelPoints), RAW_PTR(this->modelNormals),
    //      RAW_PTR(scene->getModelPoints()), RAW_PTR(scene->getModelNormals()),
    //      RAW_PTR(this->transformations),
    //      this->votes->size());

    #ifdef DEBUG
        // end cuda timer
        HANDLE_ERROR(cudaEventRecord(stop, 0));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        float elapsedTime;
        HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
        fprintf(stderr, "Time to lookup model:  %3.1f ms\n", elapsedTime);
    #endif
}

void Model::accumulateVotes(){
    this->voteCodes = new thrust::device_vector<unsigned long>();
    this->voteCounts = new thrust::device_vector<unsigned int>();
    histogram(*(this->votes), *(this->voteCodes), *(this->voteCounts));
}

thrust::device_vector<float> Model::getTransformations(){
    return this->transformations;
}
