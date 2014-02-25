#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>

#include "model.h"
#include "impl/ppf_utils.hpp"
#include "kernel.h"
#include "book.h"

Model::Model(thrust::host_vector<float3> *points, thrust::host_vector<float3> *normals, int n){
    this->initPPFs(points, normals, n);

    // for each ppf, compute a 32-bit hash and concatenate it with
    // a 32-bit int representing the index of the ppf in d_ppfs
    thrust::device_vector<unsigned long> *d_codes =
        new thrust::device_vector<unsigned long>(this->modelPPFs->size());

    ppf_encode_kernel<<<this->modelPPFs->size()/BLOCK_SIZE,BLOCK_SIZE>>>(RAW_PTR(this->modelPPFs),
                                                                         RAW_PTR(d_codes),
                                                                         this->modelPPFs->size());
    thrust::sort(d_codes->begin(), d_codes->end());

    // split codes into array of hashKeys (high 32 bits) and
    // key2ppfMap, the associated indices (low 32 bits)
    this->key2ppfMap = new thrust::device_vector<unsigned int>(this->modelPPFs->size());
    thrust::device_vector<unsigned int> *hashKeys_old =
        new thrust::device_vector<unsigned int>(this->modelPPFs->size());

    ppf_decode_kernel<<<this->modelPPFs->size()/BLOCK_SIZE,BLOCK_SIZE>>>(RAW_PTR(d_codes),
                                                                          RAW_PTR(this->key2ppfMap),
                                                                          RAW_PTR(hashKeys_old),
                                                                          this->modelPPFs->size());
    delete d_codes;

    this->hashKeys = new thrust::device_vector<unsigned int>();
    this->ppfCount = new thrust::device_vector<unsigned int>();
    histogram_destructive(*hashKeys_old, *(this->hashKeys), *(this->ppfCount));
    delete hashKeys_old;

    // create list of beginning indices of blocks of ppfs having equal hashes
    this->firstPPFIndex = new thrust::device_vector<unsigned int>(this->hashKeys->size());

    thrust::exclusive_scan(this->ppfCount->begin(),
                           this->ppfCount->end(),
                           this->firstPPFIndex->begin());

    this->votes = NULL;
    this->voteCodes = NULL;
    this->voteCounts = NULL;
}

Model::~Model(){
    delete this->ppfCount;
    delete this->firstPPFIndex;
    delete this->key2ppfMap;
    if (this->votes != NULL) delete this->votes;
    if (this->voteCodes != NULL) delete this->voteCodes;
    if (this->voteCounts != NULL) delete this->voteCounts;
}

// TODO: finish
void Model::ppf_lookup(Scene *scene){

    // find possible starting indices of blocks matching Model hashKeys
    thrust::device_vector<unsigned int> *sceneIndices =
        new thrust::device_vector<unsigned int>(scene->getModelPPFs()->size());
    thrust::lower_bound(this->hashKeys->begin(),
                        this->hashKeys->end(),
                        scene->getHashKeys()->begin(),
                        scene->getHashKeys()->end(),
                        sceneIndices->begin());

    thrust::device_vector<unsigned int> *found_ppf_starts =
        new thrust::device_vector<unsigned int>(scene->getModelPPFs()->size());
    thrust::device_vector<unsigned int> *found_ppf_count =
        new thrust::device_vector<unsigned int>(scene->getModelPPFs()->size());

    // launch voting kernel instance for each scene reference pointt
    this->votes = new thrust::device_vector<unsigned long>(scene->getModelPPFs()->size());
    ppf_vote_kernel<<<n/BLOCK_SIZE,BLOCK_SIZE>>>(RAW_PTR(scene->getHashKeys()), RAW_PTR(sceneIndices),
                                                 RAW_PTR(this->hashKeys), RAW_PTR(this->ppfCount),
                                                 RAW_PTR(this->firstPPFIndex), RAW_PTR(this->key2ppfMap),
                                                 RAW_PTR(this->modelPoints), RAW_PTR(this->modelNormals),
                                                 this->n, RAW_PTR(scene->getModelPoints()),
                                                 RAW_PTR(scene->getModelNormals()), scene->numPoints(),
                                                 RAW_PTR(this->votes), scene->numPoints());
    this->accumulateVotes();
}

void Model::accumulateVotes(){
    thrust::sort(this->votes->begin(), this->votes->end());
    this->voteCodes = new thrust::device_vector<unsigned long>();
    this->voteCounts = new thrust::device_vector<unsigned int>();
    histogram_destructive(*(this->votes), *(this->voteCodes), *(this->voteCounts));
}
