#ifndef TRAIN_WEIGHTS_H
#define TRAIN_WEIGHTS_H

#include <PSO_Optimizer.h>
#include <SolutionSet.h>
#include <FitnessSet.h>
#include <vector>

#include "model.h"

extern Model *MODEL_OBJ;
extern Scene *SCENE_OBJ;
extern Eigen::Matrix4f TRUTH;

void myFitness(const CudaOptimize::SolutionSet *solset, CudaOptimize::FitnessSet *fitset,
               dim3 calculateFitnessGrid, dim3 calculateFitnessBlock);
std::vector<float> optimize_weights(int votes_size);


#endif /* TRAIN_WEIGHTS_H */
