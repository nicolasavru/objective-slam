// #include <PSO_Optimizer.h>
#include <DE_Optimizer.h>
#include <SolutionSet.h>
#include <FitnessSet.h>
#include <vector>

#include "train_weights.h"


void myFitness(const CudaOptimize::SolutionSet *solset, CudaOptimize::FitnessSet *fitset,
               dim3 calculateFitnessGrid, dim3 calculateFitnessBlock)
{
  //create a temporary array that will contain fitnesses
  float *fitnesses = new float[solset->getSolutionNumber()];
  // float *weight_dev_ptr = solset->getDevicePositionsConst();

  for(int i = 0; i < solset->getSolutionNumber(); i++){
      fitnesses[i] = MODEL_OBJ->ScorePose(solset->getDevicePositionsConst(0, i),
                                          TRUTH, *SCENE_OBJ->cloud_ptr); //0 is the number of set
      /* DEBUG */
      fprintf(stderr, "score[%d]: %f\n", i, fitnesses[i]);
      /* DEBUG */
  }

  //write fitness values
  fitset->set(fitnesses);
}

std::vector<float> optimize_weights(int votes_size){
    /* DEBUG */
    fprintf(stderr, "num_points: %d\n", MODEL_OBJ->numPoints());
    /* DEBUG */
    // CudaOptimize::PSO_Optimizer p(&myFitness, MODEL_OBJ->numPoints(), 1, 64);
    CudaOptimize::DE_Optimizer p(&myFitness, MODEL_OBJ->numPoints(), 1, 1536);
    float2 bounds = {0.0, 8.0};
    p.setBounds(bounds);
    p.setMutation(CudaOptimize::DE_TARGET_TO_BEST);
    // p.setCrossover(CudaOptimize::DE_EXPONENTIAL);
    p.setF(0.5);
    p.setCR(0.9);
    p.setGenerations(10);
    p.optimize();
    float *myResults = p.getBestSolution();
    /* DEBUG */
    fprintf(stderr, "myResults[0]: %f\n", myResults[0]);
    /* DEBUG */
    std::vector<float> optimal_weights(myResults, myResults + MODEL_OBJ->numPoints());
    return optimal_weights;
}
