// #include <PSO_Optimizer.h>
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>

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
      BOOST_LOG_TRIVIAL(info) << boost::format("score[%d]: %f") % i % fitnesses[i];
  }

  //write fitness values
  fitset->set(fitnesses);
}

std::vector<float> optimize_weights(int votes_size){
    BOOST_LOG_TRIVIAL(info) << boost::format("num_points: %d") % MODEL_OBJ->numPoints();
    // MODEL_OBJ->numPoints() must be <= 1024 (IOptimzer.cpp:83)
    CudaOptimize::DE_Optimizer p(&myFitness, MODEL_OBJ->numPoints(), 1, 1536);
    float2 bounds = {0.0, 8.0};
    p.setBounds(bounds);
    p.setMutation(CudaOptimize::DE_TARGET_TO_BEST);
    // p.setCrossover(CudaOptimize::DE_EXPONENTIAL);
    p.setF(0.5);
    p.setCR(0.9);
    p.setGenerations(10);
    // p.setVerbosity(logDEBUG4);
    p.optimize();
    float *myResults = p.getBestSolution();
    std::vector<float> optimal_weights(myResults, myResults + MODEL_OBJ->numPoints());
    return optimal_weights;
}
