#ifndef PATH_REDUCTION_GPU_ALGORITHM_H
#define PATH_REDUCTION_GPU_ALGORITHM_H


#include "IWatershedDelineationAlgorithm.h"
#include "WatershedDelineationUtilities.h"
#include <cuda_runtime_api.h>
#include <cuda.h>


class PathReductionGpuAlgorithm: public IWatershedDelineationAlgorithm
{
  private:
    const int blockSize;

  public:
    PathReductionGpuAlgorithm(int blockSize);
    BasinIndexMatrix execute(const FlowDirectionMatrix& directionMatrix, const std::vector<CellMarker<unsigned char>>& outlet);
};


__global__ void pathReductionKernel(unsigned int* targetArray, unsigned int size, bool* changes);


#endif
