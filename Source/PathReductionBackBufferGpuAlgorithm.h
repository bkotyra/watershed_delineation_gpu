#ifndef PATH_REDUCTION_BACK_BUFFER_GPU_ALGORITHM_H
#define PATH_REDUCTION_BACK_BUFFER_GPU_ALGORITHM_H


#include "IWatershedDelineationAlgorithm.h"
#include "WatershedDelineationUtilities.h"
#include <cuda_runtime_api.h>
#include <cuda.h>


class PathReductionBackBufferGpuAlgorithm: public IWatershedDelineationAlgorithm
{
  private:
    const int blockSize;

  public:
    PathReductionBackBufferGpuAlgorithm(int blockSize);
    BasinIndexMatrix execute(const FlowDirectionMatrix& directionMatrix, const std::vector<CellMarker<unsigned char>>& outlet);
};


__global__ void pathReductionKernel(unsigned int* targetReadArray, unsigned int* targetWriteArray, unsigned int size, bool* changes);


#endif
