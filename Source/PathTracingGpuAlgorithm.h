#ifndef PATH_TRACING_GPU_ALGORITHM_H
#define PATH_TRACING_GPU_ALGORITHM_H


#include "IWatershedDelineationAlgorithm.h"
#include "WatershedDelineationUtilities.h"
#include <cuda_runtime_api.h>
#include <cuda.h>


class PathTracingGpuAlgorithm: public IWatershedDelineationAlgorithm
{
  private:
    const int blockSize;

  public:
    PathTracingGpuAlgorithm(int blockSize);
    BasinIndexMatrix execute(const FlowDirectionMatrix& directionMatrix, const std::vector<CellMarker<unsigned char>>& outlet);
};


__global__ void pathTracingKernel(unsigned char* directionArray, unsigned char* basinArray, int height, int width);


#endif
