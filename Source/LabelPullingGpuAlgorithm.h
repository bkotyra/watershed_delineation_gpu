#ifndef LABEL_PULLING_GPU_ALGORITHM_H
#define LABEL_PULLING_GPU_ALGORITHM_H


#include "IWatershedDelineationAlgorithm.h"
#include "WatershedDelineationUtilities.h"
#include <cuda_runtime_api.h>
#include <cuda.h>


class LabelPullingGpuAlgorithm: public IWatershedDelineationAlgorithm
{
  private:
    const int blockSize;

  public:
    LabelPullingGpuAlgorithm(int blockSize);
    BasinIndexMatrix execute(const FlowDirectionMatrix& directionMatrix, const std::vector<CellMarker<unsigned char>>& outlet);
};


__global__ void labelPullingKernel(unsigned int* targetArray, unsigned char* basinArray, unsigned int size, bool* changes);


#endif
