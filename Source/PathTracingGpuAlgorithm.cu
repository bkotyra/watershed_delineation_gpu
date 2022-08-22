#include "PathTracingGpuAlgorithm.h"


PathTracingGpuAlgorithm::PathTracingGpuAlgorithm(int blockSize):
  blockSize(blockSize)
{
}


BasinIndexMatrix PathTracingGpuAlgorithm::execute(const FlowDirectionMatrix& directionMatrix, const std::vector<CellMarker<unsigned char>>& outlet)
{
  const int height = directionMatrix.height;
  const int width = directionMatrix.width;
  const int outletCells = outlet.size();
  const unsigned int size = height * width;
  const int blocks = (size + blockSize - 1) / blockSize;
  const int outletBlocks = (outletCells + blockSize - 1) / blockSize;

  unsigned int* gpuOutletLocation = WatershedDelineationUtilities::sendOutletLocationsToGpu(outlet, width);
  unsigned char* gpuOutletLabel = WatershedDelineationUtilities::sendOutletLabelsToGpu(outlet);

  unsigned char* gpuDirectionArray;
  unsigned char* gpuBasinArray;

  cudaMalloc(&gpuDirectionArray, size * sizeof(unsigned char));
  cudaMalloc(&gpuBasinArray, size * sizeof(unsigned char));

  FlattenedMatrix<unsigned char> transferArray = WatershedDelineationUtilities::flattenDirectionMatrixParallel(directionMatrix);
  WatershedDelineationUtilities::removeOutletDirection(transferArray, outlet);
  cudaMemcpy(gpuDirectionArray, transferArray.value, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

  clearBasinArrayKernel<<<blocks, blockSize>>>(gpuBasinArray, size);
  initializeBasinArrayKernel<<<outletBlocks, blockSize>>>(gpuBasinArray, gpuOutletLocation, gpuOutletLabel, outletCells);
  pathTracingKernel<<<blocks, blockSize>>>(gpuDirectionArray, gpuBasinArray, height, width);

  cudaMemcpy(transferArray.value, gpuBasinArray, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(gpuOutletLocation);
  cudaFree(gpuOutletLabel);
  cudaFree(gpuDirectionArray);
  cudaFree(gpuBasinArray);

  return WatershedDelineationUtilities::unflattenBasinMatrixParallel(transferArray);
}


__global__ void pathTracingKernel(unsigned char* directionArray, unsigned char* basinArray, int height, int width)
{
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if ((index < height * width) && (basinArray[index] == BASIN_NONE))
  {
    int row = index / width;
    int col = index % width;

    while ((row >= 0) && (row < height) && (col >= 0) && (col < width) && (directionArray[row * width + col] != DIRECTION_NONE))
    {
      switch (directionArray[row * width + col])
      {
        case DIRECTION_RIGHT:             ++col; break;
        case DIRECTION_DOWN_RIGHT: ++row; ++col; break;
        case DIRECTION_DOWN:       ++row;        break;
        case DIRECTION_DOWN_LEFT:  ++row; --col; break;
        case DIRECTION_LEFT:              --col; break;
        case DIRECTION_UP_LEFT:    --row; --col; break;
        case DIRECTION_UP:         --row;        break;
        case DIRECTION_UP_RIGHT:   --row; ++col; break;
      }
    }

    if ((row >= 0) && (row < height) && (col >= 0) && (col < width))
    {
      basinArray[index] = basinArray[row * width + col];
    }
  }
}
