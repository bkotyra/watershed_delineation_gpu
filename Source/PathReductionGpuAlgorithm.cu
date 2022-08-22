#include "PathReductionGpuAlgorithm.h"


PathReductionGpuAlgorithm::PathReductionGpuAlgorithm(int blockSize):
  blockSize(blockSize)
{
}


BasinIndexMatrix PathReductionGpuAlgorithm::execute(const FlowDirectionMatrix& directionMatrix, const std::vector<CellMarker<unsigned char>>& outlet)
{
  const int height = directionMatrix.height;
  const int width = directionMatrix.width;
  const int outletCells = outlet.size();
  const unsigned int size = height * width;
  const int blocks = (size + blockSize - 1) / blockSize;
  const int outletBlocks = (outletCells + blockSize - 1) / blockSize;

  unsigned int* gpuOutletLocation = WatershedDelineationUtilities::sendOutletLocationsToGpu(outlet, width);
  unsigned char* gpuOutletLabel = WatershedDelineationUtilities::sendOutletLabelsToGpu(outlet);

  unsigned char* gpuTransferArray;
  unsigned int* gpuTargetArray;
  bool* gpuChanges;

  cudaMalloc(&gpuTransferArray, size * sizeof(unsigned char));
  cudaMalloc(&gpuTargetArray, size * sizeof(unsigned int));
  cudaMalloc(&gpuChanges, sizeof(bool));

  FlattenedMatrix<unsigned char> transferArray = WatershedDelineationUtilities::flattenDirectionMatrixParallel(directionMatrix);
  WatershedDelineationUtilities::removeOutletDirection(transferArray, outlet);
  cudaMemcpy(gpuTransferArray, transferArray.value, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

  directionToTargetKernel<<<blocks, blockSize>>>(gpuTransferArray, gpuTargetArray, height, width);

  bool changes;

  do
  {
    changes = false;
    cudaMemcpy(gpuChanges, &changes, sizeof(bool), cudaMemcpyHostToDevice);
    pathReductionKernel<<<blocks, blockSize>>>(gpuTargetArray, size, gpuChanges);
    cudaMemcpy(&changes, gpuChanges, sizeof(bool), cudaMemcpyDeviceToHost);
  }
  while (changes);

  clearBasinArrayKernel<<<blocks, blockSize>>>(gpuTransferArray, size);
  initializeBasinArrayKernel<<<outletBlocks, blockSize>>>(gpuTransferArray, gpuOutletLocation, gpuOutletLabel, outletCells);
  targetToBasinKernel<<<blocks, blockSize>>>(gpuTargetArray, gpuTransferArray, size);

  cudaMemcpy(transferArray.value, gpuTransferArray, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(gpuOutletLocation);
  cudaFree(gpuOutletLabel);
  cudaFree(gpuTransferArray);
  cudaFree(gpuTargetArray);
  cudaFree(gpuChanges);

  return WatershedDelineationUtilities::unflattenBasinMatrixParallel(transferArray);
}


__global__ void pathReductionKernel(unsigned int* targetArray, unsigned int size, bool* changes)
{
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if ((index < size) && (targetArray[index] != targetArray[targetArray[index]]))
  {
    targetArray[index] = targetArray[targetArray[index]];
    *changes = true;
  }
}
