#include "PathReductionBackBufferGpuAlgorithm.h"


PathReductionBackBufferGpuAlgorithm::PathReductionBackBufferGpuAlgorithm(int blockSize):
  blockSize(blockSize)
{
}


BasinIndexMatrix PathReductionBackBufferGpuAlgorithm::execute(const FlowDirectionMatrix& directionMatrix, const std::vector<CellMarker<unsigned char>>& outlet)
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
  unsigned int* gpuTargetReadArray;
  unsigned int* gpuTargetWriteArray;
  bool* gpuChanges;

  cudaMalloc(&gpuTransferArray, size * sizeof(unsigned char));
  cudaMalloc(&gpuTargetReadArray, size * sizeof(unsigned int));
  cudaMalloc(&gpuTargetWriteArray, size * sizeof(unsigned int));
  cudaMalloc(&gpuChanges, sizeof(bool));

  FlattenedMatrix<unsigned char> transferArray = WatershedDelineationUtilities::flattenDirectionMatrixParallel(directionMatrix);
  WatershedDelineationUtilities::removeOutletDirection(transferArray, outlet);
  cudaMemcpy(gpuTransferArray, transferArray.value, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

  directionToTargetKernel<<<blocks, blockSize>>>(gpuTransferArray, gpuTargetReadArray, height, width);

  bool changes;

  do
  {
    changes = false;
    cudaMemcpy(gpuChanges, &changes, sizeof(bool), cudaMemcpyHostToDevice);
    pathReductionKernel<<<blocks, blockSize>>>(gpuTargetReadArray, gpuTargetWriteArray, size, gpuChanges);
    cudaMemcpy(&changes, gpuChanges, sizeof(bool), cudaMemcpyDeviceToHost);
    std::swap(gpuTargetReadArray, gpuTargetWriteArray);
  }
  while (changes);

  clearBasinArrayKernel<<<blocks, blockSize>>>(gpuTransferArray, size);
  initializeBasinArrayKernel<<<outletBlocks, blockSize>>>(gpuTransferArray, gpuOutletLocation, gpuOutletLabel, outletCells);
  targetToBasinKernel<<<blocks, blockSize>>>(gpuTargetReadArray, gpuTransferArray, size);

  cudaMemcpy(transferArray.value, gpuTransferArray, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(gpuOutletLocation);
  cudaFree(gpuOutletLabel);
  cudaFree(gpuTransferArray);
  cudaFree(gpuTargetReadArray);
  cudaFree(gpuTargetWriteArray);
  cudaFree(gpuChanges);

  return WatershedDelineationUtilities::unflattenBasinMatrixParallel(transferArray);
}


__global__ void pathReductionKernel(unsigned int* targetReadArray, unsigned int* targetWriteArray, unsigned int size, bool* changes)
{
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if ((index < size) && (targetWriteArray[index] != targetReadArray[targetReadArray[index]]))
  {
    targetWriteArray[index] = targetReadArray[targetReadArray[index]];
    *changes = true;
  }
}
