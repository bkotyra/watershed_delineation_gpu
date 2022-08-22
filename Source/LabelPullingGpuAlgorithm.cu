#include "LabelPullingGpuAlgorithm.h"


LabelPullingGpuAlgorithm::LabelPullingGpuAlgorithm(int blockSize):
  blockSize(blockSize)
{
}


BasinIndexMatrix LabelPullingGpuAlgorithm::execute(const FlowDirectionMatrix& directionMatrix, const std::vector<CellMarker<unsigned char>>& outlet)
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
  cudaMemcpy(gpuTransferArray, transferArray.value, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

  directionToTargetKernel<<<blocks, blockSize>>>(gpuTransferArray, gpuTargetArray, height, width);
  clearBasinArrayKernel<<<blocks, blockSize>>>(gpuTransferArray, size);
  initializeBasinArrayKernel<<<outletBlocks, blockSize>>>(gpuTransferArray, gpuOutletLocation, gpuOutletLabel, outletCells);

  bool changes;

  do
  {
    changes = false;
    cudaMemcpy(gpuChanges, &changes, sizeof(bool), cudaMemcpyHostToDevice);
    labelPullingKernel<<<blocks, blockSize>>>(gpuTargetArray, gpuTransferArray, size, gpuChanges);
    cudaMemcpy(&changes, gpuChanges, sizeof(bool), cudaMemcpyDeviceToHost);
  }
  while (changes);

  cudaMemcpy(transferArray.value, gpuTransferArray, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(gpuOutletLocation);
  cudaFree(gpuOutletLabel);
  cudaFree(gpuTransferArray);
  cudaFree(gpuTargetArray);
  cudaFree(gpuChanges);

  return WatershedDelineationUtilities::unflattenBasinMatrixParallel(transferArray);
}


__global__ void labelPullingKernel(unsigned int* targetArray, unsigned char* basinArray, unsigned int size, bool* changes)
{
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size)
  {
    if ((basinArray[index] == BASIN_NONE) && (basinArray[targetArray[index]] != BASIN_NONE))
    {
      basinArray[index] = basinArray[targetArray[index]];
      *changes = true;
    }
  }
}
