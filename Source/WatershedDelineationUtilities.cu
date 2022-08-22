#include "WatershedDelineationUtilities.h"


FlattenedMatrix<unsigned char> WatershedDelineationUtilities::flattenDirectionMatrixParallel(const FlowDirectionMatrix& directionMatrix)
{
  const int height = directionMatrix.height;
  const int width = directionMatrix.width;

  FlattenedMatrix<unsigned char> flattenedMatrix(height, width);

  #pragma omp parallel
  {
    #pragma omp for
    for (int row = 1; row <= height; ++row)
    {
      for (int col = 1; col <= width; ++col)
      {
        flattenedMatrix.value[(unsigned)((row - 1) * width + (col - 1))] = directionMatrix.value[row][col];
      }
    }
  }

  return flattenedMatrix;
}


BasinIndexMatrix WatershedDelineationUtilities::unflattenBasinMatrixParallel(const FlattenedMatrix<unsigned char>& basinArray)
{
  const int height = basinArray.height;
  const int width = basinArray.width;

  BasinIndexMatrix basinMatrix(height, width);

  #pragma omp parallel
  {
    #pragma omp for
    for (int row = 1; row <= height; ++row)
    {
      for (int col = 1; col <= width; ++col)
      {
        basinMatrix.value[row][col] = basinArray.value[(unsigned)((row - 1) * width + (col - 1))];
      }
    }
  }

  return basinMatrix;
}


void WatershedDelineationUtilities::removeOutletDirection(FlattenedMatrix<unsigned char>& directionArray, const std::vector<CellMarker<unsigned char>>& outlet)
{
  const int width = directionArray.width;
  const int outletCells = outlet.size();

  for (int i = 0; i < outletCells; ++i)
  {
    directionArray.value[(unsigned)((outlet[i].row - 1) * width + (outlet[i].col - 1))] = DIRECTION_NONE;
  }
}


unsigned int* WatershedDelineationUtilities::sendOutletLocationsToGpu(const std::vector<CellMarker<unsigned char>>& outlet, int width)
{
  const int outletCells = outlet.size();
  std::vector<unsigned int> outletLocation(outletCells);

  for (int i = 0; i < outletCells; ++i)
  {
    outletLocation[i] = (outlet[i].row - 1) * width + (outlet[i].col - 1);
  }

  unsigned int* gpuOutletLocation;
  cudaMalloc(&gpuOutletLocation, outletCells * sizeof(unsigned int));
  cudaMemcpy(gpuOutletLocation, outletLocation.data(), outletCells * sizeof(unsigned int), cudaMemcpyHostToDevice);

  return gpuOutletLocation;
}


unsigned char* WatershedDelineationUtilities::sendOutletLabelsToGpu(const std::vector<CellMarker<unsigned char>>& outlet)
{
  const int outletCells = outlet.size();
  std::vector<unsigned char> outletLabel(outletCells);

  for (int i = 0; i < outletCells; ++i)
  {
    outletLabel[i] = outlet[i].label;
  }

  unsigned char* gpuOutletLabel;
  cudaMalloc(&gpuOutletLabel, outletCells * sizeof(unsigned char));
  cudaMemcpy(gpuOutletLabel, outletLabel.data(), outletCells * sizeof(unsigned char), cudaMemcpyHostToDevice);

  return gpuOutletLabel;
}


__global__ void clearBasinArrayKernel(unsigned char* basinArray, unsigned int size)
{
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size)
  {
    basinArray[index] = BASIN_NONE;
  }
}


__global__ void initializeBasinArrayKernel(unsigned char* basinArray, unsigned int* outletLocation, unsigned char* outletLabel, int outletCells)
{
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < outletCells)
  {
    basinArray[outletLocation[index]] = outletLabel[index];
  }
}


__global__ void directionToTargetKernel(unsigned char* directionArray, unsigned int* targetArray, int height, int width)
{
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < height * width)
  {
    int row = index / width;
    int col = index % width;

    switch (directionArray[index])
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

    targetArray[index] = ((row >= 0) && (row < height) && (col >= 0) && (col < width)) ? row * width + col : index;
  }
}


__global__ void targetToBasinKernel(unsigned int* targetArray, unsigned char* basinArray, unsigned int size)
{
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size)
  {
    basinArray[index] = basinArray[targetArray[index]];
  }
}
