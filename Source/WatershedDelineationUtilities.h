#ifndef WATERSHED_DELINEATION_UTILITIES_H
#define WATERSHED_DELINEATION_UTILITIES_H


#include "FlowDirectionMatrix.h"
#include "BasinIndexMatrix.h"
#include "FlattenedMatrix.h"
#include "CellMarker.h"
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <omp.h>


class WatershedDelineationUtilities
{
  public:
    static FlattenedMatrix<unsigned char> flattenDirectionMatrixParallel(const FlowDirectionMatrix& directionMatrix);
    static BasinIndexMatrix unflattenBasinMatrixParallel(const FlattenedMatrix<unsigned char>& basinArray);
    static void removeOutletDirection(FlattenedMatrix<unsigned char>& directionArray, const std::vector<CellMarker<unsigned char>>& outlet);

    static unsigned int* sendOutletLocationsToGpu(const std::vector<CellMarker<unsigned char>>& outlet, int width);
    static unsigned char* sendOutletLabelsToGpu(const std::vector<CellMarker<unsigned char>>& outlet);
};


__global__ void clearBasinArrayKernel(unsigned char* basinArray, unsigned int size);
__global__ void initializeBasinArrayKernel(unsigned char* basinArray, unsigned int* outletLocation, unsigned char* outletLabel, int outletCells);
__global__ void directionToTargetKernel(unsigned char* directionArray, unsigned int* targetArray, int height, int width);
__global__ void targetToBasinKernel(unsigned int* targetArray, unsigned char* basinArray, unsigned int size);


#endif
