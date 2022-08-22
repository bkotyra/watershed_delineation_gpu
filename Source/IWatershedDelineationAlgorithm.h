#ifndef I_WATERSHED_DELINEATION_ALGORITHM_H
#define I_WATERSHED_DELINEATION_ALGORITHM_H


#include "FlowDirectionMatrix.h"
#include "BasinIndexMatrix.h"
#include "CellMarker.h"
#include <vector>


class IWatershedDelineationAlgorithm
{
  public:
    virtual ~IWatershedDelineationAlgorithm() = 0;
    virtual BasinIndexMatrix execute(const FlowDirectionMatrix& directionMatrix, const std::vector<CellMarker<unsigned char>>& outlet) = 0;
};


#endif
