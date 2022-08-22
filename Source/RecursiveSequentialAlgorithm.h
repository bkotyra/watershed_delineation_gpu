#ifndef RECURSIVE_SEQUENTIAL_ALGORITHM_H
#define RECURSIVE_SEQUENTIAL_ALGORITHM_H


#include "IWatershedDelineationAlgorithm.h"
#include "FlowDirectionReverser.h"


class RecursiveSequentialAlgorithm: public IWatershedDelineationAlgorithm
{
  private:
    void addToBasin(const FlowDirectionMatrix& reversalDirectionMatrix, int row, int col, unsigned char index, BasinIndexMatrix& basinMatrix);

  public:
    BasinIndexMatrix execute(const FlowDirectionMatrix& directionMatrix, const std::vector<CellMarker<unsigned char>>& outlet);
};


#endif
