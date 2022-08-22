#ifndef FLOW_DIRECTION_REVERSER_H
#define FLOW_DIRECTION_REVERSER_H


#include "FlowDirectionMatrix.h"
#include <omp.h>


class FlowDirectionReverser
{
  public:
    static FlowDirectionMatrix reverse(const FlowDirectionMatrix& directionMatrix);
    static FlowDirectionMatrix reverseParallel(const FlowDirectionMatrix& directionMatrix);
};


#endif
