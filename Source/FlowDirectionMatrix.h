#ifndef FLOW_DIRECTION_MATRIX_H
#define FLOW_DIRECTION_MATRIX_H


#include "FramedMatrix.h"
#include "FlowDirectionConstants.h"


struct FlowDirectionMatrix: public FramedMatrix<unsigned char>
{
  public:
    FlowDirectionMatrix(int height, int width);
    FlowDirectionMatrix(int height, int width, unsigned char fillValue);
};


#endif
