#include "FlowDirectionMatrix.h"


FlowDirectionMatrix::FlowDirectionMatrix(int height, int width):
  FramedMatrix<unsigned char>(height, width, DIRECTION_NONE)
{
}


FlowDirectionMatrix::FlowDirectionMatrix(int height, int width, unsigned char fillValue):
  FramedMatrix<unsigned char>(height, width, DIRECTION_NONE, fillValue)
{
}
