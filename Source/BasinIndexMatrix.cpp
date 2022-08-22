#include "BasinIndexMatrix.h"


BasinIndexMatrix::BasinIndexMatrix(int height, int width):
  FramedMatrix<unsigned char>(height, width, BASIN_NONE)
{
}


BasinIndexMatrix::BasinIndexMatrix(int height, int width, unsigned char fillValue):
  FramedMatrix<unsigned char>(height, width, BASIN_NONE, fillValue)
{
}
