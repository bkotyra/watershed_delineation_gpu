#ifndef BASIN_INDEX_MATRIX_H
#define BASIN_INDEX_MATRIX_H


#include "FramedMatrix.h"
#include "BasinIndexConstants.h"


struct BasinIndexMatrix: public FramedMatrix<unsigned char>
{
  public:
    BasinIndexMatrix(int height, int width);
    BasinIndexMatrix(int height, int width, unsigned char fillValue);
};


#endif
