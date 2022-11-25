#ifndef BASIN_INDEX_LOADER_H
#define BASIN_INDEX_LOADER_H


#include "BasinIndexMatrix.h"
#include "GdalMatrixLoader.h"


class BasinIndexLoader
{
  public:
    static BasinIndexMatrix loadGdal(std::string filename, int bandIndex = 1);
    static void saveGdal(std::string filename, const BasinIndexMatrix& basinIndexMatrix);
};


#endif
