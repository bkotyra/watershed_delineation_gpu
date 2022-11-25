#include "BasinIndexLoader.h"


BasinIndexMatrix BasinIndexLoader::loadGdal(std::string filename, int bandIndex)
{
  return GdalMatrixLoader::load<BasinIndexMatrix, unsigned char>(filename, GDT_Byte, BASIN_NONE, bandIndex);
}


void BasinIndexLoader::saveGdal(std::string filename, const BasinIndexMatrix& basinIndexMatrix)
{
  GdalMatrixLoader::save<BasinIndexMatrix>(filename, basinIndexMatrix, GDT_Byte);
}
