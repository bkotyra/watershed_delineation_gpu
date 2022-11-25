#include "FlowDirectionLoader.h"


FlowDirectionMatrix FlowDirectionLoader::loadGdal(std::string filename, int bandIndex)
{
  return GdalMatrixLoader::load<FlowDirectionMatrix, unsigned char>(filename, GDT_Byte, DIRECTION_NONE, bandIndex);
}


void FlowDirectionLoader::saveGdal(std::string filename, const FlowDirectionMatrix& directionMatrix)
{
  GdalMatrixLoader::save<FlowDirectionMatrix>(filename, directionMatrix, GDT_Byte);
}
