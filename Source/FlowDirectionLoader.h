#ifndef FLOW_DIRECTION_LOADER_H
#define FLOW_DIRECTION_LOADER_H


#include "FlowDirectionMatrix.h"
#include "GdalMatrixLoader.h"


class FlowDirectionLoader
{
  public:
    static FlowDirectionMatrix loadGdal(std::string filename, int bandIndex = 1);
    static void saveGdal(std::string filename, const FlowDirectionMatrix& directionMatrix);
};


#endif
