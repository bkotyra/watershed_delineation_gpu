#include "FlowDirectionLoader.h"
#include "BasinIndexLoader.h"
#include "RecursiveSequentialAlgorithm.h"
#include "PathTracingGpuAlgorithm.h"
#include "LabelPullingGpuAlgorithm.h"
#include "PathReductionGpuAlgorithm.h"
#include "PathReductionBackBufferGpuAlgorithm.h"
#include <chrono>
#include <fstream>
#include <iostream>


const int outletPrintLimit = 8;


std::string algorithmLabel(int algorithmIndex)
{
  switch (algorithmIndex)
  {
    case 1: return "recursive (sequential)";
    case 2: return "flow path tracing (GPU)";
    case 3: return "label pulling (GPU)";
    case 4: return "flow path reduction: single buffer (GPU)";
    case 5: return "flow path reduction: back buffer (GPU)";
    default: return "";
  }
}


IWatershedDelineationAlgorithm* createAlgorithm(int algorithmIndex, int algorithmParameter)
{
  switch (algorithmIndex)
  {
    case 1: return new RecursiveSequentialAlgorithm();
    case 2: return new PathTracingGpuAlgorithm(algorithmParameter);
    case 3: return new LabelPullingGpuAlgorithm(algorithmParameter);
    case 4: return new PathReductionGpuAlgorithm(algorithmParameter);
    case 5: return new PathReductionBackBufferGpuAlgorithm(algorithmParameter);
    default: return nullptr;
  }
}


void printUsage()
{
  std::cout << "required arguments:" << std::endl
            << " 1.  flow direction filename" << std::endl
            << " 2.  outlets filename (containing one-based 'row column label' triplets)" << std::endl
            << " 3.  algorithm index" << std::endl
            << "(4.) CUDA block size (only relevant for GPU implementations, 1024 by default)" << std::endl
            << "(5.) resulting raster filename (will be overwritten if exists)" << std::endl
            << std::endl
            << "available algorithms:" << std::endl;

  for (int i = 1; i <= 5; ++i)
  {
    std::cout << ' ' << i << ".  " << algorithmLabel(i) << std::endl;
  }
}


std::vector<CellMarker<unsigned char>> loadOutletLocations(std::string filename)
{
  std::vector<CellMarker<unsigned char>> outlets;

  std::fstream file(filename, std::ios::in);
  int row, col, label;

  while (file >> row >> col >> label)
  {
    outlets.push_back({row, col, (unsigned char)label});
  }

  file.close();
  return outlets;
}


void printOutlets(std::vector<CellMarker<unsigned char>>& outlets)
{
  const int outletsTotal = outlets.size();
  const int outletsToPrint = std::min(outletsTotal, outletPrintLimit);

  std::cout << "number of outlet locations: " << outletsTotal << std::endl;

  for (int i = 0; i < outletsToPrint; ++i)
  {
    std::cout << "- row " << outlets[i].row << ", column " << outlets[i].col << ", label " << (int)outlets[i].label << std::endl;
  }

  if (outletsToPrint < outletsTotal)
  {
    std::cout << "- ..." << std::endl;
  }
}


void executeMeasurement(std::string directionFilename, std::string outletsFilename, int algorithmIndex, int algorithmParameter, std::string resultsFilename)
{
  std::cout << "loading flow direction file (" << directionFilename << ")..." << std::endl;
  FlowDirectionMatrix directionMatrix = FlowDirectionLoader::loadGdal(directionFilename);
  std::cout << "flow direction data: " << directionMatrix.height << " rows, " << directionMatrix.width << " columns" << std::endl;

  std::cout << "loading outlets file (" << outletsFilename << ")..." << std::endl;
  std::vector<CellMarker<unsigned char>> outletLocations = loadOutletLocations(outletsFilename);
  printOutlets(outletLocations);

  std::cout << "GPU device synchronization..." << std::endl;
  cudaDeviceSynchronize();

  std::cout << "executing " << algorithmLabel(algorithmIndex) << " algorithm..." << std::endl;
  IWatershedDelineationAlgorithm* algorithm = createAlgorithm(algorithmIndex, algorithmParameter);

  auto stamp_begin = std::chrono::high_resolution_clock::now();
  BasinIndexMatrix basinMatrix = algorithm->execute(directionMatrix, outletLocations);
  auto stamp_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> stamp_diff = stamp_end - stamp_begin;

  std::cout << "execution time (ms): " << lround(stamp_diff.count()) << std::endl;

  if (!resultsFilename.empty())
  {
    std::cout << "saving results (" << resultsFilename << ")..." << std::endl;
    BasinIndexLoader::saveGdal(resultsFilename, basinMatrix);
  }
}


int main(int argc, char** argv)
{
  if (argc < 4)
  {
    printUsage();
  }

  else
  {
    const std::string directionFilename = argv[1];
    const std::string outletsFilename = argv[2];
    const int algorithmIndex = atoi(argv[3]);
    const int algorithmParameter = (argc > 4) ? atoi(argv[4]) : 1024;
    const std::string resultsFilename = (argc > 5) ? argv[5] : "";

    executeMeasurement(directionFilename, outletsFilename, algorithmIndex, algorithmParameter, resultsFilename);
  }
}
