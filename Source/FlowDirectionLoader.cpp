#include "FlowDirectionLoader.h"


const int FlowDirectionLoader::binaryDimensionSize = 4;
const int FlowDirectionLoader::binaryValueSize = 1;


FlowDirectionMatrix FlowDirectionLoader::loadGdal(std::string filename, int bandIndex)
{
  GDALAllRegister();

  GDALDataset* dataset = (GDALDataset*) GDALOpen(filename.c_str(), GA_ReadOnly);
  GDALRasterBand* band = dataset->GetRasterBand(bandIndex);

  const int height = band->GetYSize();
  const int width = band->GetXSize();

  FlowDirectionMatrix directionMatrix(height, width);

  for (int i = 0; i < height; ++i)
  {
    CPLErr errorCode = band->RasterIO(GF_Read, 0, i, width, 1, directionMatrix.value[i + 1] + 1, width, 1, GDT_Byte, 0, 0);
  }

  int noDataValueExists;
  double noDataValue = band->GetNoDataValue(&noDataValueExists);

  if (noDataValueExists == 1)
  {
    for (int row = 1; row <= height; ++row)
    {
      for (int col = 1; col <= width; ++col)
      {
        if (directionMatrix.value[row][col] == noDataValue)
        {
          directionMatrix.value[row][col] = DIRECTION_NONE;
        }
      }
    }
  }

  GDALClose(dataset);

  return directionMatrix;
}


void FlowDirectionLoader::saveGdal(std::string filename, const FlowDirectionMatrix& directionMatrix, int bandIndex)
{
  const int height = directionMatrix.height;
  const int width = directionMatrix.width;

  GDALAllRegister();
  GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");

  GDALDataset* dataset = driver->Create(filename.c_str(), width, height, 1, GDT_Byte, NULL);
  GDALRasterBand* band = dataset->GetRasterBand(bandIndex);

  for (int i = 0; i < height; ++i)
  {
    CPLErr errorCode = band->RasterIO(GF_Write, 0, i, width, 1, directionMatrix.value[i + 1] + 1, width, 1, GDT_Byte, 0, 0);
  }

  GDALClose(dataset);
}


FlowDirectionMatrix FlowDirectionLoader::loadBinary(std::string filename)
{
  std::fstream file(filename, std::ios::in | std::ios::binary);

  int height;
  int width;

  file.read((char*) &height, binaryDimensionSize);
  file.read((char*) &width, binaryDimensionSize);

  FlowDirectionMatrix directionMatrix(height, width);

  for (int i = 0; i < height; ++i)
  {
    file.read((char*) (directionMatrix.value[i + 1] + 1), width * binaryValueSize);
  }

  file.close();

  return directionMatrix;
}


void FlowDirectionLoader::saveBinary(std::string filename, const FlowDirectionMatrix& directionMatrix)
{
  const int height = directionMatrix.height;
  const int width = directionMatrix.width;

  std::fstream file(filename, std::ios::out | std::ios::binary);

  file.write((char*) &height, binaryDimensionSize);
  file.write((char*) &width, binaryDimensionSize);

  for (int i = 0; i < height; ++i)
  {
    file.write((char*) (directionMatrix.value[i + 1] + 1), width * binaryValueSize);
  }

  file.close();
}
