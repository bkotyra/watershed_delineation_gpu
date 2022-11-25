#ifndef GDAL_MATRIX_LOADER_H
#define GDAL_MATRIX_LOADER_H


#include "gdal/gdal_priv.h"
#include "gdal/cpl_conv.h"


class GdalMatrixLoader
{
  public:
    template <class M, typename T> static M load(std::string filename, GDALDataType dataType, T noDataValue, int bandIndex = 1);
    template <class M> static void save(std::string filename, const M& matrix, GDALDataType dataType);
};


template <class M, typename T>
M GdalMatrixLoader::load(std::string filename, GDALDataType dataType, T noDataValue, int bandIndex)
{
  GDALAllRegister();

  GDALDataset* dataset = (GDALDataset*) GDALOpen(filename.c_str(), GA_ReadOnly);
  GDALRasterBand* band = dataset->GetRasterBand(bandIndex);

  const int height = band->GetYSize();
  const int width = band->GetXSize();

  M matrix(height, width);

  for (int i = 0; i < height; ++i)
  {
    CPLErr errorCode = band->RasterIO(GF_Read, 0, i, width, 1, matrix.value[i + 1] + 1, width, 1, dataType, 0, 0);
  }

  int fileNoDataExists;
  double fileNoDataValue = band->GetNoDataValue(&fileNoDataExists);

  if ((fileNoDataExists == 1) && (fileNoDataValue != noDataValue))
  {
    for (int row = 1; row <= height; ++row)
    {
      for (int col = 1; col <= width; ++col)
      {
        if (matrix.value[row][col] == fileNoDataValue)
        {
          matrix.value[row][col] = noDataValue;
        }
      }
    }
  }

  GDALClose(dataset);
  return matrix;
}


template <class M>
void GdalMatrixLoader::save(std::string filename, const M& matrix, GDALDataType dataType)
{
  GDALAllRegister();
  GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");

  const int height = matrix.height;
  const int width = matrix.width;

  GDALDataset* dataset = driver->Create(filename.c_str(), width, height, 1, dataType, NULL);
  GDALRasterBand* band = dataset->GetRasterBand(1);

  for (int i = 0; i < height; ++i)
  {
    CPLErr errorCode = band->RasterIO(GF_Write, 0, i, width, 1, matrix.value[i + 1] + 1, width, 1, dataType, 0, 0);
  }

  GDALClose(dataset);
}


#endif
