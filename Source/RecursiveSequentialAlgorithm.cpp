#include "RecursiveSequentialAlgorithm.h"


void RecursiveSequentialAlgorithm::addToBasin(const FlowDirectionMatrix& reversalDirectionMatrix, int row, int col, unsigned char index, BasinIndexMatrix& basinMatrix)
{
  if (basinMatrix.value[row][col] == BASIN_NONE)
  {
    basinMatrix.value[row][col] = index;

    if (reversalDirectionMatrix.value[row][col] & DIRECTION_RIGHT)
      addToBasin(reversalDirectionMatrix, row    , col + 1, index, basinMatrix);

    if (reversalDirectionMatrix.value[row][col] & DIRECTION_DOWN_RIGHT)
      addToBasin(reversalDirectionMatrix, row + 1, col + 1, index, basinMatrix);

    if (reversalDirectionMatrix.value[row][col] & DIRECTION_DOWN)
      addToBasin(reversalDirectionMatrix, row + 1, col    , index, basinMatrix);

    if (reversalDirectionMatrix.value[row][col] & DIRECTION_DOWN_LEFT)
      addToBasin(reversalDirectionMatrix, row + 1, col - 1, index, basinMatrix);

    if (reversalDirectionMatrix.value[row][col] & DIRECTION_LEFT)
      addToBasin(reversalDirectionMatrix, row    , col - 1, index, basinMatrix);

    if (reversalDirectionMatrix.value[row][col] & DIRECTION_UP_LEFT)
      addToBasin(reversalDirectionMatrix, row - 1, col - 1, index, basinMatrix);

    if (reversalDirectionMatrix.value[row][col] & DIRECTION_UP)
      addToBasin(reversalDirectionMatrix, row - 1, col    , index, basinMatrix);

    if (reversalDirectionMatrix.value[row][col] & DIRECTION_UP_RIGHT)
      addToBasin(reversalDirectionMatrix, row - 1, col + 1, index, basinMatrix);
  }
}


BasinIndexMatrix RecursiveSequentialAlgorithm::execute(const FlowDirectionMatrix& directionMatrix, const std::vector<CellMarker<unsigned char>>& outlet)
{
  const int height = directionMatrix.height;
  const int width = directionMatrix.width;
  const int outletCells = outlet.size();

  FlowDirectionMatrix reversalDirectionMatrix = FlowDirectionReverser::reverse(directionMatrix);
  BasinIndexMatrix basinMatrix(height, width, BASIN_NONE);

  for (int i = 0; i < outletCells; ++i)
  {
    basinMatrix.value[outlet[i].row][outlet[i].col] = outlet[i].label;
  }

  for (int i = 0; i < outletCells; ++i)
  {
    basinMatrix.value[outlet[i].row][outlet[i].col] = BASIN_NONE;
    addToBasin(reversalDirectionMatrix, outlet[i].row, outlet[i].col, outlet[i].label, basinMatrix);
  }

  return basinMatrix;
}
