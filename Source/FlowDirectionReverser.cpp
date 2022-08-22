#include "FlowDirectionReverser.h"


FlowDirectionMatrix FlowDirectionReverser::reverse(const FlowDirectionMatrix& directionMatrix)
{
  const int height = directionMatrix.height;
  const int width = directionMatrix.width;

  FlowDirectionMatrix reversalMatrix(height, width);

  for (int row = 1; row <= height; ++row)
  {
    for (int col = 1; col <= width; ++col)
    {
      reversalMatrix.value[row][col] = ((directionMatrix.value[row    ][col + 1] & DIRECTION_LEFT) >> 4) |
                                       ((directionMatrix.value[row + 1][col + 1] & DIRECTION_UP_LEFT) >> 4) |
                                       ((directionMatrix.value[row + 1][col    ] & DIRECTION_UP) >> 4) |
                                       ((directionMatrix.value[row + 1][col - 1] & DIRECTION_UP_RIGHT) >> 4) |
                                       ((directionMatrix.value[row    ][col - 1] & DIRECTION_RIGHT) << 4) |
                                       ((directionMatrix.value[row - 1][col - 1] & DIRECTION_DOWN_RIGHT) << 4) |
                                       ((directionMatrix.value[row - 1][col    ] & DIRECTION_DOWN) << 4) |
                                       ((directionMatrix.value[row - 1][col + 1] & DIRECTION_DOWN_LEFT) << 4);
    }
  }

  return reversalMatrix;
}


FlowDirectionMatrix FlowDirectionReverser::reverseParallel(const FlowDirectionMatrix& directionMatrix)
{
  const int height = directionMatrix.height;
  const int width = directionMatrix.width;

  FlowDirectionMatrix reversalMatrix(height, width);

  #pragma omp parallel
  {
    #pragma omp for
    for (int row = 1; row <= height; ++row)
    {
      for (int col = 1; col <= width; ++col)
      {
        reversalMatrix.value[row][col] = ((directionMatrix.value[row    ][col + 1] & DIRECTION_LEFT) >> 4) |
                                         ((directionMatrix.value[row + 1][col + 1] & DIRECTION_UP_LEFT) >> 4) |
                                         ((directionMatrix.value[row + 1][col    ] & DIRECTION_UP) >> 4) |
                                         ((directionMatrix.value[row + 1][col - 1] & DIRECTION_UP_RIGHT) >> 4) |
                                         ((directionMatrix.value[row    ][col - 1] & DIRECTION_RIGHT) << 4) |
                                         ((directionMatrix.value[row - 1][col - 1] & DIRECTION_DOWN_RIGHT) << 4) |
                                         ((directionMatrix.value[row - 1][col    ] & DIRECTION_DOWN) << 4) |
                                         ((directionMatrix.value[row - 1][col + 1] & DIRECTION_DOWN_LEFT) << 4);
      }
    }
  }

  return reversalMatrix;
}
