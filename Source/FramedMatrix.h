#ifndef FRAMED_MATRIX_H
#define FRAMED_MATRIX_H


template <typename T>
struct FramedMatrix
{
  public:
    const int height;
    const int width;

    T** value;

  public:
    FramedMatrix(int height, int width);
    FramedMatrix(int height, int width, T frameValue);
    FramedMatrix(int height, int width, T frameValue, T fillValue);
    FramedMatrix(FramedMatrix&& source);
    ~FramedMatrix();

    bool operator==(const FramedMatrix<T>& second) const;
    bool operator!=(const FramedMatrix<T>& second) const;
};


template <typename T>
FramedMatrix<T>::FramedMatrix(int height, int width):
  height(height),
  width(width)
{
  value = new T*[height + 2];

  for (int row = 0; row < height + 2; ++row)
  {
    value[row] = new T[width + 2];
  }
}


template <typename T>
FramedMatrix<T>::FramedMatrix(int height, int width, T frameValue):
  height(height),
  width(width)
{
  value = new T*[height + 2];

  for (int row = 0; row < height + 2; ++row)
  {
    value[row] = new T[width + 2];

    value[row][0] = frameValue;
    value[row][width + 1] = frameValue;
  }

  for (int col = 1; col <= width; ++col)
  {
    value[0][col] = frameValue;
    value[height + 1][col] = frameValue;
  }
}


template <typename T>
FramedMatrix<T>::FramedMatrix(int height, int width, T frameValue, T fillValue):
  FramedMatrix<T>(height, width, frameValue)
{
  for (int row = 1; row <= height; ++row)
  {
    for (int col = 1; col <= width; ++col)
    {
      value[row][col] = fillValue;
    }
  }
}


template <typename T>
FramedMatrix<T>::FramedMatrix(FramedMatrix<T>&& source):
  height(source.height),
  width(source.width),
  value(source.value)
{
  source.value = nullptr;
}


template <typename T>
FramedMatrix<T>::~FramedMatrix()
{
  if (value != nullptr)
  {
    for (int row = 0; row < height + 2; ++row)
    {
      delete[] value[row];
    }

    delete[] value;
  }
}


template <typename T>
bool FramedMatrix<T>::operator==(const FramedMatrix<T>& second) const
{
  if ((height != second.height) || (width != second.width))
  {
    return false;
  }

  for (int row = 1; row <= height; ++row)
  {
    for (int col = 1; col <= width; ++col)
    {
      if (value[row][col] != second.value[row][col])
      {
        return false;
      }
    }
  }

  return true;
}


template <typename T>
bool FramedMatrix<T>::operator!=(const FramedMatrix<T>& second) const
{
  return !(*this == second);
}


#endif
