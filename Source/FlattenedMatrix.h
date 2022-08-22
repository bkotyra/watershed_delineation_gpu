#ifndef FLATTENED_MATRIX_H
#define FLATTENED_MATRIX_H


template <typename T>
struct FlattenedMatrix
{
  public:
    const int height;
    const int width;

    T* value;

  public:
    FlattenedMatrix(int height, int width);
    ~FlattenedMatrix();
};


template <typename T>
FlattenedMatrix<T>::FlattenedMatrix(int height, int width):
  height(height),
  width(width)
{
  value = new T[(unsigned)(height * width)];
}


template <typename T>
FlattenedMatrix<T>::~FlattenedMatrix()
{
  delete[] value;
}


#endif
