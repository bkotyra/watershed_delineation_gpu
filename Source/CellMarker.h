#ifndef CELL_MARKER_H
#define CELL_MARKER_H


template <typename T>
struct CellMarker
{
  public:
    int row;
    int col;
    T label;

  public:
    CellMarker();
    CellMarker(int row, int col, T label);
};


template <typename T>
CellMarker<T>::CellMarker()
{
}


template <typename T>
CellMarker<T>::CellMarker(int row, int col, T label):
  row(row),
  col(col),
  label(label)
{
}


#endif
