This repository is part of an ongoing publication.

---

Probably the simplest way to build the measurement application (using the Nvidia CUDA Compiler):

```
nvcc -std=c++11 -O3 -Xcompiler -fopenmp *.cpp *.cu -lgdal -o time_measurement
```

[GDAL](https://gdal.org/) is the only external dependency required (used here to load and save geospatial raster data).
