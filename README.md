This repository is part of the work presented in [High-performance watershed delineation algorithm for GPU using CUDA and OpenMP](https://doi.org/10.1016/j.envsoft.2022.105613).

---

Probably the simplest way to build the measurement application (using the Nvidia CUDA Compiler):

```
nvcc -std=c++11 -O3 -Xcompiler -fopenmp *.cpp *.cu -lgdal -o time_measurement
```

[GDAL](https://gdal.org/) is the only external dependency required (used here to load and save geospatial raster data).

---

The sample dataset provided in this repository contains a tiny (5x5 cells) synthetic case. It is intended to be illustrative and helpful in preparing input data for the measurement application. Please note that due to its small size, this dataset is **not** suitable for performance measurements.

![Sample Dataset](Sample%20Dataset/sample_dataset.png)
