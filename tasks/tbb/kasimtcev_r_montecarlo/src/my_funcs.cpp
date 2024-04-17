// Copyright 2024 Kasimtcev Roman

#include <cmath>

#include "tbb/kasimtcev_r_montecarlo/include/my_funcs.hpp"

double flinear(double x, double y) { return x + y; }
double fsinxsiny(double x, double y) { return sin(x) + sin(y); }
double fcosxcosy(double x, double y) { return cos(x) + cos(y); }
double fxy(double x, double y) { return x * y; }
double fxyy(double x, double y) { return x * y * y; }