#include "loess.h"

using namespace arma;

extern "C" {
void loess_raw(double *y, double *x, double *weights, double *robust, Sint *d,
               Sint *n, double *span, Sint *degree, Sint *nonparametric,
               Sint *drop_square, Sint *sum_drop_sqr, double *cell,
               char **surf_stat, double *surface, Sint *parameter,
               Sint *a, double *xi, double *vert, double *vval, double *diagonal,
               double *trL, double *one_delta, double *two_delta, Sint *setLf);
}

vec loess_fit(vec const& y, vec const& x, vec const& weights, const double span_) {
  int n = y.n_elem;
  int d = 1;
  int max_kd = std::max(n, 200);

  int degree = 2;
  int nonparametric = 1;
  int drop_square = 2;
  int sum_drop_square = 0;
  double span = span_;
  double cell = span * 0.2;
  char* surf_stat = const_cast<char *>("interpolate/1.approx");
  
  vec fitted_values(n);
  int parameter[7];
  ivec a(max_kd);
  vec xi(max_kd);
  vec vert(2 * d);
  vec vval((d+1) * max_kd);
  vec diagonal(n);
  double trL, one_delta, two_delta;
  int setLf = 0;


  loess_raw(const_cast<double *>(y.begin()),
            const_cast<double *>(x.begin()),
               const_cast<double *>(weights.begin()),
               const_cast<double *>(weights.begin()),
               &d, &n, &span,
               &degree, &nonparametric,
               &drop_square, &sum_drop_square, &cell, &surf_stat,
               fitted_values.begin(), parameter, a.begin(), 
               xi.begin(), vert.begin(), vval.begin(),
               diagonal.begin(), &trL, &one_delta, &two_delta, &setLf);
  
  return fitted_values;
}
