#include <Rmath.h>
#include <R.h>

#include <vector>

#include "loess.h"

using namespace arma;
using std::vector;

typedef union {void *p; DL_FUNC fn;} fn_ptr;

DL_FUNC R_ExternalPtrAddrFn(SEXP s){
   fn_ptr tmp;
   tmp.p =  EXTPTR_PTR(s);
   return tmp.fn;
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

  static void (*loess_raw)(double *y, double *x, double *weights, double *robust, Sint *d,
                           Sint *n, double *span, Sint *degree, Sint *nonparametric,
                           Sint *drop_square, Sint *sum_drop_sqr, double *cell,
                           char **surf_stat, double *surface, Sint *parameter,
                           Sint *a, double *xi, double *vert, double *vval, double *diagonal,
                           double *trL, double *one_delta, double *two_delta, Sint *setLf) = NULL;
  if (!loess_raw) {
    Rcpp::Function getNativeSymbolInfo("getNativeSymbolInfo");
    Rcpp::List nativeSymbolInfo = getNativeSymbolInfo("loess_raw");
    SEXP xp = nativeSymbolInfo["address"];
    DL_FUNC loess_raw_p = R_ExternalPtrAddrFn( xp ) ;
    
    loess_raw = (void(*)(double *, double *, double *, double *, Sint *,
                         Sint *, double *, Sint *, Sint *,
                         Sint *, Sint *, double *,
                         char **, double *, Sint *,
                         Sint *, double *, double *, double *, double *,
                         double *, double *, double *, Sint *)) loess_raw_p;
  }

  (*loess_raw)(const_cast<double *>(y.begin()),
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
