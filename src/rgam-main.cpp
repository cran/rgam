#include "rgam.h"
#include "rgam_poisson.h"
#include "rgam_binomial.h"
#include "loess.h"
#include "span_poisson.h"
#include "span_binomial.h"

using namespace Rcpp;
using namespace arma;

static List wrap(RgamResult result) {
  NumericVector prediction = wrap(result.prediction);
  prediction.attr( "dim" ) = R_NilValue;

  NumericVector eta = wrap(result.eta);
  eta.attr( "dim" ) = R_NilValue;

  NumericVector cv_result = wrap(result.cv_result);
  cv_result.attr( "dim" ) = R_NilValue;

  return List::create(_["fitted.values"] = prediction,
                      _["opt.alpha"] = result.alpha,
                      _["cv.results"] = cv_result,
                      _["smooth"] = result.s,
                      _["additive.predictors"] = eta,
                      _["iterations"] = result.iterations,
                      _["convergence.criterion"] = result.con,
                      _["converged"] = result.converged);
}

RcppExport SEXP rgam_p(SEXP x_in,
                       SEXP y_in,
                       SEXP epsilon_in,
                       SEXP max_iterations_in,
                       SEXP k_in,
                       SEXP span_type_in,
                       SEXP alphas_in,
                       SEXP s_in,
                       SEXP trace_in) {
  RNGScope scope;               // keeping R's RNG in sync
  const double epsilon = as<double>(epsilon_in);
  const int max_iterations = as<int>(max_iterations_in);
  const double k = as<double>(k_in);
  const bool trace = as<bool>(trace_in);

  NumericMatrix x_r(x_in);
  mat x(x_r.begin(), x_r.nrow(), x_r.ncol(), false);

  NumericVector y_r(y_in);
  vec y(y_r.begin(), y_r.size(), false);

  NumericVector alphas_r(alphas_in);
  vec alphas(alphas_r.begin(), alphas_r.size(), false);
  
  NumericMatrix s_r(s_in);
  mat s(s_r.begin(), s_r.nrow(), s_r.ncol(), false);

  RgamPoisson rgam = RgamPoisson(x, y,  k);

  if (alphas.n_elem == 1) {
    return wrap(rgam(s, alphas(0), epsilon, max_iterations, trace));
  }

  switch (as<int>(span_type_in)) {
  case 0:
    return wrap(rgam(s, CvCrossValidator(y), alphas, epsilon, max_iterations, trace));
  case 2:
    return wrap(rgam(s, DcvCrossValidator(y), alphas, epsilon, max_iterations, trace));
  case 3:
    return wrap(rgam(s, RdcvCrossValidator(y), alphas, epsilon, max_iterations, trace));
  default:
    return wrap(rgam(s, RcvCrossValidator(y, k), alphas, epsilon, max_iterations, trace));
  }
}


RcppExport SEXP rgam_b(SEXP x_in,
                       SEXP y_in,
                       SEXP ni_in,
                       SEXP epsilon_in,
                       SEXP max_iterations_in,
                       SEXP k_in,
                       SEXP span_type_in,
                       SEXP alphas_in,
                       SEXP s_in,
                       SEXP trace_in) {
  RNGScope scope;               // keeping R's RNG in sync

  const double epsilon = as<double>(epsilon_in);
  const int max_iterations = as<int>(max_iterations_in);
  const double k = as<double>(k_in);
  const bool trace = as<bool>(trace_in);

  NumericMatrix x_r(x_in);
  mat x(x_r.begin(), x_r.nrow(), x_r.ncol(), false);

  NumericVector y_r(y_in);
  vec y(y_r.begin(), y_r.size(), false);

  IntegerVector ni_r(ni_in);
  ivec ni(ni_r.begin(), ni_r.size(), false);

  NumericVector alphas_r(alphas_in);
  vec alphas(alphas_r.begin(), alphas_r.size(), false);
  
  NumericMatrix s_r(s_in);
  mat s(s_r.begin(), s_r.nrow(), s_r.ncol(), false);
  
  RgamBinomial rgam = RgamBinomial(x, y, ni,  k);

  if (alphas.n_elem == 1) {
    return wrap(rgam(s, alphas(0), epsilon, max_iterations, trace));
  }

  switch (as<int>(span_type_in)) {
  case 0: return wrap(rgam(s, CvSpanBinomial(y, ni), alphas, epsilon, max_iterations, trace));
  case 2: return wrap(rgam(s, DcvSpanBinomial(y, ni), alphas, epsilon, max_iterations, trace));
  case 3: return wrap(rgam(s, RdcvSpanBinomial(y, ni), alphas, epsilon, max_iterations, trace));
  default: return wrap(rgam(s, RcvSpanBinomial(y, ni, k), alphas, epsilon, max_iterations, trace));
  }
}


RcppExport SEXP my_loess_fit(SEXP y_in,
                             SEXP x_in,
                             SEXP weights_in,
                             SEXP span_in) {
  NumericVector x_r(x_in);
  vec x(x_r.begin(), x_r.size(), false);

  NumericVector y_r(y_in);
  vec y(y_r.begin(), y_r.size(), false);

  NumericVector weights_r(weights_in);
  vec weights(weights_r.begin(), weights_r.size(), false);

  const double span = as<double>(span_in);

  const vec foo = loess_fit(y, x, weights, span);
  return wrap(foo);
}

static const R_CallMethodDef CallEntries[] = {
    {"rgam_p", (DL_FUNC) &rgam_p, 9},
    {"rgam_b", (DL_FUNC) &rgam_b, 10},
    {NULL, NULL, 0}
};

RcppExport void R_init_rgam(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
