#include "utils.h"
#include <Rmath.h>

#include <cmath>
#include <algorithm>

using namespace arma;

vec floor(vec x) {
  vec y(x.n_elem);
  for (unsigned int i = 0; i < x.n_elem; ++i) {
    y(i) = floor(x(i));
  }
  return y;
}

vec ceiling(vec x) {
  vec y(x.n_elem);
  for (unsigned int i = 0; i < x.n_elem; ++i) {
    y(i) = ceil(x(i));
  }
  return y;
}

double logit(double x) {
  return log(x / (1 - x));
}

vec logit(vec x) {
  return log(x / (1 - x));
}

vec p_pois(vec const& x, vec const& lambda) {
  vec y(x.n_elem);
  for (unsigned int i = 0; i < x.n_elem; ++i) {
    y(i) = ::Rf_ppois(x(i), lambda(i), 1, 0);
  }
  return y;
}

vec d_pois(vec const& x, vec const& lambda) {
  vec y(x.n_elem);
  for (unsigned int i = 0; i < x.n_elem; ++i) {
    y(i) = ::Rf_dpois(x(i), lambda(i), 0);
  }
  return y;
}


vec p_binom(vec const& x, ivec const& n, vec const& p) {
  vec y(x.n_elem);
  for (unsigned int i = 0; i < x.n_elem; ++i) {
    y(i) = ::Rf_pbinom(x(i), n(i), p(i), 1, 0);
  }
  return y;
}


vec apply_norm_col(const mat& m) {
  const unsigned int p = m.n_cols;
  vec result(p);
  for (unsigned int i = 0; i < p; ++i) {
    result(i) = norm(m.col(i), 2);
  }
  return result;
}


vec huber_psi(const vec& x, const double k) {
  vec result(x.n_elem);
  for (unsigned int i = 0; i < x.n_elem; i++) {
    result(i) = huber_psi(x(i), k);
  }
  return result;
}

double huber_psi(const double x, const double k) {
  return std::max(-k, std::min(x, k));
}
