#include "rgam_poisson.h"
#include <Rmath.h>
#include <R_ext/Utils.h>

#include <algorithm>
#include <vector>

#include "utils.h"
#include "loess.h"

using namespace arma;

static mat init_s(const vec& y, const unsigned int n, const unsigned int p);
static vec E_psi(const vec& eta, const vec& x, const double k);
static vec E_r(vec const& mu, vec const& a, vec const& b);


BackfitInfo RgamPoisson::rgam(mat s,
                              const double span,
                              const double epsilon,
                              const int max_iterations,
                              const bool trace) {
  R_CheckUserInterrupt();

  vec eta;
  double con = epsilon + 1;
  int i = 0;
  for ( ; i < max_iterations && con > epsilon; ++i) {
    eta = sum(s, 1);
    const vec mu(exp(eta));

    // E(psi(r))
    vec E_ps = E_psi(eta, y, k);
    const double h = 1E-5;
    vec E_ps0 = E_psi(eta-h, y, k);
    vec E_ps1 = E_psi(eta+h, y, k);
    vec E_d_E_ps = (E_ps1 - E_ps0) / (2 * h);

    const vec a(ceiling(mu - k * sqrt(mu)));
    const vec b(floor(mu + k * sqrt(mu)));
    vec E_ps_p = p_pois(b, mu) - p_pois(a - 1, mu); // E(psi'(r))

    const vec r((y - mu) / sqrt(mu));
    const vec H(huber_psi(r, k) - E_ps);
    vec l = - E_ps_p % sqrt(mu) - E_r(mu, a, b)/2 - E_d_E_ps;
    
    vec z = eta - H / l;        // adjusted depended variable
    vec weights = -l % sqrt(mu);
    mat s0(s);
    mat s1 = gam(z, x, weights, span, epsilon, max_iterations);
    s = s1;
    con = sum(apply_norm_col(s-s0)) / sum(apply_norm_col(s0));
    if (trace) {
      Rprintf("i: %d, con: %lf\n", i, con);
    }
  }

  return BackfitInfo(eta, s, i, con, con <= epsilon);
}


/*
 * Run a (less-exact) RGAM on the subset of data, leaving out one
 * column at a time, and calculate the error using the 'span'
 * cross-validation formula
 */
double RgamPoisson::crossValidate(const CrossValidator& span, 
                           const double alpha, 
                           const bool trace) {
  const unsigned int n = x.n_rows;
  const unsigned int p = x.n_cols;
  
  vec rr = zeros<vec>(n-2);

  vec xx(x.submat(1, 0, n-1, 0));
  vec yy(y.rows(1, n-1));
  for (unsigned int j = 1; j < (n-1); ++j) {
    xx(j-1) = x(j-1); // xx <- x[-j]
    yy(j-1) = y(j-1); // yy <- y[-j]
    BackfitInfo backfit = RgamPoisson(xx, yy,  k)
      .rgam(init_s(yy, n-1, p), 
            alpha, 
            1E-4, // higher convergence tolerance
            100,  // fewer iterations
            trace);

    const vec m = backfit.eta;
    const double pr = m(j-1) + (x(j) - x(j-1)) * (m(j) - m(j-1)) /
      (x(j+1) - x(j-1));
    const double ppr = exp(pr);
    rr(j-1) = span(ppr, j);
  }
  return accu(rr);
}


static mat init_s(const vec& y, const unsigned int n, const unsigned int p) {
  mat s = zeros<mat>(n, p+1);
  s.col(0).fill(log(accu(y)/n));
  return s;
}


static vec E_psi(vec const& eta, vec const& y, const double k) {
  const vec mu(exp(eta));
  vec const& v = mu;
  
  const vec j1(floor(mu - k*sqrt(v)));
  const vec j2(floor(mu + k*sqrt(v)));
  
  vec r = (y - mu) / sqrt(v);

  // E(psi(r))
  vec E_ps = k * (1 - p_pois(j2, mu) - p_pois(j1, mu)) +
    (d_pois(j1, mu) - d_pois(j2, mu)) % mu / sqrt(v);
  return E_ps;
}


static vec E_r(vec const& mu, vec const& a, vec const& b) {
  const unsigned int n = mu.n_elem;
  vec result(n);
  for (unsigned int i = 0; i < n; i++) {
    const int t1 = int(a(i));
    const int t2 = int(b(i));

    vec x1(t2 - t1 + 1);
    vec x3(x1.n_elem);
    for (unsigned int j = 0; j < x1.n_elem; ++j) {
      x1(j) = j + t1;
      x3(j) = ::Rf_dpois(x1(j), mu(i), 0);
    }
    const vec x2((x1 - mu(i)) / sqrt(mu(i)));
    result(i) = sum(vec(x2 % x3));
  }
  return result;
}

