#include "rgam_binomial.h"
#include <Rmath.h>
#include <R_ext/Utils.h>

#include "utils.h"
#include "loess.h"

using namespace arma;
using namespace Rcpp;

static vec E_psi(vec const& eta, vec const& my, vec const& v, 
                 const ivec& ni, const double k);
static vec E_r(const ivec& ni, vec const& pi, vec const& a, vec const& b);

BackfitInfo RgamBinomial::rgam(mat s,
                               double const span,
                               const double epsilon,
                               const int max_iterations,
                               const bool trace) {
  R_CheckUserInterrupt();
  
  vec eta;
  double con = epsilon + 1;
  int i = 0;
  for ( ; i < max_iterations && con > epsilon; ++i) {
    eta = sum(s, 1);
    const vec pi = exp(eta) / (1 + exp(eta));
    const vec mu = ni % pi;
    const vec v = ni % pi % (1 - pi);
  
    // E(psi(r))
    vec E_ps = E_psi(eta, mu, v, ni, k);
    const double h = 1E-5;
    vec E_ps0 = E_psi(eta-h, mu, v, ni, k);
    vec E_ps1 = E_psi(eta+h, mu, v, ni, k);
    vec E_d_E_ps = (E_ps1 - E_ps0) / (2 * h);

    const vec a(ceiling(mu - k * sqrt(v)));
    const vec b(floor(mu + k * sqrt(v)));
    vec E_ps_p = p_binom(b, ni, pi) - p_binom(a - 1, ni, pi); // E(psi'(r))

    const vec r = (y - mu) / sqrt(v);
    const vec H(huber_psi(r, k) - E_ps);
    vec l = - E_ps_p % sqrt(v) - E_r(ni, pi, a, b)/(2 * (1 + exp(eta))) - E_d_E_ps;
    
    vec z = eta - H / l;        // adjusted depended variable
    vec weights = -l % sqrt(v);
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


static mat init_s(const vec& y, const ivec& ni, const unsigned int n, const unsigned int p);


double RgamBinomial::
crossValidate(const CrossValidator& span,
              const double alpha,
              const bool trace) {
  const unsigned int n = x.n_rows;
  const unsigned int p = x.n_cols;
  vec rr = zeros<vec>(n-2);

  vec xx(x.submat(1, 0, n-1, 0));
  vec yy(y.rows(1, n-1));
  ivec nni(ni.rows(1, n-1));
  for (unsigned int j = 1; j < (n-1); ++j) {
    xx(j-1) = x(j-1); // xx <- x[-j]
    yy(j-1) = y(j-1); // yy <- y[-j]
    nni(j-1) = ni(j-1); // nni <- ni[-j]
    BackfitInfo backfit = RgamBinomial(xx, yy, nni, k)
      .rgam(init_s(yy, nni, n-1, p), 
            alpha, 1E-4, 100, trace);
    const vec m = backfit.eta;
    const double pr = m(j-1) + (x(j) - x(j-1)) * (m(j) - m(j-1)) /
      (x(j+1) - x(j-1));
    const double pb = exp(pr) / (1 + exp(pr));
    rr(j-1) = span(pb, j);
  }
  return accu(rr);
}

static mat init_s(const vec& y, const ivec& ni, const unsigned int n, const unsigned int p) {
  mat s = zeros<mat>(n, p+1);
  s.col(0).fill(logit(accu(y/ni) / n));
  return s;
}


static vec E_psi(vec const& eta, vec const& mu, vec const& v, 
                 ivec const& ni, const double k) {
  const vec pi = exp(eta) / (1 + exp(eta));
  const vec mu_p = ni % pi;
  const vec v_p = ni % pi % (1 - pi);
  
  const vec j1(floor(mu_p - k*sqrt(v_p)));
  const vec j2(floor(mu_p + k*sqrt(v_p)));

  // E(psi(r))
  vec E_ps = k * (1 - p_binom(j2, ni, pi) - p_binom(j1, ni, pi)) +
    ((p_binom(j2-1, ni-1, pi) - p_binom(j1-1, ni-1, pi)) -
     (p_binom(j2, ni, pi) - p_binom(j1, ni, pi))) % mu / sqrt(v);

  return E_ps;
}

static vec E_r(ivec const& ni, vec const& pi, vec const& a, vec const& b) {
  const unsigned int n = pi.n_elem;
  vec result(n);
  for (unsigned int i = 0; i < n; i++) {
    const int t1 = int(a(i));
    const int t2 = int(b(i));
    const int t3 = ni(i);
    const double t4 = double(pi(i));

    vec x1(t2 - t1 + 1);
    const double x2 = t3 * t4;
    const double x3 = x2 * (1 - t4);
    vec x4(x1.n_elem);
    vec x5(x1.n_elem);
    for (unsigned int j = 0; j < x1.n_elem; ++j) {
      x1(j) = j + t1;
      x4(j) = (x1(j) - x2) / sqrt(x3);
      x5(j) = ::Rf_dbinom(x1(j), t3, t4, 0);
    }
    result(i) = sum(vec(x4 % x5));
  }
  return result;
}
