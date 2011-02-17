#ifndef RGAM_UTILS_H
#define RGAM_UTILS_H

#include <RcppArmadillo.h>

arma::vec floor(arma::vec x);

arma::vec ceiling(arma::vec x);

double logit(double x);

arma::vec logit(arma::vec x);

arma::vec p_pois(arma::vec const& x, arma::vec const& lambda);

arma::vec d_pois(arma::vec const& x, arma::vec const& lambda);

arma::vec p_binom(arma::vec const& x, arma::ivec const& n, arma::vec const& p);

arma::vec apply_norm_col(const arma::mat& m);
arma::vec huber_psi(const arma::vec& x, const double k);
double huber_psi(const double x, const double k);

#endif
