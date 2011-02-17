#ifndef RGAM_BINOMIAL_H
#define RGAM_BINOMIAL_H

#include <RcppArmadillo.h>

#include <algorithm>
#include <functional>

#include "rgam.h"

class RgamBinomial : public Rgam {
public:
  
  RgamBinomial(const arma::mat& x,
               const arma::vec& y,
               const arma::ivec& ni,
               const double k)
    : Rgam(x, y, k),
      ni(ni)
  {}

protected:
  BackfitInfo rgam(arma::mat s,
                   const double span,
                   const double epsilon,
                   const int max_iterations,
                   const bool trace);

  double crossValidate(const CrossValidator& span, 
                       const double alpha,
                       const bool trace);

  arma::vec predict(arma::vec eta) { return 1 / (1 + arma::exp(-eta)); }
  
private:
  const arma::ivec ni;
};

#endif
