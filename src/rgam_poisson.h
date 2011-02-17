#ifndef RGAM_POISSON_H
#define RGAM_POISSON_H

#include <RcppArmadillo.h>

#include <algorithm>
#include <functional>

#include "rgam.h"

class RgamPoisson : public Rgam {
public:
  
  RgamPoisson(const arma::mat& x,
              const arma::vec& y,
              const double k)
    : Rgam(x, y, k)
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
  arma::vec predict (arma::vec eta) { return arma::exp(eta); }

};

#endif
