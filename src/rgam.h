#ifndef RGAM_H
#define RGAM_H

#include <RcppArmadillo.h>

struct RgamResult;
struct BackfitInfo;

#include "span.h"

struct BackfitInfo {
  BackfitInfo(const arma::vec& eta,
              const arma::mat& s, 
              const int iterations, 
              const double con,
              const bool converged)
    : eta(eta), s(s), iterations(iterations),
      con(con), converged(converged) {}

  const arma::vec eta;
  const arma::mat s;
  const int iterations;
  const double con;
  const bool converged;
};

struct RgamResult {

  RgamResult(const arma::vec& prediction,
             const double alpha,
             const arma::vec& cv_result,
             const BackfitInfo& backfit)
    : prediction(prediction),
      alpha(alpha), cv_result(cv_result),
      s(backfit.s), eta(backfit.eta), 
      iterations(backfit.iterations),
      con(backfit.con), converged(backfit.converged) {}

  RgamResult(const arma::vec& prediction,
             const double alpha,
             const BackfitInfo& backfit)
    : prediction(prediction),
      alpha(alpha), cv_result(),
      s(backfit.s), eta(backfit.eta), 
      iterations(backfit.iterations),
      con(backfit.con), converged(backfit.converged) {}

  // predicted Y
  const arma::vec prediction;
  
  // alpha used for the prediction
  const double alpha;

  // cross-validation result for each candidate alpha
  const arma::vec cv_result;

  const arma::mat s;
  const arma::vec eta;
  const int iterations;
  const double con;
  const bool converged;
};

class Rgam {
public:
  Rgam(const arma::mat& x,
       const arma::vec& y,
       const double k)
    : x(x),
      y(y),
      k(k)
  {}

  RgamResult operator() (const arma::mat& s,
                         const CrossValidator& span,
                         const arma::vec& alphas,
                         const double epsilon, 
                         const int max_iterations,
                         const bool trace);

  RgamResult operator() (const arma::mat& s,
                         const double alpha,
                         const double epsilon,
                         const int max_iterations,
                         const bool trace);

  static arma::mat gam(const arma::vec& y,
                       const arma::mat& x, 
                       const arma::vec& weights,
                       const double span,
                       const double epsilon,
                       const int max_iterations);
protected:
  arma::vec find_span(const arma::vec& alphas,
                      const CrossValidator& span,
                      const bool trace);

  virtual BackfitInfo rgam(arma::mat s,
                   const double span,
                   const double epsilon,
                   const int max_iterations,
                   const bool trace) = 0;

  virtual double crossValidate(const CrossValidator& span, 
                       const double alpha,
                       const bool trace) = 0;
  virtual arma::vec predict (arma::vec eta) = 0;

  const arma::mat x;
  const arma::vec y;
  const double k;

private:
  std::pair<double, arma::vec> 
  chooseAlpha(const arma::mat& s,
              const CrossValidator& span,
              const arma::vec& alphas,
              const bool trace);
       
};


#endif
