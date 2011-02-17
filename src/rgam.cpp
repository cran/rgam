#include "rgam.h"

#include "loess.h"
#include "utils.h"

#include <algorithm>
#include <utility>
#include <vector>

#include <R_ext/Utils.h>

using namespace arma;


vec Rgam::find_span(const vec& alphas,
                    const CrossValidator& span,
                    const bool trace) {
  const int n_alphas = alphas.n_elem;
  
  vec result(n_alphas);

  // Cross-validate with each alpha value
  for (int i = 0; i < n_alphas; ++i) {
    const double alpha = alphas(i);
    result(i) = crossValidate(span, alpha, trace);
  }
  
  return result;
}

std::pair<double, vec> 
Rgam::chooseAlpha(const mat& s,
                  const CrossValidator& span,
                  const vec& alphas,
                  const bool trace) {
  // span value for each alpha
  const vec spans = find_span(alphas, span, trace);

  // find the smallest alpha that yields the minimum span value
  const double min_span = min(spans);
  std::vector<double> min_alphas;
  for (unsigned int i = 0; i < alphas.n_elem; ++i) {
    if (spans(i) == min_span) {
      const double alpha = alphas(i);
      min_alphas.push_back(alpha);
    }
  }

  return std::make_pair(* std::min_element(min_alphas.begin(), 
                                           min_alphas.end()),
                        spans);
}

RgamResult Rgam::operator() (const mat& s,
                             const CrossValidator& span,
                             const vec& alphas,
                             const double epsilon,
                             const int max_iterations,
                             const bool trace) {
  
  const std::pair<double, vec> pp = chooseAlpha(s, span, alphas, trace);
  const double alpha = pp.first;
  const vec cv_result = pp.second;

  BackfitInfo backfit = rgam(s, alpha, epsilon, max_iterations, trace);

  return RgamResult(predict(backfit.eta),
                    alpha,
                    cv_result,
                    backfit);
}


RgamResult Rgam::operator() (const mat& s,
                             const double alpha,
                             const double epsilon,
                             const int max_iterations,
                             const bool trace) {
  BackfitInfo backfit = rgam(s, alpha, epsilon, max_iterations, trace);

  return RgamResult(predict(backfit.eta),
                    alpha,
                    backfit);
}

mat Rgam::gam(const vec& y,
              const mat& x, 
              const vec& weights,
              const double span,
              const double epsilon,
              const int max_iterations) {
  const int n = x.n_rows;
  const int p = x.n_cols;

  mat s = zeros<mat>(n, p+1);
  s.col(0).fill(accu(y % weights) / accu(weights));

  double ave = var(y) * (n-1) / n;
  double last_ave = ave + 10 * epsilon;
  for (int i = 0 ; 
       i < max_iterations && 
         (last_ave - ave)/ave > epsilon; ++i) {
    for (int j = 1; j <= p; ++j) {
      vec column_residual = y - (sum(s, 1) - s.col(j));
      s.col(j) = loess_fit(column_residual, x.col(j-1), weights, span);
    }
    vec residual = y - sum(s, 1);
    last_ave = ave;
    ave = mean(vec(square(residual)));
  }
  if ((last_ave - ave) / ave > epsilon) {
    // TODO: signal that we did not converge
  }
  return s;
}

