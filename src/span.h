#ifndef SPAN_H
#define SPAN_H

#include <RcppArmadillo.h>

class CrossValidator : public std::binary_function<double, int, double> {
public:

  CrossValidator(const arma::vec& y)
    : y(y)
  {}

  virtual double operator() (const double ppr, const int i) const = 0;

protected:
  const arma::vec y;
};

#endif

