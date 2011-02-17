#ifndef RGAM_SPAN_H
#define RGAM_SPAN_H

#include "span.h"

#include <algorithm>
#include <functional>

class AbstractCvCrossValidator : public CrossValidator {
public:
  AbstractCvCrossValidator(const arma::vec& y)
    : CrossValidator(y)
  {}

  virtual double r_calc(const double ppr, const int i) const = 0;

  double operator() (const double ppr, const int i) const {
    const double r = r_calc(ppr, i);
    return r * r;
  }
};

class CvCrossValidator : public AbstractCvCrossValidator {
public:
  CvCrossValidator(const arma::vec& y)
    : AbstractCvCrossValidator(y)
  {}
  double r_calc(const double ppr, const int i) const {
    return (y(i) - ppr) / sqrt(ppr);
  }
};

class RcvCrossValidator : public AbstractCvCrossValidator {
public:

  RcvCrossValidator(const arma::vec& y, const double k)
    : AbstractCvCrossValidator(y),
      k(k)
  {}

  double r_calc(const double ppr, const int i) const {
    const double r = (y(i) - ppr) / sqrt(ppr);
    return std::max(-k, std::min(r, k));
  }

private:
  const double k;
};

class DcvCrossValidator : public CrossValidator {
public:
  DcvCrossValidator(const arma::vec& y)
    : CrossValidator(y)
  {}

  double operator() (const double ppr, const int i) const {
    return (y(i) <= 0) ? ppr :
      (ppr - y(i)) + y(i) * log(y(i) / ppr);
  }
};

class RdcvCrossValidator : public CrossValidator {
public:
  RdcvCrossValidator(const arma::vec& y)
    : CrossValidator(y),
      d(0.5)
  {}
  RdcvCrossValidator(const arma::vec& y, const double d)
    : CrossValidator(y),
      d(0.5)
  {}

  double operator() (const double ppr, const int i) const {
    const double dd =  (y(i) <= 0) ? ppr :
      (ppr - y(i)) + y(i) * log(y(i) / ppr);
    
    // calculate rho_by: 
    if (dd > d) {
      return -2 * exp(-sqrt(dd)) * (1 + sqrt(dd)) +
        exp(-sqrt(d)) * (2 * (1 + sqrt(d)) + d);
    }
    else return dd * exp(-sqrt(d));
  }

private:
  const double d;
};

#endif
