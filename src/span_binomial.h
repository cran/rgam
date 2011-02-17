#ifndef RGAM_SPAN_BINOMIAL_H
#define RGAM_SPAN_BINOMIAL_H

#include "span.h"

#include <algorithm>
#include <functional>

class SpanBinomial : public CrossValidator {
public:

  SpanBinomial(const arma::vec& y, const arma::ivec& ni)
    : CrossValidator(y),
      ni(ni)
  {}

protected:
  const arma::ivec ni;
};


class CvSpanBinomial : public SpanBinomial {
public:
  CvSpanBinomial(const arma::vec& y, 
                 const arma::ivec& ni)
    : SpanBinomial(y, ni)
  {}

  double operator() (const double pb, const int i) const {
    const double mub = ni(i) * pb;
    const double r = (y(i) - mub) / sqrt(mub * (1 - pb));
    return r * r;
  }
};

class RcvSpanBinomial : public SpanBinomial {
public:
  RcvSpanBinomial(const arma::vec& y, 
                  const arma::ivec& ni,
                  const double k)
    : SpanBinomial(y, ni),
      k(k)
  {}

  double operator() (const double pb, const int i) const {
    const double mub = ni(i) * pb;
    const double r = (y(i) - mub) / sqrt(mub * (1 - pb));
    const double huber_psi = std::max(-k, std::min(r, k));
    /* std::cout << "ppr: " << ppr << ", y: " << y <<  */
    /*   ", r: " << r << ", h_psi: " << huber_psi << std::endl; */
    return huber_psi * huber_psi;
  }
private:
  const double k;
};

class DcvSpanBinomial : public SpanBinomial {
public:
  DcvSpanBinomial(const arma::vec& y, 
                  const arma::ivec& ni)
    : SpanBinomial(y, ni)
  {}

  double operator() (const double pb, const int i) const {
    const double mub = ni(i) * pb;

    return (y(i) <= 0) ? -ni(i) * log(1-pb) :
      (y(i) < ni(i)) ? (y(i) * log(y(i)/mub) + 
                        (ni(i) - y(i)) * log((ni(i)-y(i)) / (ni(i)-mub))) :
      y(i) * log(y(i) / mub);
  }
};

class RdcvSpanBinomial : public DcvSpanBinomial {
public:
  RdcvSpanBinomial(const arma::vec& y, 
                   const arma::ivec& ni)
    : DcvSpanBinomial(y, ni),
      d(0.5)
  {}

  RdcvSpanBinomial(const arma::vec& y, 
                   const arma::ivec& ni,
                   const double d)
    : DcvSpanBinomial(y, ni),
      d(d)
  {}

  double operator() (const double pb, const int i) const {
    const double dd =  DcvSpanBinomial::operator()(pb, i);
    
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
