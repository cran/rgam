#ifndef LOESS_H
#define LOESS_H

#include <RcppArmadillo.h>

arma::vec loess_fit(arma::vec const& y, arma::vec const& x, 
                    arma::vec const& weights, const double span);

#endif
