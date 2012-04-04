\name{predict.rgam}
\alias{predict.rgam}
\title{Predict method for RGAM fits}
\usage{\method{predict}{rgam}(object, type=c("link", "response", "terms"), ...)}
\description{Obtains predictions from a robustly fitted generalized
additive model object}
\details{Serves as the extractor function on objects of class
\code{rgam}.}
\value{the component of the \sQuote{object} based on the value
of \code{type}: if the \sQuote{code} is \sQuote{response}, returns the
\code{$fitted.values}; if the type is \sQuote{link}, returns the
\code{$additive.predictors}; and if the type is \sQuote{terms},
returns the \code{$smooth} component.}
\author{Matias Salibian-Barrera \email{matias@stat.ubc.ca}
and Davor Cubranic \email{cubranic@stat.ubc.ca}}
\arguments{\item{object}{a fitted \code{rgam} object}
\item{type}{a character string specifying the type of predictions.
Can be one of \sQuote{link} (the default), \sQuote{response},
or \sQuote{terms}.}
\item{...}{additional arguments passed from other methods}}
