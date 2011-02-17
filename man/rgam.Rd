\name{rgam}
\alias{rgam}
\title{Outlier-robust fit for Generalized Additive Models}
\usage{rgam(x, y, family=c("poisson", "binomial"), ni, epsilon=1e-08,
    max.it=50, k=1.5, trace=FALSE, cv.method=c("rcv", "cv", "dcv",
    "rdcv"), alpha=seq(0.1, 0.9, by = 0.1), s.i)}
\description{\code{rgam} is used to obtain an outlier-robust fit for generalized
additive models. It uses the backfitting algorithm with weights
derived from robust quasi-likelihood equations. Currently, only
local regression smoothers are supported. Bandwidth selection using
robust and non-robust cross-validation criteria is currently only
implemented for models with a single covariate.}
\details{The \code{gam} model is fit using the robust local scoring
algorithm, which iteratively fits weighted additive
models by backfitting. The weights are derived from
robust quasi-likelihood estimating equations and thus
effectively reduce the potentially damaging effect of
outliers.

Currently, this function only implements local regression
smoothers (as calculated by \code{loess}). The method can be
applied to other smoothers as well.}
\value{returns an object of class \code{rgam}. It
contains the following components:
\item{additive.predictors}{the additive fit, the
sum of the columns of the \code{$smooth} component}
\item{fitted.values}{the fitted mean values, obtained
by transforming the component 'additive.predictors'
using the inverse link function}
\item{smooth}{the matrix of smooth terms, columns
correspond to the smooth predictors in the model}
\item{iterations}{number of robust local scoring
iterations used}
\item{convergence.criterion}{last relative change of
the additive predictors}
\item{converged}{a logical value indicating whether
the algorithm stopped due to the relative change
of consecutive additive predictors being less than
the tolerance specified in the \code{epsilon} argument
(TRUE) or because the maximum number of iterations
(in the argument \code{max.it}) was reached (FALSE)}
\item{alpha}{the candidate bandwidth values that
were considered}
\item{cv.method}{a character string indicating the
cross-validation method used to choose the
bandwidth of the smoother}
\item{cv.results}{a vector of the cross-validation
criteria values obtained with each entry of the
argument alpha}
\item{opt.alpha}{the value in the argument \code{alpha}
that produced the smallest cross-validation
criterion. This is the bandwidth used for the
reported fit.}}
\author{Matias Salibian-Barrera \email{matias@stat.ubc.ca}
and Davor Cubranic \email{cubranic@stat.ubc.ca}}
\references{Azadeh, A. and Salibian-Barrera, M. (2011).
\emph{An outlier-robust fit for Generalized
Additive Models with applications to disease
outbreak detection.} To appear in the Journal of the
American Statistical Association.}
\keyword{models,}
\keyword{regression,}
\keyword{robust,}
\keyword{smooth}
\arguments{\item{x}{a vector or matrix of covariates}
\item{y}{a vector of responses}
\item{family}{a character string indicating the assumed
distribution of the response (conditional on the covariates).
Only \sQuote{poisson} and \sQuote{binomial} are implemented. The link
function is currently chosen to be the canonical link for
the selected family (\code{log} for \sQuote{poisson} and \code{logit} for
\sQuote{binomial})}
\item{ni}{a vector of the same length as \code{y} containing
the number of tries of the binomial distribution of
each entry of y. Only relevant if the argument family
equals \sQuote{binomial}}
\item{epsilon}{tolerance for the convergence of the
robust local scoring algorithm}
\item{max.it}{maximum number of robust local scoring
iterations}
\item{k}{tuning constant for the robust quasi-likelihood
score equations. Large values of \code{k} make the estimators
closer to the classical fit (and hence less robust),
while smaller values of \code{k} produce a more robust fit.
Values between 1.5 and 3 generally result in a fit with
good robustness properties}
\item{trace}{logical flag to turn on debugging output}
\item{cv.method}{character string indicating which
cross-validation criterion is to be mimized to select
the bandwidth from the list given in the argument
\code{alphas}. Accepted values are \sQuote{rcv} (for a weighted
squared loss where the effect of outliers is reduced);
\sQuote{cv} (for the \dQuote{classical} squared loss); \sQuote{dcv} (for
the classical deviance loss); \sQuote{rdcv} (for a robustly
weighted deviance loss). See the references for more
details}
\item{alpha}{a scalar (for models with a single
covariate it can be a vector of numbers) between 0
and 1. If \code{length(alphas)==1}, its value is used as
bandwidth for the local regression smoother, as
described in \code{loess}. If alphas is a vector, then
the value that minimizes the cross-validation
criterion specified in the argument \sQuote{cv} is used.}
\item{s.i}{optional matrix of initial values for
the additive predictors (including the intercept).
If missing the predictors are initialized at zero
and the intercept is taken to be the transformed
sample mean of the responses.}}
\examples{x <- ili.visits$week
y <- ili.visits$visits
set.seed(123)
x <- x + rnorm(x, mean=0, sd=.01)
#
# the following command needs to run over 890 fits
# and takes about 22 mins on an Intel Xeon CPU (3.2GHz)
#
# a <- rgam(x=x, y=y, family='poisson', cv.method='rcv',
#  epsilon=1e-5, alpha=12:20/80, max.it=500)
#
# the optimal is found at alpha = 17/80
#
a <- rgam(x=x, y=y, family='poisson', cv.method='rcv',
epsilon=1e-7, alpha=17/80, max.it=500)

pr.rgam.a <- predict(a, type='response')
plot(x, y, xlab='Week', ylab='ILI visits', pch=19, col='grey75')
lines(x[order(x)], pr.rgam.a[order(x)], lwd=3, col='red')}
