\name{ili.visits}
\alias{ili.visits}
\docType{data}
\title{
  Influenza-like-illness outpatient visits.
}
\description{
  These data contain the number of reported influenza-like-illness
  outpatient visits in the US for the 2006, 2007 and 2008 seasons. A
  season consists of weeks 40 to week 20 of the following year, so that,
  for example, the 2008 season includes up to week 20 of 2009, where the
  H1N1 flu epidemic had already caused a noticeable increase in ILI
  visits.
}
\usage{ili.visits}
\format{
  A data frame with 99 observations on the following 3 variables.
  \describe{
    \item{\code{visits}}{the number of reported ILI visits for that week
      (numeric)}
    \item{\code{season}}{the season year (numeric)}
    \item{\code{week}}{the week number for that season (numeric)}
  }
}
\source{
  US Centre for Disease Control, \url{http://www.cdc.gov/flu/weekly/fluactivity.htm}
}
\examples{
library(rgam)
plot(visits ~ week, data=ili.visits, xlab='Week', ylab='ILI visits',
pch=19, col='grey75')
with(ili.visits, {
  a <- rgam(x=week, y=visits, family='poisson', cv.method='rcv',
            epsilon=1e-7, alpha=17/80, max.it=500)
  pr.rgam <- predict(a, type='response')
  lines(week[order(week)], pr.rgam[order(week)], lwd=3, col='red')
})
}
\keyword{datasets}
