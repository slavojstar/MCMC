# Making an MCMC run using Robust Adaptive Metropolis to animate

library(mvtnorm)

nits <- 1000
d <- 2
x_curr <- vector(mode = "numeric", length = d)
sigma_curr <-  diag(d)
S_curr <- diag(d) # adaptive cholesky factor
seed <- 1234

target_sigma <- 0.01 * matrix(c(101, 99, 99, 101), nrow = 2, ncol = 2)
target_S <- t(chol(target_sigma))
target_mu <- c(0, 0)

logpi <- function(x, sigma, mu) {
  # normal log(density)
  return(-0.5 * colSums(as.vector(x - mu) * (solve(sigma) %*% as.vector(x - mu))))
}

outer_prod <- function(x, y) {
  out <- matrix(nrow = length(x), ncol = length(y))
  for (i in 1:length(x)) {
    for (j in 1:length(y)) {
      out[i, j] <- x[i] * y[j]
    }
  }
  
  return(out)
}

RAM <- function(nits,
                d,
                x_curr,
                S_curr,
                target_sigma,
                target_mu,
                seed) {
  set.seed(seed)
  # work out the log(density) of the current point
  logpi_curr <- logpi(x_curr, target_sigma, target_mu)
  
  # places to store things
  x_store <- matrix(nrow = nits + 1, ncol = d)
  S_store <- array(data = rep(0, d * d * (nits + 1)), c(d, d, (nits + 1)))
  Sigma_store <- array(data = rep(0, d * d * (nits + 1)), c(d, d, (nits + 1)))
  u_store <- matrix(nrow = nits, ncol = d)
  a_store <- vector(mode = 'numeric', length = nits)
  y_store <- matrix(nrow = nits, ncol = d)
  eta_store <- vector(mode = 'numeric', length = nits)
  logpi_store <- vector(mode = 'numeric', length = nits + 1)
  accept_store <- vector(mode = 'numeric', length = nits)
  
  # store the initial values
  x_store[1, ] <- x_curr
  S_store[, ,1] <- S_curr
  Sigma_store[, ,1] <- S_curr %*% t(S_curr)
  logpi_store[1] <- logpi_curr
  
  # let's go
  for (i in 1:nits) {
    U <- t(rmvnorm(n = 1, mean = rep(0, d)))
    y <- x_curr + as.vector(S_curr %*% U)
    
    # accept/reject
    u <- runif(1)
    logpi_prop <- logpi(y, target_sigma, target_mu)
    loga <- logpi_prop - logpi_curr
    if (log(u) < loga) {
      # accept
      x_curr <- y
      logpi_curr <- logpi_prop
      accept_store[i] <- 1
    } else {
      # reject
      accept_store[i] <- 0
    }
    # store more things
    x_store[(i + 1), ] <- x_curr
    u_store[i, ] <- U
    a <- min(1, exp(loga))
    a_store[i] <- a
    y_store[i, ] <- y
    logpi_store[(i + 1)] <- logpi_curr
    
    # adapt
    u_part <- outer_prod(U, U) / sum(U ^ 2)
    eta <- (i ^ (-0.7))
    middle_part <- diag(d) + eta * (a - 0.234) * u_part
    sigma <- S_curr %*% middle_part %*% t(S_curr)
    S_curr <- t(chol(sigma)) # maybe try cholesky() here if chol() doesn't work
    
    # store everything else
    S_store[, ,(i + 1)] <- S_curr
    Sigma_store[, ,(i + 1)] <- sigma
    eta_store[i] <- eta
  }
  return(list(x_store = x_store,
              S_store = S_store,
              Sigma_store = Sigma_store,
              u_store = u_store,
              a_store = a_store,
              y_store = y_store,
              eta_store = eta_store,
              logpi_store = logpi_store,
              accept_store = accept_store))
}

run_1 <- RAM(nits = nits,
             d = d,
             x_curr = x_curr,
             S_curr = S_curr,
             target_sigma = target_sigma,
             target_mu = target_mu,
             seed = seed)

nits <- 100000
run_2 <- RAM(nits = nits,
             d = d,
             x_curr = x_curr,
             S_curr = S_curr,
             target_sigma = target_sigma,
             target_mu = target_mu,
             seed = seed)








