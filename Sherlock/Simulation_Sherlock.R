rm(list=ls())

#. Function eligibility:
Eligible <- function (criteria, data){
  
  expr <- paste (paste0 ("(data$", criteria, ")"), collapse = " & ")
  vars <- all.vars (parse (text = expr))[-1]
  
  if (all (vars %in% colnames (data))){
    with (data, eval (parse (text = expr)))
  } else {
    stop ("Not all criteria are found in dataframe!")
  }
  
}

# Calculate cumulative baseline hazard manually
breslowEstimator <- function(STIME, STATUS, X, B){
  data <- data.frame(STIME, STATUS, X)
  data <- data[order(data$STIME), ]
  t    <- unique(data$STIME)
  k    <- length(t)
  h    <- rep(0,k)
  
  for(i in 1:k) {
    lp <- (data.matrix(data[,-c(1:2)]) %*% B)[data$STIME>=t[i]]
    risk <- exp(lp)
    h[i] <- sum(data$STATUS[data$STIME==t[i]]) / sum(risk)
  }
  
  res <- cumsum(h)
  return(res)
}

cutoff = function (cutoffValue, w, muZ1, muZ2, muZ3, sigmaZ1, sigmaZ2, sigmaZ3, sigmaX1, delta, alphaEQ) {
  
  theta = delta - qnorm(1-alphaEQ/2)*sigmaZ2
  cov.Z1Z2 = sigmaX1^2
  
  # function Z1
  
  probability1 <- pmvnorm(lower = c(cutoffValue, theta), upper = c(Inf, Inf), mean = c(muZ1, muZ2), sigma = matrix(c(sigmaZ1^2, cov.Z1Z2, cov.Z1Z2, sigmaZ2^2), nrow = 2)) + 
    pmvnorm(lower = c(cutoffValue, -Inf), upper = c(Inf, -theta), mean = c(muZ1, muZ2), sigma = matrix(c(sigmaZ1^2, cov.Z1Z2, cov.Z1Z2, sigmaZ2^2), nrow = 2)) +
    pmvnorm(lower = c(-Inf, -Inf), upper = c(-cutoffValue, -theta), mean = c(muZ1, muZ2), sigma = matrix(c(sigmaZ1^2, cov.Z1Z2, cov.Z1Z2, sigmaZ2^2), nrow = 2)) +
    pmvnorm(lower = c(-Inf, theta), upper = c(-cutoffValue, Inf), mean = c(muZ1, muZ2), sigma = matrix(c(sigmaZ1^2, cov.Z1Z2, cov.Z1Z2, sigmaZ2^2), nrow = 2))

  
  # function Z3
  
  probability2 <- pmvnorm(lower = c(cutoffValue, -theta), upper = c(Inf, theta), mean = c(muZ3, muZ2), sigma = matrix(c(sigmaZ3^2, 0, 0,  sigmaZ2^2), nrow = 2)) +
    pmvnorm(lower = c(-Inf, -theta), upper = c(-cutoffValue, theta), mean = c(muZ3, muZ2), sigma = matrix(c(sigmaZ3^2, 0, 0,  sigmaZ2^2), nrow = 2))

  
  return(c(probability1 + probability2, probability1, probability2))
}

standardized_cutoff = function (cutoffValue, muZ1, muZ2, muZ3, sigmaZ1, sigmaZ2, sigmaZ3, sigmaX1, delta, alphaEQ) {
  
  theta = delta/sigmaZ2 - qnorm(1-alphaEQ/2)
  cov.Z1Z2 = sigmaX1^2/sigmaZ1/sigmaZ2
  
  # function Z1
  
  probability1 <- pmvnorm(lower = c(cutoffValue, theta), upper = c(Inf, Inf), mean = c(muZ1, muZ2), sigma = matrix(c(1, cov.Z1Z2, cov.Z1Z2, 1), nrow = 2)) + 
    pmvnorm(lower = c(cutoffValue, -Inf), upper = c(Inf, -theta), mean = c(muZ1, muZ2), sigma = matrix(c(1, cov.Z1Z2, cov.Z1Z2, 1), nrow = 2)) +
    pmvnorm(lower = c(-Inf, -Inf), upper = c(-cutoffValue, -theta), mean = c(muZ1, muZ2), sigma = matrix(c(1, cov.Z1Z2, cov.Z1Z2, 1), nrow = 2)) +
    pmvnorm(lower = c(-Inf, theta), upper = c(-cutoffValue, Inf), mean = c(muZ1, muZ2), sigma = matrix(c(1, cov.Z1Z2, cov.Z1Z2, 1), nrow = 2))

  
  # function Z3
  
  probability2 <- pmvnorm(lower = c(cutoffValue, -theta), upper = c(Inf, theta), mean = c(muZ3, muZ2), sigma = matrix(c(1, 0, 0, 1), nrow = 2)) +
    pmvnorm(lower = c(-Inf, -theta), upper = c(-cutoffValue, theta), mean = c(muZ3, muZ2), sigma = matrix(c(1, 0, 0, 1), nrow = 2))

  
  return(c(probability1 + probability2, probability1, probability2))
}

Var_W <- function (Z1, Z2,
                   var_Z1, var_Z2, sigma_X1,
                   delta, w, nsim) {
  
  # E[I(abs(Z2) < theta) * Z2]
  
  E1 = integrate(function(z) {z*dnorm(z, Z2, sqrt(var_Z2))}, -delta, delta)$value
  
  # E[I(abs(Z2) < theta) * Z2^2]
  
  E2 = integrate(function(z) {z^2*dnorm(z, Z2, sqrt(var_Z2))}, -delta, delta)$value
  
  var_W = var_Z1 + w^2*(E2 - E1^2) - 2*w*(E2 - Z2*E1)*sigma_X1^2/var_Z2
  
  return(var_W)
}

Matcher.v2 <- function (data){
  
  #. Required vars in data:
  ## PS = Score to be matched on
  ## ENROLL = [1] enrolled, [0] control
  ## ID = identifying variable
  
  #. Define datasets
  enroll_0 <- data[data$ENROLL == 0, ]
  enroll_1 <- data[data$ENROLL == 1, ]
  
  # Calculate the matchMatrix efficiently
  matchMatrix <- abs (outer (enroll_1$PS, enroll_0$PS, "-"))
  
  # Check for too few matches
  if (nrow(matchMatrix) > ncol(matchMatrix)) {
    stop("Too few matches")
  }
  
  # Initialize vectors to store matches and calipers
  num_matches <- nrow (matchMatrix)
  match_indices <- matrix (0, nrow = num_matches, ncol = 2)
  match.caliper <- numeric (num_matches)
  
  lapply (1:num_matches, function(i) {
    min_indices <- which (matchMatrix == min (matchMatrix), arr.ind = TRUE)
    min_indices <- min_indices[1, ]
    match_indices[i, ] <<- min_indices
    match.caliper[i] <<- matchMatrix[min_indices[1], min_indices[2]]
    
    # Set matched row and column to a high value to exclude them
    matchMatrix[min_indices[1], ] <<- Inf
    matchMatrix[, min_indices[2]] <<- Inf
  })
  
  # Extract matches
  match.ii <- enroll_1$ID[match_indices[, 1]]
  match.jj <- enroll_0$ID[match_indices[, 2]]
  
  #. Save results & match with PS
  R <- data.frame (TrialID = match.ii,
                   MatchID = match.jj,
                   Caliper = match.caliper,
                   ID = match.ii,
                   ID2 = match.jj) 
  
  R <- merge (R, data[, c ("ID", "PS")])
  R <- R[, !colnames (R) == "ID"]
  colnames (R)[colnames (R) == "ID2"] <- "ID"
  colnames (R)[colnames (R) == "PS"] <- "TrialPS"
  R <- merge (R, data[, c ("ID", "PS")])
  R <- R[, !colnames (R) == "ID"]
  colnames (R)[colnames (R) == "PS"] <- "MatchPS"
  
  #. Return 
  return (R)
  
}

logML <- function(mean1, mean2, sigmaX1, sigmaX2, alpha,sig0,mu0){
  # sig0 is the prior variance and mu0 is the prior of shared mu
  
  # Origin
  signew <- 1/(1/(sigmaX2)^2 * alpha + 1/sig0^2)
  munew <- signew * (1/(sigmaX2)^2 * alpha * mean2 + mu0/sig0^2)
  sigstar <- 1/(1/(sigmaX2)^2 * alpha + 1/sig0^2 + 1/(sigmaX1)^2)
  mustar <- sigstar * (1/(sigmaX2)^2 * alpha * mean2 + mu0/sig0^2 + 1/(sigmaX1)^2 * mean1)
  logllkOrigin <- 0.5 * log(1/signew^2) - 0.5 * munew^2/signew^2 + 0.5 * mustar^2/sigstar^2
  # Deriv
  logllkDeriv <- 1/(1/(sigmaX2)^2 * alpha + 1/sig0^2) -
    (2 * (1/(sigmaX2)^2 * alpha * mean2 + mu0/sig0^2) * mean2 * (1/(sigmaX2)^2 + 1/sig0^2) -
       (1/(sigmaX2)^2 * alpha * mean2 + mu0/sig0^2)^2)/(1/(sigmaX2)^2 * alpha + 1/sig0^2)^2 +
    (2 * (1/(sigmaX2)^2 * alpha * mean2 + mu0/sig0^2 + 1/(sigmaX1)^2 * mean1) * mean2 * (1/(sigmaX2)^2 + 1/sig0^2 + 1/(sigmaX1)^2) -
       (1/(sigmaX2)^2 * alpha * mean2 + mu0/sig0^2 + 1/(sigmaX1)^2 * mean1)^2)/(1/(sigmaX2)^2 * alpha + 1/sig0^2 + 1/(sigmaX1)^2)^2
  logllkDeriv <- logllkDeriv * 1/(sigmaX2)^2
  return(c(logllkOrigin,logllkDeriv))
}

logML2 = function (alpha, X1, sigma_X1, X2, sigma_X2, mu0, sigma0) {
  numerator = integrate(function(mu) {dnorm(x = mu, mean = X1, sd = sigma_X1)*dnorm(x = mu, mean = X2, sd = sigma_X2)^alpha*dnorm(x = mu, mean = mu0, sd = sigma0)}, -Inf, Inf)$value
  denominator = integrate(function(mu) {dnorm(x = mu, mean = X2, sd = sigma_X2)^alpha*dnorm(x = mu, mean = mu0, sd = sigma0)}, -Inf, Inf)$value
  return(numerator/denominator)
}

inference = function (dataset, alpha_p, delta, alphaEQ) {
  
  res = c()
  for (i in 1:nrow(dataset)) {
    dataentry = dataset[i,]
    
    w = dataentry$w
    
    hr_trt_control = dataentry$hr_trt_control
    hr_control_RWD = dataentry$hr_control_RWD
    hr_borrow = dataentry$hr_borrow
    
    Z1 = dataentry$Z1
    Z2 = dataentry$Z2
    Z3 = dataentry$Z3
    B1 = dataentry$B1
    B2 = dataentry$B2
    
    sigma_Z1 = dataentry$sigma_Z1
    sigma_Z2 = dataentry$sigma_Z2
    sigma_Z3 = dataentry$sigma_Z3
    var_B1 = dataentry$var_B1
    var_B2 = dataentry$var_B2
    
    var_Z1 = sigma_Z1^2
    var_Z2 = sigma_Z2^2
    var_Z3 = sigma_Z3^2
    
    #. Covariance between Z1 and Z2
    covZ1Z2 =  dataentry$covZ1Z2
    covB1B2 =  dataentry$covB1B2
    
    #. delta and alphaEQ are the equivalence boundary and the significance level of the equivalence test
    #. if the absolute value of Z2 is lower than theta, we borrow. Otherwise, we do not borrow
    theta = delta/sigma_Z2 - qnorm(1-alphaEQ/2)
    borrow = abs (Z2/sigma_Z2) <= theta
    
    ### Normal approximation (Approach 1) ###
    
    #. Calculate the standard deviation of the overall test statistics, which is expressed as big W
    sigma_W = sqrt(Var_W(Z1, Z2,
                         var_Z1, var_Z2, sqrt(covZ1Z2),
                         theta*sigma_Z2, w))
    
    ### Split type I error (Approach 2) ###
    
    #. v is an array of proportion split to the non-borrowing cases 
    v = seq(0,1,0.1)
    names(v) = paste0("cutoffValue_nonborrowing_", v)
    lower = -1e4
    upper = 1e4
    
    #. When cutoff value equals infinity, that is, we always reject null, the followings are the borrowing probability and non-borrowing probability
    #. We can not split a proportion of type I error that is higher than the borrowing probability to borrowing case. Vice versa
    max_value_nonborrowing = cutoff(0, w, 0, 0, 0, sigma_Z1, sigma_Z2, sigma_Z3, sqrt(covZ1Z2), delta, alphaEQ)[2]
    max_value_borrowing = cutoff(0, w, 0, 0, 0, sigma_Z1, sigma_Z2, sigma_Z3, sqrt(covZ1Z2), delta, alphaEQ)[3]
    
    #. Proportion of type I error split to non-borrowing cases and borrowing cases 
    typeIerror_nonborrowing = alpha_p*(v)
    typeIerror_borrowing = alpha_p*(1-v)
    
    #. If we split too much to non borrowing cases:
    if (sum(typeIerror_nonborrowing > max_value_nonborrowing)>0) {
      cutoffValue_nonborrowing = sapply(1:length(v), function(ii) {ifelse((typeIerror_nonborrowing > max_value_nonborrowing)[ii] == TRUE, 0, uniroot(function(cutoffValue){
        cutoff(cutoffValue, w, 0, 0, 0, sigma_Z1, sigma_Z2, sigma_Z3, sqrt(covZ1Z2), delta, alphaEQ)[2] - typeIerror_nonborrowing[ii]
      },lower = lower, upper = upper, tol = 1e-8, maxiter = 1e4)$root)})
      typeIerror_left = sapply(typeIerror_nonborrowing - max_value_nonborrowing, function(ii){ifelse(ii > 0, ii, 0)})
      
      typeIerror_borrowing = typeIerror_borrowing + typeIerror_left
      cutoffValue_borrowing = sapply(typeIerror_borrowing, function(ii) {
        uniroot(function(cutoffValue){
          cutoff(cutoffValue, w, 0, 0, 0, sigma_Z1, sigma_Z2, sigma_Z3, sqrt(covZ1Z2), delta, alphaEQ)[3] - ii
        },lower = lower, upper = upper, tol = 1e-8, maxiter = 1e4)$root
      })
    }
    #. If we split too much to borrowing cases:
    else if (sum(typeIerror_borrowing > max_value_borrowing)>0) {
      cutoffValue_borrowing = sapply(1:length(v), function(ii) {ifelse((typeIerror_borrowing > max_value_borrowing)[ii] == TRUE, 0, uniroot(function(cutoffValue){
        cutoff(cutoffValue, w, 0, 0, 0, sigma_Z1, sigma_Z2, sigma_Z3, sqrt(covZ1Z2), delta, alphaEQ)[3] - typeIerror_borrowing[ii]
      },lower = lower, upper = upper, tol = 1e-8, maxiter = 1e4)$root)})
      typeIerror_left = sapply(typeIerror_borrowing - max_value_borrowing, function(ii){ifelse(ii > 0, ii, 0)})
      
      typeIerror_nonborrowing = typeIerror_nonborrowing + typeIerror_left
      cutoffValue_nonborrowing = sapply(typeIerror_nonborrowing, function(ii) {
        uniroot(function(cutoffValue){
          cutoff(cutoffValue, w, 0, 0, 0, sigma_Z1, sigma_Z2, sigma_Z3, sqrt(covZ1Z2), delta, alphaEQ)[2] - ii
        },lower = lower, upper = upper, tol = 1e-8, maxiter = 1e4)$root
      })
    }
    else {
      cutoffValue_borrowing = sapply (typeIerror_borrowing, function(ii) {
        uniroot(function(cutoffValue){
          cutoff(cutoffValue, w, 0, 0, 0, sigma_Z1, sigma_Z2, sigma_Z3, sqrt(covZ1Z2), delta, alphaEQ)[3] - ii
        },lower = lower, upper = upper, tol = 1e-8, maxiter = 1e4)$root
      })
      
      cutoffValue_nonborrowing = sapply (typeIerror_nonborrowing, function(ii) {
        uniroot(function(cutoffValue){
          cutoff(cutoffValue, w, 0, 0, 0, sigma_Z1, sigma_Z2, sigma_Z3, sqrt(covZ1Z2), delta, alphaEQ)[2] - ii
        },lower = lower, upper = upper, tol = 1e-8, maxiter = 1e4)$root
      })
    }
    ############################## New ############################
    if (cutoff(qnorm (1-alpha_p/2)*sigma_Z1, w, 0, 0, 0, sigma_Z1, sigma_Z2, sigma_Z3, sqrt(covZ1Z2), delta, alphaEQ)[2] > alpha_p) { #code seems wired, but this is because floating error
      cutoffValue_new = Inf
    }
    else {
      cutoffValue_new = uniroot(function(cutoffValue){cutoff(cutoffValue, w, 0, 0, 0, sigma_Z1, sigma_Z2, sigma_Z3, sqrt(covZ1Z2), delta, alphaEQ)[3] - (alpha_p - cutoff(qnorm (1-alpha_p/2)*sigma_Z1, w, 0, Z2, -w*Z2, sigma_Z1, sigma_Z2, sigma_Z3, sqrt(covZ1Z2), delta, alphaEQ)[2])},
                                lower = lower, upper = upper, tol = 1e-8, maxiter = 1e4)$root
    }
    ### Common cutoff value (Approach 3) ###
    
    cutoffValue <- uniroot(function(cutoffValue){
      standardized_cutoff(cutoffValue, 0, 0, 0, sigma_Z1, sigma_Z2, sigma_Z3, sqrt(covZ1Z2), delta, alphaEQ)[1] - alpha_p # standardize
    },lower = -10, upper = 10, tol = 1e-8, maxiter = 1e4)$root
    
    ### Power prior (Bayesian approach) ###
    
    Z4 = Z1 - Z2
    var_Z4 = var_Z1 + var_Z2 - 2*covZ1Z2
    sigma_Z4 = sqrt(var_Z4)
    
    # sig0 is the prior variance and mu0 is the prior of shared mu
    deriv = logML(Z1, Z4, sigma_Z1, sigma_Z4, alpha = 1, sig0 = 1e2, mu0 = 0)[2]
    
    # Update alpha
    if(deriv > 0){
      alphahat <- 1
    }
    else {
      alphahat <- uniroot(function(alpha){
        logML(Z1, Z4, sigma_Z1, sigma_Z4, alpha, sig0 = 1e2, mu0 = 0)[2]
      },lower = 0,upper = 1,tol = 1e-8, maxiter = 1e4)$root
    }
    
    alphahat2 = optimize(logML2, interval = c(0, 1), X1 = Z1, sigma_X1 = sigma_Z1, X2 = Z4, sigma_X2 = sigma_Z4,  mu0 = 0, sigma0 = 1e2, maximum = TRUE, tol = .Machine$double.eps^0.5)$maximum
    alphahat3 = optimize(logML2, interval = c(0, 1), X1 = B1, sigma_X1 = sqrt(var_B1), X2 = B2, sigma_X2 = sqrt(var_B2),  mu0 = 0, sigma0 = 1e2, maximum = TRUE, tol = .Machine$double.eps^0.5)$maximum
    
    # the combined estimator
    w_hat = w*alphahat
    w_hat_tian = covZ1Z2/(covZ1Z2 + (var_Z2-covZ1Z2)/alphahat)
    w_hat2 = w*alphahat2
    w_hat2_tian = covZ1Z2/(covZ1Z2 + (var_Z2-covZ1Z2)/alphahat2)
    w_hat3 = covB1B2*alphahat3/var_B2
    w_hat3_tian = covB1B2/(covB1B2 + (var_B2-covB1B2)/alphahat3)
    estimator = (1-w_hat_tian)*Z1+w_hat_tian*Z4
    estimator2 = (1-w_hat2_tian)*Z1+w_hat2_tian*Z4
    estimator3 = (1-w_hat3_tian)*B1+w_hat3_tian*B2
    estimator.sd = sqrt((1-w_hat_tian)^2*sigma_Z1^2 + w_hat_tian^2*sigma_Z4^2 + 2*w_hat_tian*(1-w_hat_tian)*(sigma_Z1^2-covZ1Z2))
    estimator.sd2 = sqrt((1-w_hat2_tian)^2*sigma_Z1^2 + w_hat2_tian^2*sigma_Z4^2 + 2*w_hat2_tian*(1-w_hat2_tian)*(sigma_Z1^2-covZ1Z2))
    estimator.sd3 = sqrt((1-w_hat3_tian)^2*var_B1 + w_hat3_tian^2*var_B2 + 2*w_hat3_tian*(1-w_hat3_tian)*(covB1B2))
    
    res = rbind(res, cbind(data.frame(hr_trt_control = hr_trt_control,
                                      hr_control_RWD = hr_control_RWD,
                                      hr_borrow = hr_borrow,
                                      Z1 = Z1,
                                      Z2 = Z2,
                                      Z3 = Z3,
                                      Z4 = Z4,
                                      B1 = B1,
                                      B2 = B2,
                                      var_Z1 = var_Z1,
                                      var_Z2 = var_Z2,
                                      var_Z3 = var_Z3,
                                      var_Z4 = var_Z4,
                                      var_B1 = var_B1,
                                      var_B2 = var_B2,
                                      sigma_Z1 = sigma_Z1,
                                      sigma_Z2 = sigma_Z2,
                                      sigma_Z3 = sigma_Z3,
                                      sigma_Z4 = sigma_Z4,
                                      w = w,
                                      covZ1Z2 = covZ1Z2,
                                      covB1B2 = covB1B2,
                                      theta = theta,
                                      borrow = borrow,
                                      sigma_W = sigma_W,
                                      cutoffValue = cutoffValue,
                                      cutoffValue_new = cutoffValue_new,
                                      alphahat = alphahat,
                                      alphahat2 = alphahat2,
                                      alphahat3 = alphahat3,
                                      w_hat = w_hat,
                                      w_hat_tian = w_hat_tian,
                                      w_hat2 = w_hat2,
                                      w_hat2_tian = w_hat2_tian,
                                      w_hat3 = w_hat3,
                                      w_hat3_tian = w_hat3_tian,
                                      estimator = estimator,
                                      estimator2 = estimator2,
                                      estimator3 = estimator3,
                                      estimator.sd = estimator.sd,
                                      estimator.sd2 = estimator.sd2,
                                      estimator.sd3 = estimator.sd3,
                                      n_patients_trt = dataentry$n_patients_trt,
                                      n_patients_control = dataentry$n_patients_control,
                                      n_events_trt = dataentry$n_events_trt,
                                      n_events_control = dataentry$n_events_control,
                                      n_events_RWD = dataentry$n_events_RWD
    ),
    matrix(cutoffValue_nonborrowing, ncol = length(cutoffValue_nonborrowing), dimnames = list(NULL,paste0("cv_nonborrowing_", v))),
    matrix(cutoffValue_borrowing,    ncol = length(cutoffValue_borrowing),    dimnames = list(NULL,paste0("cv_borrowing_", v)))
    ))
  }
  
  pBorrowing <- sum(res$borrow)/nrow(res)
  type1Error = (sum (res$Z1/sqrt(res$var_Z1) < qnorm(alpha_p/2)) + sum (res$Z1/sqrt(res$var_Z1) > qnorm(1 - alpha_p/2))) / nrow(res)
  type1Error_n = (sum(ifelse(res$borrow, res$Z3/sqrt(res$var_Z3), res$Z1/sqrt(res$var_Z1))  < qnorm (alpha_p/2)) + sum(ifelse(res$borrow, res$Z3/sqrt(res$var_Z3), res$Z1/sqrt(res$var_Z1)) > qnorm (1 - alpha_p/2)))/nrow(res)
  type1Error_normal = (sum (ifelse(res$borrow, res$Z3, res$Z1)/res$sigma_W < qnorm(alpha_p/2)) + sum (ifelse(res$borrow, res$Z3, res$Z1)/res$sigma_W > qnorm(1 - alpha_p/2)))/ nrow(res)
  type1Error_s <- sapply (seq(0,1,0.1), function (ii){
    (sum(ifelse(res$borrow, res$Z3 < -res[, paste0("cv_borrowing_", ii)], res$Z1 < -res[, paste0("cv_nonborrowing_", ii)])) + sum(ifelse(res$borrow, res$Z3 > res[, paste0("cv_borrowing_", ii)], res$Z1 > res[, paste0("cv_nonborrowing_", ii)])))/nrow(res)
  })
  type1Error_c = (sum(ifelse(res$borrow, res$Z3/res$sigma_Z3, res$Z1/res$sigma_Z1) < -res$cutoffValue) + sum(ifelse(res$borrow, res$Z3/res$sigma_Z3, res$Z1/res$sigma_Z1) > res$cutoffValue))/nrow(res)
  type1Error_b = (sum (res$estimator/res$estimator.sd < qnorm(alpha_p/2)) + sum (res$estimator/res$estimator.sd > qnorm(1 - alpha_p/2))) / nrow(res)
  type1Error_b2 = (sum (res$estimator2/res$estimator.sd2 < qnorm(alpha_p/2)) + sum (res$estimator2/res$estimator.sd2 > qnorm(1 - alpha_p/2))) / nrow(res)
  type1Error_b3 = (sum (res$estimator3/res$estimator.sd3 < qnorm(alpha_p/2)) + sum (res$estimator3/res$estimator.sd3 > qnorm(1 - alpha_p/2))) / nrow(res)
  type1Error_new = (sum(ifelse(res$borrow, res$Z3 < -res$cutoffValue_new, res$Z1 < qnorm (alpha_p/2)*sqrt(res$var_Z1))) + sum(ifelse(res$borrow, res$Z3 > res$cutoffValue_new, res$Z1 > qnorm (1-alpha_p/2)*sqrt(res$var_Z1))))/nrow(res)
  return (list(res = res,
               pBorrowing = pBorrowing,
               type1Error = type1Error,
               type1Error_n = type1Error_n,
               type1Error_normal = type1Error_normal,
               type1Error_s = type1Error_s,
               type1Error_c = type1Error_c,
               type1Error_b = type1Error_b,
               type1Error_b2 = type1Error_b2,
               type1Error_b3 = type1Error_b3,
               type1Error_new = type1Error_new
               
  ))
}

Simulation = function (dataset, 
                       HR_trt_control, 
                       HR_control_RWD, 
                       ratio, 
                       criteria,
                       form,
                       stime.var = "STIME",
                       status.var = "STATUS",
                       nsim, seed = NULL) {
  
  # Set seed
  if (is.null(seed) == FALSE) {
    set.seed(seed)
  }
  
  # Data manipulation
  dataset$ID <- paste0 ("ID", 1:nrow (dataset))
  dataset$EE <- factor (dataset$EE)
  dataset$SEX <- as.numeric (dataset$SEX == "M")
  dataset$ONSET <- as.numeric (dataset$ONSET == "B")
  dataset$TRT = dataset$CONTROL = dataset$ENROLL = 0
  dataset$RWD = 1
  
  # Fit cox model on the registry data
  coxModel = coxph(Surv (STIME, STATUS) ~ AGE + I (EE == 'DEF') + DISDUR + FVC + TOTAL + exp (SLOPE) + BMI, x = TRUE, data = dataset, method = "breslow")
  time = basehaz(coxModel, centered = FALSE)$time
  cumulativeBaselineHazard = data.frame(CH = basehaz(coxModel, centered = FALSE)$hazard, Time = time)
  baselineSurvival = data.frame(Prob = c(1, exp(-cumulativeBaselineHazard$CH)), Time = c(0, time))
  baselineCDF = data.frame(Prob = 1 - baselineSurvival$Prob, Time = c(0, time))
  Spline = stats::splinefun(x = baselineSurvival$Time, y = baselineSurvival$Prob, method = "hyman")
  baselineSurvival = data.frame(Prob = Spline(0:max(time)), Time = 0:max(time))
  
  # Get the exponential of linear predictor given the covariates
  hazardRatioRegistry = predict(coxModel, newdata = dataset, type = "risk", reference = "zero")
  # Get the survival curve of each patient
  survivalCurveRegistry = t(sapply(1:nrow(dataset), function(ii){baselineSurvival$Prob^hazardRatioRegistry[ii]}))
  colnames(survivalCurveRegistry) = 0:max(time)
  rownames(survivalCurveRegistry) = 1:nrow(dataset)
  
  survivalTimeRegistry = apply(survivalCurveRegistry, 1, FUN=function(x){
    z <- diff(x < runif(1))
    r <- ifelse(all(z==0), max(time), which.max(z))
    return(r)
  })
  
  dataset$STIME  = survivalTimeRegistry
  dataset$STATUS = ifelse(survivalTimeRegistry == max(time), 0, 1)
  p_trt = 1 - baselineSurvival$Prob[max(time)]^(HR_trt_control*HR_control_RWD)
  p_control = 1 - baselineSurvival$Prob[max(time)]^HR_control_RWD
  nsample_trt = ceiling(100/p_trt)
  nsample_control = ceiling(100/p_control)
  
  res = c()
  for (i in 1:nsim) {
    # Sample patients from the registry
    sampling = sample(nrow(dataset), nsample_trt + nsample_control, replace = FALSE)
    trtPatients = sample(sampling, nsample_trt, replace = FALSE)
    controlPatients = setdiff(sampling, trtPatients)
    
    # Treatment allocation
    trtSet = dataset[trtPatients, ]
    controlSet = dataset[controlPatients, ]
    trtSet$TRT = 1
    trtSet$CONTROL = 0
    controlSet$TRT = 0
    controlSet$CONTROL = 1
    trialSet = rbind(trtSet, controlSet)
    trialSet$ENROLL <- 1
    trialSet$RWD <- 0
    
    # Add treatment effect and cohort effect
    hazardRatioTrial = ifelse(trialSet$TRT == 1, HR_trt_control*HR_control_RWD, HR_control_RWD)*predict(coxModel, newdata = trialSet, type = "risk", reference = "zero")
    
    # Similar as the code from 366th row to 395th row 
    survivalCurveTrial = t(sapply(1:nrow(trialSet), function(ii){ baselineSurvival$Prob^hazardRatioTrial[ii]}))
    colnames(survivalCurveTrial) = 0:max(time)
    rownames(survivalCurveTrial) = 1:nrow(trialSet)
    
    survivalTimeTrial = apply(survivalCurveTrial, 1, FUN=function(x){
      z <- diff(x < runif(1))
      r <- ifelse(all(z==0), max(time), which.max(z))
      return(r)
    })
    
    trialSet$STIME  = survivalTimeTrial
    trialSet$STATUS = ifelse(survivalTimeTrial == max(time), 0, 1)
    
    registrySet <- dataset[!dataset$ID %in% trialSet$ID, ]
    registrySet <- registrySet[Eligible (criteria = criteria, data = registrySet), ]
    registrySet$TRT = registrySet$ENROLL = 0
    finalSet <- rbind (trialSet, registrySet)
    
    #. Get PS score
    m <- glm (form, data = finalSet, family = "binomial")
    finalSet$PS <- fitted (m)
    
    #. Get matches
    R <- Matcher.v2 (finalSet)
    
    #. Combined dataset
    data.match <- finalSet[finalSet$ID %in% c (R$MatchID, R$TrialID), ]
    
    # #. Sub-dataset
    # test.TRT.CONTROL = sum(data.match[data.match$TRT == 1, "STATUS"] == 1)
    # test.CONTROL.RWD = data.match[data.match$TRT == 0, ]
    # 
    # autoplot(survfit(as.formula (paste0 ("Surv (", stime.var, ", ", status.var, ") ~ ENROLL + TRT")), data = data.match))
    
    # Model fitting
    
    # cox proportional model
    m = coxph(as.formula (paste0 ("Surv (", stime.var, ", ", status.var, ") ~ TRT + ENROLL")), data = data.match)
    m2 = coxph(as.formula (paste0 ("Surv (", stime.var, ", ", status.var, ") ~ CONTROL + RWD")), data = data.match)
    # m.TRT.CONTROL = coxph(as.formula (paste0 ("Surv (", stime.var, ", ", status.var, ") ~ TRT")), data = test.TRT.CONTROL)
    # m.CONTROL.RWD = coxph(as.formula (paste0 ("Surv (", stime.var, ", ", status.var, ") ~ ENROLL")), data = test.CONTROL.RWD)
    # m.BORROW = coxph(as.formula (paste0 ("Surv (", stime.var, ", ", status.var, ") ~  TRT")), data = data.match)
    
    # Get coefficients for cox model
    Z1 = as.numeric(coef(m)["TRT"])
    Z2 = as.numeric(-coef(m)["ENROLL"])
    # Z3 = as.numeric(coef(m.BORROW)["TRT"])
    hr_trt_control = exp(Z1)
    hr_control_RWD = exp(-Z2)
    # hr_borrow = exp(Z3)
    
    # Get variance and standard deciation for cox model
    var_Z1 = vcov(m)["TRT", "TRT"]
    var_Z2 = vcov(m)["ENROLL", "ENROLL"]
    # var_Z3 = vcov(m.BORROW)["TRT", "TRT"]
    sigma_Z1 = sqrt(var_Z1)
    sigma_Z2 = sqrt(var_Z2)
    # sigma_Z3 = sqrt(var_Z3)
    
    #. Covariance and Borrowing weight
    covZ1Z2 = -vcov(m)["TRT", "ENROLL"]
    w = covZ1Z2/sigma_Z2^2
    
    #. Z3 
    Z3 = Z1 - w*Z2
    var_Z3 = var_Z1 + w^2*var_Z2 - 2*w*covZ1Z2
    sigma_Z3 = sqrt(var_Z3)
    hr_borrow = exp(Z3)
    
    #n_events
    n_events_trt = sum(data.match[data.match$TRT == 1, "STATUS"] == 1)
    n_events_control = sum(data.match[data.match$TRT == 0 & data.match$ENROLL == 1, "STATUS"] == 1)
    n_events_RWD = sum(data.match[data.match$ENROLL == 0, "STATUS"] == 1)
    
    # log(hr_CONTROL) - log(hr_TRT)
    B1 = as.numeric(coef(m2)["CONTROL"])
    var_B1 = vcov(m2)["CONTROL", "CONTROL"]
    # log(hr_RWD) - log(hr_TRT)
    B2 = as.numeric(coef(m2)["RWD"])
    var_B2 = vcov(m2)["RWD", "RWD"]
    covB1B2 = vcov(m2)["CONTROL", "RWD"]
    
    
    res = rbind(res, c(hr_trt_control, hr_control_RWD, hr_borrow, Z1, Z2, Z3, B1, B2, sigma_Z1, sigma_Z2, sigma_Z3, var_B1, var_B2, covZ1Z2, covB1B2, w, nrow(trtSet), nrow(controlSet), n_events_trt, n_events_control, n_events_RWD))
  }
  res = as.data.frame(res)
  colnames(res) = c("hr_trt_control", "hr_control_RWD", "hr_borrow", "Z1", "Z2", "Z3", "B1", "B2", "sigma_Z1", "sigma_Z2", "sigma_Z3", "var_B1", "var_B2", "covZ1Z2", "covB1B2", "w", "n_patients_trt", "n_patients_control", "n_events_trt", "n_events_control", "n_events_RWD")
  return(res)
}

##############################
########## 1. Data ###########
library("openxlsx")
library("parallel")
library("mvtnorm")
library("survival")
D <- read.xlsx ("data_rwe.xlsx")
#. Eligible population:
Criteria <- list ("DISDUR <= 36", 
                  "AGE >= 18", 
                  "AGE <= 85", 
                  "FVC >= 70")

#. Function for PS matching:
form <- as.formula (ENROLL ~  AGE + I (EE == 'DEF') + DISDUR + FVC + TOTAL + exp (SLOPE) + BMI)

args= commandArgs(trailingOnly = TRUE)

logHR_trt_control = as.numeric(args[1])
logHR_control_RWD = as.numeric(args[2])
iter = as.numeric(args[3])

data = Simulation(dataset=D, 
                  HR_trt_control = exp(logHR_trt_control), 
                  HR_control_RWD = exp(logHR_control_RWD), 
                  ratio=1, 
                  criteria=Criteria, 
                  form=form,
                  nsim = 500,
                  seed = iter)
res = inference(dataset = data, alpha_p = 0.05, delta = 0.3, alphaEQ = 0.2)                  
                 
fileName = paste0 ("Results_New_noZ2_tian/Results_", logHR_trt_control, "_", logHR_control_RWD, "_", iter, ".Rds")
saveRDS (res, fileName)