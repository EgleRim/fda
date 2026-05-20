rm(list = ls())

library(fda)
library(fda.usc)
library(fdaoutlier)
library(rainbow)

#getwd()
#setwd("/Users/aironas.vinickas/Downloads")
orig_df <- readRDS("Eurozone_10Y_Yields_Monthly.rds")

yields_fd <- readRDS("yield_changes_fd.rds")
plot(yields_fd)

#  elementary pointwise mean and standard deviation

mean_yield = mean.fd(yields_fd)
sd_yield = std.fd(yields_fd)

lines(mean_yield, lwd=4, lty=2, col="red")
lines(sd_yield, lwd=4, lty=2, col="blue")

lines(mean_yield-sd_yield, lwd=4, lty=2, col="pink")
lines(mean_yield+sd_yield, lwd=4, lty=2, col="pink")

# Section 6.1.1 The Bivariate Covariance Function v(s; t)

yields_bifd = var.fd(yields_fd)
monthtime = seq(1,12,length=50)
yields_fd_mat  = eval.bifd(monthtime, monthtime, yields_bifd)

# Figure 6.1

persp(monthtime, monthtime, yields_fd_mat,
      theta=-45, phi=25, r=3, expand = 0.5,
      ticktype='detailed',
      xlab="Month",
      ylab="Month",
      zlab="Covariance")

contour(monthtime, monthtime, yields_fd_mat,
        xlab="Month",
        ylab="Month")

# PCA
opar <- par(mfrow=c(2,2))
nharm = 4
pcalist = pca.fd(yields_fd, nharm, centerfns = TRUE)
plot(pcalist)
par(opar)

opar <- par(mfrow=c(1,1))
plot(pcalist$harmonics)
par(opar)


# pca graphs explained
c <- 2
mn <- pcalist$meanfd
phi <- pcalist$harmonics[1]
lambda <- pcalist$values[1]

f1 <- mn - c*sqrt(lambda)*phi
f2 <- mn + c*sqrt(lambda)*phi

opar <- par(mfrow=c(1,2))
plot(pcalist, harm = 1)
plot(mn, ylim = c(-20,40))
lines(f1, col = 2)
lines(f2, col = 3)
par(opar)

# PCA restore the original curves
fd.pca1.list <- list() 
fd.pca2.list <- list() 
fd.pca3.list <- list() 
fd.pca4.list <- list() 

for(i in 1:5) {
  fd.pca1.list[[i]] <- mean.fd(yields_fd) + 
    pcalist$scores[i,1]*pcalist$harmonics[1]
  
  fd.pca2.list[[i]] <- mean.fd(yields_fd) + 
    pcalist$scores[i,1]*pcalist$harmonics[1] + 
    pcalist$scores[i,2]*pcalist$harmonics[2]
  
  fd.pca3.list[[i]]<- mean.fd(yields_fd) +
    pcalist$scores[i,1]*pcalist$harmonics[1] + 
    pcalist$scores[i,2]*pcalist$harmonics[2] +
    pcalist$scores[i,3]*pcalist$harmonics[3] 
  
  fd.pca4.list[[i]]<- mean.fd(yields_fd) +
    pcalist$scores[i,1]*pcalist$harmonics[1] + 
    pcalist$scores[i,2]*pcalist$harmonics[2] +
    pcalist$scores[i,3]*pcalist$harmonics[3] +
    pcalist$scores[i,4]*pcalist$harmonics[4]
}

#### Rotation
opar <- par(mfrow=c(2,2))
varmx <- varmx.pca.fd(pcalist)
plot(varmx)
par(opar)

plot(varmx$harmonics)

plotscores(varmx, loc = 5)


# PCA restore the original curves
fd.vrm1.list <- list() 
fd.vrm2.list <- list() 
fd.vrm3.list <- list() 
fd.vrm4.list <- list() 

for(i in 1:5) {
  fd.vrm1.list[[i]] <- mean.fd(yields_fd) + 
    varmx$scores[i,1]*varmx$harmonics[1]
  
  fd.vrm2.list[[i]] <- mean.fd(yields_fd) +
    varmx$scores[i,1]*varmx$harmonics[1] + 
    varmx$scores[i,2]*varmx$harmonics[2]
  
  fd.vrm3.list[[i]]<- mean.fd(yields_fd) +
    varmx$scores[i,1]*varmx$harmonics[1] + 
    varmx$scores[i,2]*varmx$harmonics[2] +
    varmx$scores[i,3]*varmx$harmonics[3] 
  
  fd.vrm4.list[[i]]<- mean.fd(yields_fd) +
    varmx$scores[i,1]*varmx$harmonics[1] + 
    varmx$scores[i,2]*varmx$harmonics[2] +
    varmx$scores[i,3]*varmx$harmonics[3] +
    varmx$scores[i,4]*varmx$harmonics[4]
}

opar <- par(mfrow=c(2,2), ask = TRUE)
for(i in 1:5) {
  plot(fd.vrm1.list[[i]], ylim=c(-20, 40), ylab = "1 PC")
  lines(yields_fd[i], col = 2)
  
  plot(fd.vrm2.list[[i]], ylim=c(-20, 40), ylab = "2 PC")
  lines(yields_fd[i], col = 2)
  
  plot(fd.vrm3.list[[i]], ylim=c(-20, 40), ylab = "3 PC")
  lines(yields_fd[i], col = 2)
  
  plot(fd.vrm4.list[[i]], ylim=c(-20, 40), ylab = "4 PC")
  lines(yields_fd[i], col = 2)
}
par(opar)

#----------------------------------------
# Functional depth, boxplots and outliers
tt <- 1:59

#evaluating functional curves
yields_matrix <- eval.fd(tt, yields_fd)

#converting to fdata object
yields_fdata <- fdata(t(yields_matrix), tt)

dates <- as.Date(orig_df$Date)

#Fraiman-Muniz depth
out.FM <- depth.FM(yields_fdata, trim = 0.1, draw = FALSE)

par(lwd = 2)

#had to manually plot because draw = TRUE in depth.FM is not very aesthetic and legend is huge
plot(yields_fdata, col = "grey70", lwd = 1, main = "Fraiman–Muniz Functional Depth",
     xlab = "Month",ylab = "Yield change", xaxt = "n"
)

axis(1, at = seq(1, 59, by = 6), labels = format(dates[seq(1, 59, by = 6)], "%Y-%m"), cex.axis = 0.85)

lines(out.FM$median, col = "red", lwd = 3)
lines(out.FM$mtrim, col = "blue", lwd = 3, lty = 2)

legend("topright", legend = c("Median curve", "Trimmed mean (10%)"),
  col = c("red", "blue"), lwd = 3, lty = c(1,2), cex = 1, bty = "n"
)

#gray curves are individual country curves, each line represents  yield change trajectory for each country
#red line is functional median, most central trajectory among all countries, a typical eurozone yield change pattern

#blue line is 10% trimmed mean curve, where 10% of the most extreme curves are removed and mean trajectory is 
#recomputed, it is meant for robust central tendency visualization

#what we can see is that red and blue curves have a very similar pattern and most grey curves follows the pattern
#this suggests that eurozone 10Y government bond yields tend to move together across countries
#this is due to ECB policy that affects all eurozone countries, inflation shocks usually affect the entire region 

#Modal depth
out.mode <- depth.mode(yields_fdata, trim = 0.1, draw = TRUE)
plot(yields_fdata, col = "grey70", lwd = 1, main = "Modal Functional Depth",
     xlab = "Month", ylab = "Yield change", xaxt = "n"
)

axis(1, at = seq(1, 59, by = 6), labels = format(dates[seq(1, 59, by = 6)], "%Y-%m"), cex.axis = 0.85)

lines(out.mode$median, col = "red", lwd = 3)
lines(out.mode$mtrim, col = "blue", lwd = 3, lty = 2)

legend("topright", legend = c("Median curve", "Trimmed mean (10%)"),
       col = c("red", "blue"), lwd = 3, lty = c(1,2), cex = 1, bty = "n"
)

#Random Projection depth
out.RP <- depth.RP(yields_fdata, trim = 0.1, draw = TRUE)
plot(yields_fdata, col = "grey70", lwd = 1, main = "Random Projection Functional Depth",
     xlab = "Month", ylab = "Yield change", xaxt = "n"
)

axis(1, at = seq(1, 59, by = 6), labels = format(dates[seq(1, 59, by = 6)], "%Y-%m"), cex.axis = 0.85)

lines(out.RP$median, col = "red", lwd = 3)
lines(out.RP$mtrim, col = "blue", lwd = 3, lty = 2)

legend("topright", legend = c("Median curve", "Trimmed mean (10%)"),
       col = c("red", "blue"), lwd = 3, lty = c(1,2), cex = 1, bty = "n"
)

#basically all three plots show similar pattern
#Fraiman-Muniz depth evaluates centrality pointwise over time
#median curve and trimmed mean curve are very close to each other
#this indicates low influence of extreme curves and a stable central structure

#Modal functional depth identifies the curve located in the highest density region of the functional space
#Modal median curve is very similar to FM, but trimmed mean is slightly higher around 2022 peak, a group of curves
#had stronger positive yield changes and a strong increase in yields during 2022 which is followed by decline later

#Random projection depth evaluates centrality by projecting curves onto many random directions
#central curve remains very similar to the other methods, which suggests the central structure is robust to the choice of depth definition
#some curves represent potential outliers 

#converting to fds
dt <- yields_matrix

colnames(dt) <- colnames(orig_df[, -1]) 
rownames(dt) <- tt

yld <- fds(x = tt, y = dt, xname = "Month", yname = "Yield change")

#functional bagplot based on Tukey halfspace depth
fboxplot(yld, plot.type = "functional", type = "bag", projmethod = "PCAproj")
title("Functional Bagplot")

#median curve is the deepest curve
#bag is the region containing 50% of curves
#fence is inflated bag (~99% coverage)
#curves outside the fence - outliers -> Greece and Estonia
#central region is defined by geometric depth
#median

#functional highest density regions (HDR) plot
fboxplot(yld, plot.type = "functional", type = "hdr", alpha = c(0.05, 0.5), projmethod = "PCAproj")
title("Functional HDR Boxplot")
#modal curve - curve with the highest density
#50% HDR region - most probable curves
#outer HDR region - larger probability region
#outside curves - outliers -> Greece
#central region is defined by density estimation
#mode


foutliers(yld, method = "robMah")
foutliers(yld, method = "lrt")
foutliers(yld, method = "depth.trim")
foutliers(yld, method = "depth.pond")
foutliers(yld, method = "HUoutliers")



#----------------------------------------------------------------------------
#Hypothesis testing
library(fda)
library(ggplot2)
library(tidyr)
library(dplyr)


#Trace function
trace <- function(data) {
  tr <- sum(diag(data))
  return(tr)
}



#two sample tests
#pointwise test
Ztwosample <- function(x, y, t.seq, alpha = 0.05) {
  # Check input types
  if (!inherits(x, "fd")) stop("x must be an 'fd' object")
  if (!inherits(y, "fd")) stop("y must be an 'fd' object")
  # Sample sizes
  n <- dim(x$coef)[2]
  m <- dim(y$coef)[2]
  k <- length(t.seq)
  # Mean functions and their difference
  mu.x <- mean.fd(x)
  mu.y <- mean.fd(y)
  delta <- mu.x - mu.y
  delta.t <- eval.fd(t.seq, delta)
  # Centered data
  z.x <- center.fd(x)
  z.y <- center.fd(y)
  # Evaluate centered functions at time points
  z.x.t <- eval.fd(t.seq, z.x)
  z.y.t <- eval.fd(t.seq, z.y)
  z.t <- cbind(z.x.t, z.y.t)
  # Pooled variance estimator
  pooled.df <- n + m - 2
  if (pooled.df > k) {
    Sigma <- crossprod(z.t) / pooled.df
  } else {
    Sigma <- tcrossprod(z.t) / pooled.df
  }
  gamma.t <- diag(Sigma)
  # Z-statistic
  Zpointwise <- sqrt((n * m) / (n + m)) * delta.t / sqrt(gamma.t)
  # Critical value for two-sided t-test
  crit.val <- qt(1 - alpha / 2, df = pooled.df)
  # Plot
  plot(t.seq, Zpointwise, type = "l", lwd = 2, col = "darkred",
       xlab = "Time", ylab = "Z statistic",
       main = "Pointwise Two-Sample t-Test",
       ylim = range(c(Zpointwise, -crit.val, crit.val)) * 1.1)
  abline(h = c(crit.val, -crit.val), col = "blue", lty = 2, lwd = 2)
  abline(h = 0, col = "gray", lty = 3)
  legend("topright", legend = c("Z statistic", "Critical values"),
         col = c("darkred", "blue"), lty = c(1, 2), lwd = 2, bty = "n")
  # Output
  return(list(
    statistics.pointwise = Zpointwise,
    critical.value = crit.val,
    alpha = alpha,
    time.seq = t.seq
  ))
}

#bootstarp
bootstrap_FDA <- function(core_fd, per_fd, t.seq, B = 1000) {
  all_coefs <- cbind(core_fd$coefs, per_fd$coefs)
  all_fd <- core_fd
  all_fd$coefs <- all_coefs
  n_core <- dim(core_fd$coefs)[2]
  T_boot <- numeric(B)
  
  for (b in 1:B) {
    perm <- sample(ncol(all_fd$coefs))
    fd_b <- all_fd
    fd_b$coefs <- all_fd$coefs[, perm]
    
    core_b <- fd_b; core_b$coefs <- fd_b$coefs[, 1:n_core]
    per_b  <- fd_b; per_b$coefs  <- fd_b$coefs[, (n_core+1):ncol(fd_b$coefs)]
    
    res_b      <- Ztwosample(core_b, per_b, t.seq)
    T_boot[b]  <- max(abs(res_b$statistics.pointwise))
  }
  return(T_boot)
}




#f based test
F.stat.twosample <- function(x, y, t.seq, alpha=0.05, method=1:2, replications=100) {
  # method = 1: T stat
  # method = 2: Bootsrap
  if(class(x) != "fd") stop("X must be fd object")
  if(class(y) != "fd") stop("Y must be fd object")
  
  mu.x <- mean.fd(x)
  mu.y <- mean.fd(y)
  
  n <- dim(x$coefs)[2]
  m <- dim(y$coefs)[2]
  
  k <- length(t.seq)
  
  cn <- (n*m)/(n+m)
  delta <- (mu.x - mu.y)
  delta.t <- eval.fd(t.seq, delta)
  
  z.x <- center.fd(x)
  z.y <- center.fd(y)
  
  z.x.t <- eval.fd(t.seq, z.x)
  z.y.t <- eval.fd(t.seq, z.y)
  z.t <- cbind(z.x.t, z.y.t)
  
  if(n > k | m > k) {
    Sigma <- (t(z.t) %*% z.t)/(n-2)
  } else {
    Sigma <- (z.t %*% t(z.t))/(n-2)
  }
  
  A <- trace(Sigma)
  B <- trace(Sigma^2)
  
  Fstat <- (cn * t(delta.t) %*% delta.t)/A
  Fstat <- Fstat[1]
  
  btFstat <-  numeric(replications)
  
  if(method == 1) { #naive method
    kappa <- A^2/B
    pvalue <- 1-pf(Fstat, kappa, (n-2)*kappa)
    params <- list(df1 = kappa, df2 = (n-2)*kappa)
  } 
  if(method == 2) {  #bootstrapping method
    for(i in 1:replications) {
      rep1 <- sample(1:n, n, replace = TRUE)
      xstar.coefs <- x$coefs[,rep1]
      xstar.names <- x$fdnames
      xstar.names$reps <- rep1
      xstar.fd <- fd(xstar.coefs, x$basis, xstar.names)
      
      rep2 <- sample(1:m, m, replace = TRUE)
      ystar.coefs <- y$coefs[,rep2]
      ystar.names <- y$fdnames
      ystar.names$reps <- rep2
      ystar.fd <- fd(ystar.coefs, y$basis, ystar.names)
      
      mu.x.star <- mean.fd(xstar.fd)
      mu.y.star <- mean.fd(ystar.fd)
      delta.star <- (mu.x.star - mu.y.star)
      delta.star.t <- eval.fd(t.seq, delta.star)
      
      bt.z.x <- center.fd(xstar.fd)
      bt.z.y <- center.fd(ystar.fd)
      
      bt.z.x.t <- eval.fd(t.seq, bt.z.x)
      bt.z.y.t <- eval.fd(t.seq, bt.z.y)
      z.star.t <- cbind(bt.z.x.t, bt.z.y.t)
      
      if(n > k | m > k) {
        btSigma <- (t(z.star.t) %*% z.star.t)/(n-2)
      } else {
        btSigma <- (z.star.t %*% t(z.star.t))/(n-2)
      }
      
      btmu <- apply(delta.star.t,1,mean) - delta.t
      btFstat[i] <- (cn * t(btmu) %*% btmu)/trace(btSigma)
    }
    pvalue <- mean(btFstat>=Fstat)
    params <- list(btFstat)
  }
  return(list(statistics = Fstat, pvalue = pvalue, params=params))
}


# # Two sample L2 norma based bootstrap test
L2.stat.twosample <- function(x, y, t.seq, alpha=0.05, method=1:2, replications=100) {
  # method = 1: T stat
  # method = 2: Bootsrap
  if(class(x) != "fd") stop("X must be fd object")
  if(class(y) != "fd") stop("Y must be fd object")
  
  mu.x <- mean.fd(x)
  mu.y <- mean.fd(y)
  
  n <- dim(x$coefs)[2]
  m <- dim(y$coefs)[2]
  
  k <- length(t.seq)
  
  cn <- (n*m)/(n+m)
  delta <- (mu.x - mu.y)
  delta.t <- eval.fd(t.seq, delta)
  
  z.x <- center.fd(x)
  z.y <- center.fd(y)
  
  z.x.t <- eval.fd(t.seq, z.x)
  z.y.t <- eval.fd(t.seq, z.y)
  z.t <- cbind(z.x.t, z.y.t)
  
  if(n > k | m > k) {
    Sigma <- (t(z.t) %*% z.t)/(n-2)
  } else {
    Sigma <- (z.t %*% t(z.t))/(n-2)
  }
  
  A <- trace(Sigma)
  B <- trace(Sigma^2)
  
  L2stat <- cn * t(delta.t) %*% delta.t
  L2stat <- L2stat[1]
  
  btL2stat <-  numeric(replications)
  
  if(method == 1) { #naive method
    A2 <- A^2
    B2 <- B
    alp <- B2/A
    df <- A2/B2
    pvalue <- 1-pchisq(L2stat/alp, df)
    params <- list(alpha = alp, df = df)
  } 
  if(method == 2) {  #bootstrapping method
    for(i in 1:replications) {
      rep1 <- sample(1:n, n, replace = TRUE)
      xstar.coefs <- x$coefs[,rep1]
      xstar.names <- x$fdnames
      xstar.names$reps <- rep1
      xstar.fd <- fd(xstar.coefs, x$basis, xstar.names)
      
      rep2 <- sample(1:m, m, replace = TRUE)
      ystar.coefs <- y$coefs[,rep2]
      ystar.names <- y$fdnames
      ystar.names$reps <- rep2
      ystar.fd <- fd(ystar.coefs, y$basis, ystar.names)
      
      mu.x.star <- mean.fd(xstar.fd)
      mu.y.star <- mean.fd(ystar.fd)
      delta.star <- (mu.x.star - mu.y.star)
      delta.star.t <- eval.fd(t.seq, delta.star)
      
      btmu <- apply(delta.star.t,1,mean) - delta.t
      btL2stat[i] <- cn * t(btmu) %*% btmu
    }
    pvalue <- mean(btL2stat>=L2stat)
    params <- list(boot.stat=btL2stat)
  }
  return(list(statistics = L2stat, pvalue = pvalue, params=params))
}


#-----------------------------------------------------
#Hypotheses

#1)

#H0: there is no diference in mean yield functions 
#between core and peripheral countries 
#(Mean yield curve of core = mean yield curve of peripheral)

#H1: there is diference in mean yield functions 
#between the two groups.

#Core countries: Germany, France, Netherlands, Austria, Belgium
#Peripheral: Italy, Greece, Portugal, etc.

#split data to two groups
core_names <- c("Germany","France","Netherlands","Austria","Belgium")
per_names <- c("Italy", "Greece", "Portugal", "Cyprus", "Ireland",
               "Latvia", "Lithuania", "Malta", "Slovakia", 
               "Estonia", "Finland")

core_idx <- which(colnames(yields_fd$coefs) %in% core_names)
per_idx  <- which(colnames(yields_fd$coefs) %in% per_names)
core_fd <- yields_fd
core_fd$coefs <- yields_fd$coefs[, core_idx]

per_fd <- yields_fd
per_fd$coefs <- yields_fd$coefs[, per_idx]

t.seq <- seq(1, 59, length.out = 300)



#f stat test
result2 <- F.stat.twosample(core_fd, per_fd, t.seq, method = 1, replications = 5000)
result2$statistics
result2$pvalue

if(result2$pvalue < 0.05){
  print("Reject H0: Core and peripheral yield functions differ")
} else {
  print("Fail to reject H0")
}
#H0 rejected

#L2 method
# 1) — chi-squared approximation (analytical)
result_L2_m1 <- L2.stat.twosample(core_fd, per_fd, t.seq, 
                                  method = 1, 
                                  replications = 1000)

# 2) — bootstrap
result_L2_m2 <- L2.stat.twosample(core_fd, per_fd, t.seq, 
                                  method = 2, 
                                  replications = 1000)

# View results
result_L2_m1$statistics
result_L2_m1$pvalue

result_L2_m2$statistics
result_L2_m2$pvalue



#2)

#H0: there is no diference in mean yield functions 
#between two groups 
#(Mean yield curve of high credit rating = mean yield curve of low credit rating)

#H1: there is diference in mean yield functions 
#between the two groups.


#2 groups based on credit rating:
# High grade: AAA + AA
high_names <- c("Germany", "Netherlands", "Finland", 
                "Austria", "Belgium", "Ireland")

# Lower grade: A and below
lower_names <- c("France", "Portugal", "Slovakia", "Estonia",
                 "Latvia", "Lithuania", "Malta", "Cyprus",
                 "Italy", "Greece")


high_idx <- which(colnames(yields_fd$coefs) %in% high_names)
lower_idx  <- which(colnames(yields_fd$coefs) %in% lower_names)
high_fd <- yields_fd
high_fd$coefs <- yields_fd$coefs[, high_idx]

lower_fd <- yields_fd
lower_fd$coefs <- yields_fd$coefs[, lower_idx]

t.seq <- seq(1, 59, length.out = 300)


#f stat test
result2 <- F.stat.twosample(high_fd, lower_fd, t.seq, method = 1, replications = 1000)
result2$statistics
result2$pvalue

if(result2$pvalue < 0.05){
  print("Reject H0: High and low credit rating country yield functions differ")
} else {
  print("Fail to reject H0")
}
#H0 rejected again

#L2 method
# 1) — chi-squared approximation (analytical)
result_L2_m1 <- L2.stat.twosample(high_fd, lower_fd, t.seq, 
                                  method = 1, 
                                  replications = 1000)

# 2) — bootstrap
result_L2_m2 <- L2.stat.twosample(high_fd, lower_fd, t.seq, 
                                  method = 2, 
                                  replications = 1000)

# View results
result_L2_m1$statistics
result_L2_m1$pvalue

result_L2_m2$statistics
result_L2_m2$pvalue



#3)

#change point analysis
#H0: The mean yield change function is stable over time
#H1: There exists at least one structural break in the functional mean

library(fda)
library(ecp)
library(EnvCpt)
library(changepoint)

t.seq.monthly <- seq(1, 59, length.out = 59)

# Evaluate fd object at monthly resolution
yields_mat_monthly <- eval.fd(t.seq.monthly, yields_fd)

# Cross-sectional mean and SD at each month
mean_curve <- rowMeans(yields_mat_monthly)
sd_curve   <- apply(yields_mat_monthly, 1, sd)

# Correct date labels
months <- seq(as.Date("2021-01-01"), as.Date("2025-11-01"), by = "month")

# Helper: convert index to date
idx_to_date <- function(idx, months) {
  idx <- idx[idx >= 1 & idx <= length(months)]
  format(months[idx], "%Y-%m")
}


#Plot mean curve with SD
plot(months, mean_curve,
     type = "l", lwd = 2, col = "darkblue",
     xlab = "Date", ylab = "Mean yield change",
     main = "Cross-sectional Mean Yield Curve with SD Envelope")
lines(months, mean_curve + sd_curve, col = "red",  lty = 2, lwd = 1.5)
lines(months, mean_curve - sd_curve, col = "red",  lty = 2, lwd = 1.5)
abline(h = 0, col = "gray", lty = 3)
legend("topright",
       legend = c("Mean", "Mean ± SD"),
       col    = c("darkblue", "red"),
       lty    = c(1, 2), lwd = 2, bty = "n")


#changepoint package 
vec <- as.vector(mean_curve)

# --- AMOC: at most one change point ---
cpt_amoc <- cpt.mean(vec, method = "AMOC")
cat("AMOC change point index:  ", cpts(cpt_amoc), "\n")
cat("AMOC change point date:   ", idx_to_date(cpts(cpt_amoc), months), "\n\n")

# --- PELT with multiple penalties ---
# Default penalty often overfits smooth functional data — compare all options
cpt_pelt_default <- cpt.mean(vec, method = "PELT")
cpt_pelt_bic     <- cpt.mean(vec, penalty = "BIC",          method = "PELT")
cpt_pelt_mbic    <- cpt.mean(vec, penalty = "MBIC",         method = "PELT")
cpt_pelt_aic     <- cpt.mean(vec, penalty = "AIC",          method = "PELT")
cpt_pelt_hq      <- cpt.mean(vec, penalty = "Hannan-Quinn", method = "PELT")
# Manual stronger penalty: 3*log(n) — recommended for smooth series
cpt_pelt_manual  <- cpt.mean(vec, penalty  = "Manual",
                             pen.value = 3 * log(length(vec)),
                             method    = "PELT")

cat("=== PELT penalty comparison ===\n")
cat("Default:       ", ncpts(cpt_pelt_default), "breaks:",
    idx_to_date(cpts(cpt_pelt_default), months), "\n")
cat("AIC:           ", ncpts(cpt_pelt_aic),     "breaks:",
    idx_to_date(cpts(cpt_pelt_aic),     months), "\n")
cat("BIC:           ", ncpts(cpt_pelt_bic),     "breaks:",
    idx_to_date(cpts(cpt_pelt_bic),     months), "\n")
cat("MBIC:          ", ncpts(cpt_pelt_mbic),    "breaks:",
    idx_to_date(cpts(cpt_pelt_mbic),    months), "\n")
cat("Hannan-Quinn:  ", ncpts(cpt_pelt_hq),      "breaks:",
    idx_to_date(cpts(cpt_pelt_hq),      months), "\n")
cat("Manual 3log(n):", ncpts(cpt_pelt_manual),  "breaks:",
    idx_to_date(cpts(cpt_pelt_manual),  months), "\n\n")

# Choose best PELT — prefer BIC or MBIC for small n
# Use the one with smaller number of breaks
cpt_pelt_best <- cpt_pelt_mbic

# SegNeigh with BIC
cpt_segneigh <- cpt.mean(vec, penalty = "BIC", method = "SegNeigh")
cat("SegNeigh BIC breaks:", idx_to_date(cpts(cpt_segneigh), months), "\n\n")

# BinSeg
cpt_binseg <- cpt.mean(vec, method = "BinSeg")
cat("BinSeg breaks:      ", idx_to_date(cpts(cpt_binseg), months), "\n\n")


#changepoint panel plot 
par(mfrow = c(2, 2))
plot(cpt_amoc,      main = "AMOC",         xlab = "Month index", ylab = "Mean yield")
plot(cpt_pelt_best, main = "PELT (BIC)",   xlab = "Month index", ylab = "Mean yield")
plot(cpt_segneigh,  main = "SegNeigh BIC", xlab = "Month index", ylab = "Mean yield")
plot(cpt_binseg,    main = "BinSeg",       xlab = "Month index", ylab = "Mean yield")
par(mfrow = c(1, 1))


#ecp package 
# e.divisive: non-parametric, energy-based, multiple change points
output_div <- e.divisive(matrix(vec), R = 499, alpha = 2)
cat("e.divisive detected:", output_div$k.hat, "change points\n")
cat("e.divisive dates:   ",
    idx_to_date(output_div$estimates, months), "\n\n")

# e.agglo: agglomerative alternative
output_agg <- e.agglo(X = matrix(vec), alpha = 1)
cat("e.agglo dates:      ",
    idx_to_date(output_agg$estimates, months), "\n\n")

# ecp panel plot
par(mfrow = c(1, 2))

plot(months, vec, type = "l", lwd = 2, col = "darkblue",
     xlab = "Date", ylab = "Mean yield change",
     main = "e.divisive")
abline(v = months[output_div$estimates[
  output_div$estimates >= 1 & output_div$estimates <= 63]],
  col = "red", lty = 2, lwd = 2)

plot(months, vec, type = "l", lwd = 2, col = "darkblue",
     xlab = "Date", ylab = "Mean yield change",
     main = "e.agglo")
abline(v = months[output_agg$estimates[
  output_agg$estimates >= 1 & output_agg$estimates <= 63]],
  col = "orange", lty = 2, lwd = 2)

par(mfrow = c(1, 1))


#EnvCpt model selection
models_yields <- envcpt(vec,
                        models = c("mean",      "meanar1",     "meanar2",
                                   "meanar1cpt","meanar2cpt",
                                   "trend",     "trendcpt",
                                   "trendar1",  "trendar2",
                                   "trendar1cpt","trendar2cpt"))

# Diagnostic plots
plot(models_yields, type = "fit")
plot(models_yields, type = "aic")
plot(models_yields, type = "bic")

best_aic <- names(which.min(AIC(models_yields)))
best_bic <- names(which.min(BIC(models_yields)))
cat("EnvCpt best model (AIC):", best_aic, "\n")
cat("EnvCpt best model (BIC):", best_bic, "\n\n")

# Extract change points from best BIC model
# The @cpts slot contains break indices
best_model_obj <- models_yields[[best_bic]]
envcpt_cpts <- tryCatch(
  best_model_obj@cpts,
  error = function(e) {
    cat("Note: best BIC model has no change points\n")
    integer(0)
  }
)
# Remove terminal index (EnvCpt always includes n as last cpt)
envcpt_cpts <- envcpt_cpts[envcpt_cpts < length(vec)]

cat("EnvCpt break dates (", best_bic, "):",
    idx_to_date(envcpt_cpts, months), "\n\n")


#Consensus: which breaks are most credible
#A break found by 2+ methods = strong evidence 
# Collect all break indices from methods with <= 6 breaks
collect_breaks <- function(idx_vec, max_breaks = 6) {
  idx_vec <- idx_vec[idx_vec >= 1 & idx_vec <= 63]
  if (length(idx_vec) <= max_breaks) idx_vec else integer(0)
}

agg_breaks <- output_agg$estimates[
  output_agg$estimates > 1 & output_agg$estimates < length(vec)]

all_breaks <- c(
  collect_breaks(cpts(cpt_amoc)),
  collect_breaks(cpts(cpt_pelt_best)),
  collect_breaks(cpts(cpt_segneigh)),
  collect_breaks(cpts(cpt_binseg)),
  collect_breaks(output_div$estimates),
  collect_breaks(agg_breaks),         
  collect_breaks(envcpt_cpts)
)

# Cluster breaks within a 3-month window
cluster_breaks <- function(breaks, window = 3) {
  if (length(breaks) == 0) return(integer(0))
  breaks  <- sort(breaks)
  cluster <- rep(0, length(breaks))
  cid     <- 1
  cluster[1] <- cid
  for (i in 2:length(breaks)) {
    if (breaks[i] - breaks[i-1] <= window) {
      cluster[i] <- cid
    } else {
      cid <- cid + 1
      cluster[i] <- cid
    }
  }
  # Representative = median index of each cluster
  sapply(unique(cluster), function(c) round(median(breaks[cluster == c])))
}

break_freq   <- table(all_breaks)
credible_idx <- as.integer(names(break_freq[break_freq >= 2]))
credible_idx <- cluster_breaks(credible_idx, window = 3)

cat("=== CONSENSUS BREAK POINTS ===\n")
cat("(Detected by 2+ methods)\n\n")
if (length(credible_idx) == 0) {
  cat("No consensus break points found\n")
} else {
  for (i in seq_along(credible_idx)) {
    cat(sprintf("Break %d: index %d → %s\n",
                i, credible_idx[i],
                idx_to_date(credible_idx[i], months)))
  }
}


#summary plot 
method_colors <- c("red", "darkgreen", "purple", "steelblue",
                   "orange", "hotpink", "brown")
method_ltys   <- c(2, 3, 4, 5, 6, 2, 3)
method_names  <- c("AMOC", "PELT BIC", "SegNeigh", "BinSeg",
                   "e.divisive", "e.agglo", "EnvCpt BIC")

method_breaks <- list(
  collect_breaks(cpts(cpt_amoc)),
  collect_breaks(cpts(cpt_pelt_best)),
  collect_breaks(cpts(cpt_segneigh)),
  collect_breaks(cpts(cpt_binseg)),
  collect_breaks(output_div$estimates),
  collect_breaks(agg_breaks),         
  collect_breaks(envcpt_cpts)
)

plot(months, vec,
     type = "l", lwd = 2.5, col = "darkblue",
     xlab = "Date",
     ylab = "Cross-sectional mean yield change",
     main = "Change Point Detection — All Methods",
     las  = 2)
abline(h = 0, col = "gray70", lty = 3)

# Plot each method's breaks
for (m in seq_along(method_breaks)) {
  breaks_m <- method_breaks[[m]]
  if (length(breaks_m) > 0) {
    abline(v   = months[breaks_m],
           col = method_colors[m],
           lty = method_ltys[m],
           lwd = 1.8)
  }
}

# Highlight consensus breaks with shaded band
if (length(credible_idx) > 0) {
  for (ci in credible_idx) {
    rect(months[max(1, ci-1)], par("usr")[3],
         months[min(63, ci+1)], par("usr")[4],
         col = adjustcolor("gold", alpha.f = 0.25),
         border = NA)
  }
}

legend("topright",
       legend = c("Mean curve", method_names, "Consensus zone"),
       col    = c("darkblue", method_colors, "gold"),
       lty    = c(1, method_ltys, 1),
       lwd    = c(2.5, rep(1.8, 6), 8),
       bty    = "n", cex = 0.75)


# Results summary table  
cat("\n==========================================\n")
cat("   CHANGE POINT ANALYSIS — FINAL RESULTS  \n")
cat("==========================================\n\n")

cat("--- Break points by method ---\n")
cat(sprintf("%-18s %s\n", "Method", "Break dates"))
cat(strrep("-", 50), "\n")
cat(sprintf("%-18s %s\n", "AMOC",
            paste(idx_to_date(cpts(cpt_amoc), months), collapse = ", ")))
cat(sprintf("%-18s %s\n", "PELT (BIC)",
            paste(idx_to_date(cpts(cpt_pelt_best), months), collapse = ", ")))
cat(sprintf("%-18s %s\n", "SegNeigh (BIC)",
            paste(idx_to_date(cpts(cpt_segneigh), months), collapse = ", ")))
cat(sprintf("%-18s %s\n", "BinSeg",
            paste(idx_to_date(cpts(cpt_binseg), months), collapse = ", ")))
cat(sprintf("%-18s %s\n", "e.divisive",
            paste(idx_to_date(
              output_div$estimates[output_div$estimates > 1 &
                                     output_div$estimates < 63],
              months), collapse = ", ")))

cat(sprintf("%-18s %s\n", "e.agglo",           # NEW LINE
            paste(idx_to_date(agg_breaks, months), collapse = ", ")))

cat(sprintf("%-18s %s\n", paste0("EnvCpt (", best_bic, ")"),
            paste(idx_to_date(envcpt_cpts, months), collapse = ", ")))
cat(strrep("-", 50), "\n")
cat(sprintf("%-18s %s\n", "CONSENSUS",
            paste(idx_to_date(credible_idx, months), collapse = ", ")))
cat(strrep("=", 50), "\n\n")

cat("Conclusion: ")
if (length(credible_idx) == 0) {
  cat("No robust evidence against H0.\n")
} else {
  cat("H0 rejected. Structural breaks detected at:\n")
  for (i in seq_along(credible_idx)) {
    cat(sprintf("  Break %d: %s\n", i,
                idx_to_date(credible_idx[i], months)))
  }
}


#final nice plot

plot(months, vec,
     type = "l", lwd = 2.5, col = "darkblue",
     xlab = "Date",
     ylab = "Cross-sectional mean yield change (%)",
     main = "Structural Breaks in Eurozone Mean Yield Curve",
     las  = 2, cex.axis = 0.8)
abline(h = 0, col = "gray70", lty = 3)

# Shade consensus break regions
for (ci in credible_idx) {
  rect(months[max(1, ci-1)], par("usr")[3],
       months[min(63, ci+1)], par("usr")[4],
       col    = adjustcolor("gold", alpha.f = 0.3),
       border = NA)
}

# Add vertical lines at consensus breaks
abline(v   = months[credible_idx],
       col = "darkred", lty = 2, lwd = 2)


legend("topright",
       legend = c("Cross-sectional mean", "Consensus break", "Break region"),
       col    = c("darkblue", "darkred", "gold"),
       lty    = c(1, 2, 1),
       lwd    = c(2.5, 2, 8),
       bty    = "n", cex = 0.85)






#final concensus table
cat("==============================================\n")
cat("  CHANGE POINT ANALYSIS — METHOD COMPARISON  \n")
cat("==============================================\n\n")
cat(sprintf("%-20s %-30s %s\n", "Method", "Break dates", "N breaks"))
cat(strrep("-", 60), "\n")
cat(sprintf("%-20s %-30s %s\n", "AMOC",
            paste(idx_to_date(cpts(cpt_amoc), months), collapse=", "),
            ncpts(cpt_amoc)))
cat(sprintf("%-20s %-30s %s\n", "PELT (BIC)",
            paste(idx_to_date(cpts(cpt_pelt_best), months), collapse=", "),
            ncpts(cpt_pelt_best)))
cat(sprintf("%-20s %-30s %s\n", "SegNeigh (BIC)",
            paste(idx_to_date(cpts(cpt_segneigh), months), collapse=", "),
            ncpts(cpt_segneigh)))
cat(sprintf("%-20s %-30s %s\n", "BinSeg",
            paste(idx_to_date(cpts(cpt_binseg), months), collapse=", "),
            ncpts(cpt_binseg)))
cat(sprintf("%-20s %-30s %s\n", "e.divisive",
            paste(idx_to_date(
              output_div$estimates[output_div$estimates > 1 &
                                     output_div$estimates < 63],
              months), collapse=", "),
            output_div$k.hat))
cat(sprintf("%-20s %-30s %s\n", "e.agglo",
            paste(idx_to_date(agg_breaks, months), collapse=", "),
            length(agg_breaks)))
cat(sprintf("%-20s %-30s %s\n", paste0("EnvCpt(", best_bic,")"),
            paste(idx_to_date(envcpt_cpts, months), collapse=", "),
            length(envcpt_cpts)))
cat(strrep("-", 60), "\n")

#count agreements per consensus break
# Rebuild all per-method break lists with date strings for matching
method_break_dates <- list(
  idx_to_date(collect_breaks(cpts(cpt_amoc)),            months),
  idx_to_date(collect_breaks(cpts(cpt_pelt_best)),       months),
  idx_to_date(collect_breaks(cpts(cpt_segneigh)),        months),
  idx_to_date(collect_breaks(cpts(cpt_binseg)),          months),
  idx_to_date(collect_breaks(output_div$estimates[
    output_div$estimates > 1 &
      output_div$estimates < 63]),                         months),
  idx_to_date(collect_breaks(agg_breaks),                months),
  idx_to_date(collect_breaks(envcpt_cpts),               months)
)
method_labels <- c("AMOC", "PELT", "SegNeigh", "BinSeg",
                   "e.divisive", "e.agglo", "EnvCpt")

# For each consensus break, find which methods detected it
# A method "agrees" if it has a break within 3 months of the consensus index
cat(sprintf("\n%-12s %-14s %-6s %s\n",
            "Break date", "Index", "Agree", "Methods"))
cat(strrep("-", 60), "\n")

for (ci in credible_idx) {
  
  # Which methods have a break within window of ci
  agreeing <- c()
  for (m in seq_along(method_break_dates)) {
    # Convert method date strings back to indices for window comparison
    m_dates <- method_break_dates[[m]]
    if (length(m_dates) == 0) next
    # Find month indices for this method's breaks
    m_idx <- which(format(months, "%Y-%m") %in% m_dates)
    # Check if any are within 3 months of ci
    if (any(abs(m_idx - ci) <= 3)) {
      agreeing <- c(agreeing, method_labels[m])
    }
  }
  
  cat(sprintf("%-12s %-14s %-6s %s\n",
              idx_to_date(ci, months),
              paste0("(t=", ci, ")"),
              paste0(length(agreeing), "/7"),
              paste(agreeing, collapse=", ")))
}

cat(strrep("=", 60), "\n")
cat(sprintf("Threshold: breaks detected by >= 2 methods (window = ±3 months)\n"))
cat(sprintf("Total methods used: 7\n"))


























#convert files to python readable form

library(fda)
orig_df <- readRDS("Eurozone_10Y_Yields_Monthly.rds")
write.csv(orig_df, "Eurozone_10Y_Yields_Monthly.csv", row.names = FALSE)

yields_fd <- readRDS("yield_changes_fd.rds")

# Evaluate on the full monthly grid
tt <- 1:59
yields_matrix <- eval.fd(tt, yields_fd)   # (59 × n_curves) matrix

# Save the evaluated matrix with country names as column headers
write.csv(
  as.data.frame(yields_matrix),
  "yields_matrix.csv",
  row.names = TRUE   # row names = time index 1..59
)

# Save basis metadata separately
basis_info <- data.frame(
  t_min    = yields_fd$basis$rangeval[1],
  t_max    = yields_fd$basis$rangeval[2],
  n_basis  = yields_fd$basis$nbasis,
  basis_type = yields_fd$basis$type
)
write.csv(basis_info, "basis_info.csv", row.names = FALSE)

# Save the raw coefficient matrix (n_basis × n_curves)
write.csv(
  as.data.frame(yields_fd$coefs),
  "yields_coefs.csv",
  row.names = FALSE
)

cat("Done. Files written:\n")
cat("  yields_matrix.csv\n")
cat("  yields_coefs.csv\n")
cat("  basis_info.csv\n")
