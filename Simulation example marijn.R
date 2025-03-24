##################################################
# 1. Setup
##################################################
library(beat)   # For balanced_causal_forest(), possibly balanced_regression_forest()
library(grf)    # For causal_forest(), regression_forest(), get_forest_weights()
library(data.table)
library(ggpubr)
library(gridExtra)
set.seed(42)

##################################################
# 2. Simulate Data with ~50/50 Y
##################################################

n1 <- 1000  # training
n2 <- 1000  # testing
n  <- n1 + n2

# Feature dimensions
p_continuous <- 4
p_discrete   <- 3

# Continuous features ~ N(0,1)
X_cont <- matrix(rnorm(n * p_continuous), n, p_continuous)
# Discrete features ~ Bernoulli(0.3)
X_disc <- matrix(rbinom(n * p_discrete, 1, 0.3), n, p_discrete)
# Combine
X <- cbind(X_cont, X_disc)

# Protected attribute Z
Z <- rbinom(n, 1, 1 / (1 + exp(-X_cont[, 2])))

# Treatment W (binary)
W <- rbinom(n, 1, 0.5)

# "tau" effect
tau <- (-1 + pmax(X[,1], 0) + X[,2] + abs(X[,3]) + X[,5])

# Create latent base
latent_base <- (X[,1] - 2*X[,2] + X[,4] + 3*Z + tau*W)

# Solve for intercept 'a' so that mean of logistic(...) = 0.5
f_obj <- function(a) {
  mean(1 / (1 + exp(-(latent_base + a)))) - 0.5
}
a_star <- uniroot(f_obj, interval = c(-100, 100))$root

# Probability p
p <- 1 / (1 + exp(-(latent_base + a_star)))
# Final Y ~ Bernoulli(p)
Y <- rbinom(n, size = 1, prob = p)

cat("Empirical recidivism rate:", mean(Y), "\n")  # ~0.50

##################################################
# 3. Split Train / Test
##################################################
train_idx <- 1:n1
test_idx  <- (n1 + 1):(n1 + n2)

X_train <- X[train_idx, ]
X_test  <- X[test_idx, ]
Z_train <- Z[train_idx]
Z_test  <- Z[test_idx]
W_train <- W[train_idx]
W_test  <- W[test_idx]
Y_train <- Y[train_idx]
Y_test  <- Y[test_idx]

##################################################
# 4. Prepare Data for "No Z" vs. "All Data"
##################################################
# "No Z": we simply ignore Z as a feature.
X_train_noZ <- X_train
X_test_noZ  <- X_test

# "All Data": we append Z as an extra feature column
X_train_all <- cbind(X_train, Z = Z_train)
X_test_all  <- cbind(X_test,  Z = Z_test)

##################################################
# 5. Fit Six Different Forests
##################################################

num_trees  <- 500  # smaller for example speed; can increase
my_penalty <- 10

# (1) Causal_beat
fit_causal_beat <- balanced_causal_forest(
  X_train_noZ, Y_train, W_train,
  target.weights = as.matrix(Z_train),
  target.weight.penalty = my_penalty,
  num.trees = num_trees,
  target.weight.penalty.metric = "custom.metric"
)

# (2) Regression_beat
fit_regression_beat <- balanced_regression_forest(
  X_train_noZ, Y_train,
  target.weights = as.matrix(Z_train),
  target.weight.penalty = my_penalty,
  num.trees = num_trees
)

# (3) Causal_grf_noZ (standard GRF causal forest, ignoring Z)
fit_causal_grf_noZ <- causal_forest(X_train_noZ, Y_train, W_train, num.trees = num_trees)

# (4) Regression_grf_noZ (standard GRF regression forest, ignoring Z)
fit_regression_grf_noZ <- regression_forest(X_train_noZ, Y_train, num.trees = num_trees)

# (5) Causal_grf_all (standard GRF causal forest, X includes Z)
fit_causal_grf_all <- causal_forest(X_train_all, Y_train, W_train, num.trees = num_trees)

# (6) Regression_grf_all (standard GRF regression forest, X includes Z)
fit_regression_grf_all <- regression_forest(X_train_all, Y_train, num.trees = num_trees)

##################################################
# 6. Get In-Sample Weights & Compute RSS
##################################################
compute_in_sample_rss <- function(fit, Y_train) {
  # Attempt to extract in-sample weights
  Wmat <- tryCatch(
    get_forest_weights(fit),
    error = function(e) { return(NULL) }
  )
  if (is.null(Wmat)) {
    # Not supported for this forest type
    return(rep(NA, length(Y_train)))
  }
  # Multiply by Y=1
  as.numeric(Wmat %*% Y_train)
}

rss_causal_beat        <- compute_in_sample_rss(fit_causal_beat,      Y_train)
rss_regression_beat    <- compute_in_sample_rss(fit_regression_beat,  Y_train)
rss_causal_grf_noZ     <- compute_in_sample_rss(fit_causal_grf_noZ,   Y_train)
rss_regression_grf_noZ <- compute_in_sample_rss(fit_regression_grf_noZ, Y_train)
rss_causal_grf_all     <- compute_in_sample_rss(fit_causal_grf_all,   Y_train)
rss_regression_grf_all <- compute_in_sample_rss(fit_regression_grf_all, Y_train)

##################################################
# 7. Classification Metrics Over Thresholds
##################################################
library(dplyr)

evaluate_rss <- function(rss_vec, Y_true, model_label) {
  thresholds <- seq(0, 1, by = 0.1)
  n <- length(Y_true)

  res_list <- lapply(thresholds, function(th) {
    pred <- ifelse(rss_vec > th, 1, 0)
    TP <- sum(pred == 1 & Y_true == 1)
    FP <- sum(pred == 1 & Y_true == 0)
    TN <- sum(pred == 0 & Y_true == 0)
    FN <- sum(pred == 0 & Y_true == 1)

    precision <- if ((TP + FP) == 0) NA else TP / (TP + FP)
    accuracy  <- (TP + TN) / (TP + FP + TN + FN)
    fp_pct    <- 100 * FP / n
    fn_pct    <- 100 * FN / n

    data.frame(
      model    = model_label,
      threshold= th,
      precision= precision,
      accuracy = accuracy,
      false_positive_pct = fp_pct,
      false_negative_pct = fn_pct
    )
  })
  do.call(rbind, res_list)
}

results_df <- bind_rows(
  evaluate_rss(rss_causal_beat,        Y_train, "Causal_BEAT"),
  evaluate_rss(rss_regression_beat,    Y_train, "Regression_BEAT"),
  evaluate_rss(rss_causal_grf_noZ,     Y_train, "Causal_GRF_noZ"),
  evaluate_rss(rss_regression_grf_noZ, Y_train, "Regression_GRF_noZ"),
  evaluate_rss(rss_causal_grf_all,     Y_train, "Causal_GRF_all"),
  evaluate_rss(rss_regression_grf_all, Y_train, "Regression_GRF_all")
)

print(head(results_df, 20))

##################################################
# 8. Out-of-Sample Predictions (Fix for "All" Models)
##################################################
# For the "all" models that include Z in X, we must use X_test_all
# to match the training dimension.

# Causal models
beat_causal_test       <- predict(fit_causal_beat,       X_test)$predictions
grf_causal_noZ_test    <- predict(fit_causal_grf_noZ,    X_test)$predictions
grf_causal_all_test    <- predict(fit_causal_grf_all,    X_test_all)$predictions  # <-- FIXED

# Regression models
beat_regression_test    <- predict(fit_regression_beat,    X_test)$predictions
grf_regression_noZ_test <- predict(fit_regression_grf_noZ, X_test)$predictions
grf_regression_all_test <- predict(fit_regression_grf_all, X_test_all)$predictions # <-- FIXED

##################################################
# 9. Create a Data Table for Plotting
##################################################
# If your original code still has 'tau' and 'Y_r' in test_data, we can plot them:
test_data <- data.frame(
  Y = Y_test,
  tau = tau[test_idx],
  Y_r = latent_base[test_idx]  # or your original "Y_r" if stored
)

dat.plot <- data.table(
  true_causal        = test_data$tau,    # or NA if not tracked
  beat_causal        = beat_causal_test,
  grf_causal_noZ     = grf_causal_noZ_test,
  grf_causal_all     = grf_causal_all_test,

  true_reg           = test_data$Y_r,    # or test_data$Y if you want predicted vs. actual
  beat_regression    = beat_regression_test,
  grf_regression_noZ = grf_regression_noZ_test,
  grf_regression_all = grf_regression_all_test,

  Z = as.factor(Z_test)
)

##################################################
# 10. Make Density Plots for Causal
##################################################
p1_causal <- ggdensity(
  data    = dat.plot,
  x       = "true_causal",
  color   = "Z",
  fill    = "Z",
  alpha   = 0.2,
  add     = "mean",
  title   = "True Causal Effect"
)

p2_causal <- ggdensity(
  data    = dat.plot,
  x       = "beat_causal",
  color   = "Z",
  fill    = "Z",
  alpha   = 0.2,
  add     = "mean",
  title   = "BEAT Causal Predictions"
)

p3_causal <- ggdensity(
  data    = dat.plot,
  x       = "grf_causal_noZ",
  color   = "Z",
  fill    = "Z",
  alpha   = 0.2,
  add     = "mean",
  title   = "GRF Causal (No Z)"
)

p4_causal <- ggdensity(
  data    = dat.plot,
  x       = "grf_causal_all",
  color   = "Z",
  fill    = "Z",
  alpha   = 0.2,
  add     = "mean",
  title   = "GRF Causal (All)"
)

p_causal <- ggarrange(p1_causal, p2_causal, p3_causal, p4_causal, ncol = 2, nrow = 2)

##################################################
# 11. Make Density Plots for Regression
##################################################
p1_reg <- ggdensity(
  data    = dat.plot,
  x       = "true_reg",
  color   = "Z",
  fill    = "Z",
  alpha   = 0.2,
  add     = "mean",
  title   = "True Regression (Y_r)"
)

p2_reg <- ggdensity(
  data    = dat.plot,
  x       = "beat_regression",
  color   = "Z",
  fill    = "Z",
  alpha   = 0.2,
  add     = "mean",
  title   = "BEAT Regression Predictions"
)

p3_reg <- ggdensity(
  data    = dat.plot,
  x       = "grf_regression_noZ",
  color   = "Z",
  fill    = "Z",
  alpha   = 0.2,
  add     = "mean",
  title   = "GRF Regression (No Z)"
)

p4_reg <- ggdensity(
  data    = dat.plot,
  x       = "grf_regression_all",
  color   = "Z",
  fill    = "Z",
  alpha   = 0.2,
  add     = "mean",
  title   = "GRF Regression (All)"
)

p_reg <- ggarrange(p1_reg, p2_reg, p3_reg, p4_reg, ncol = 2, nrow = 2)

##################################################
# 12. Display or Save Plots
##################################################
print(p_causal)
print(p_reg)
