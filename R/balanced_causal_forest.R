#' Balanced Causal forest
#'
#' Trains a causal forest that can be used to estimate
#' conditional average treatment effects tau(X). When
#' the treatment assignment W is binary and unconfounded,
#' we have tau(X) = E[Y(1) - Y(0) | X = x], where Y(0) and
#' Y(1) are potential outcomes corresponding to the two possible
#' treatment states. When W is continuous, we effectively estimate
#' an average partial effect Cov[Y, W | X = x] / Var[W | X = x],
#' and interpret it as a treatment effect given unconfoundedness.
#'
#' @param X The covariates used in the causal regression.
#' @param Y The outcome (must be a numeric vector with no NAs).
#' @param W The treatment assignment (must be a binary or real numeric vector with no NAs).
#' @param Y.hat Estimates of the expected responses E[Y | Xi], marginalizing
#'              over treatment. If Y.hat = NULL, these are estimated using
#'              a separate regression forest. See section 6.1.1 of the GRF paper for
#'              further discussion of this quantity. Default is NULL.
#' @param W.hat Estimates of the treatment propensities E[W | Xi]. If W.hat = NULL,
#'              these are estimated using a separate regression forest. Default is NULL.
#' @param num.trees Number of trees grown in the forest. Note: Getting accurate
#'                  confidence intervals generally requires more trees than
#'                  getting accurate predictions. Default is 2000.
#' @param sample.weights Weights given to each sample in estimation.
#'                       If NULL, each observation receives the same weight.
#'                       Note: To avoid introducing confounding, weights should be
#'                       independent of the potential outcomes given X. Default is NULL.
#' @param clusters Vector of integers or factors specifying which cluster each observation corresponds to.
#'  Default is NULL (ignored).
#' @param equalize.cluster.weights If FALSE, each unit is given the same weight (so that bigger
#'  clusters get more weight). If TRUE, each cluster is given equal weight in the forest. In this case,
#'  during training, each tree uses the same number of observations from each drawn cluster: If the
#'  smallest cluster has K units, then when we sample a cluster during training, we only give a random
#'  K elements of the cluster to the tree-growing procedure. When estimating average treatment effects,
#'  each observation is given weight 1/cluster size, so that the total weight of each cluster is the
#'  same. Note that, if this argument is FALSE, sample weights may also be directly adjusted via the
#'  sample.weights argument. If this argument is TRUE, sample.weights must be set to NULL. Default is
#'  FALSE.
#' @param sample.fraction Fraction of the data used to build each tree.
#'                        Note: If honesty = TRUE, these subsamples will
#'                        further be cut by a factor of honesty.fraction. Default is 0.5.
#' @param mtry Number of variables tried for each split. Default is
#'             \eqn{\sqrt p + 20} where p is the number of variables.
#' @param min.node.size A target for the minimum number of observations in each tree leaf. Note that nodes
#'                      with size smaller than min.node.size can occur, as in the original randomForest package.
#'                      Default is 5.
#' @param honesty Whether to use honest splitting (i.e., sub-sample splitting). Default is TRUE.
#'  For a detailed description of honesty, honesty.fraction, honesty.prune.leaves, and recommendations for
#'  parameter tuning, see the grf
#'  \href{https://grf-labs.github.io/grf/REFERENCE.html#honesty-honesty-fraction-honesty-prune-leaves}{algorithm reference}.
#' @param honesty.fraction The fraction of data that will be used for determining splits if honesty = TRUE. Corresponds
#'                         to set J1 in the notation of the paper. Default is 0.5 (i.e. half of the data is used for
#'                         determining splits).
#' @param honesty.prune.leaves If TRUE, prunes the estimation sample tree such that no leaves
#'  are empty. If FALSE, keep the same tree as determined in the splits sample (if an empty leave is encountered, that
#'  tree is skipped and does not contribute to the estimate). Setting this to FALSE may improve performance on
#'  small/marginally powered data, but requires more trees (note: tuning does not adjust the number of trees).
#'  Only applies if honesty is enabled. Default is TRUE.
#' @param alpha A tuning parameter that controls the maximum imbalance of a split. Default is 0.05.
#' @param imbalance.penalty A tuning parameter that controls how harshly imbalanced splits are penalized. Default is 0.
#' @param stabilize.splits Whether or not the treatment should be taken into account when
#'                         determining the imbalance of a split. Default is TRUE.
#' @param ci.group.size The forest will grow ci.group.size trees on each subsample.
#'                      In order to provide confidence intervals, ci.group.size must
#'                      be at least 2. Default is 2.
#' @param tune.parameters A vector of parameter names to tune.
#'  If "all": all tunable parameters are tuned by cross-validation. The following parameters are
#'  tunable: ("sample.fraction", "mtry", "min.node.size", "honesty.fraction",
#'   "honesty.prune.leaves", "alpha", "imbalance.penalty"). If honesty is FALSE the honesty.* parameters are not tuned.
#'  Default is "none" (no parameters are tuned).
#' @param tune.num.trees The number of trees in each 'mini forest' used to fit the tuning model. Default is 200.
#' @param tune.num.reps The number of forests used to fit the tuning model. Default is 50.
#' @param tune.num.draws The number of random parameter values considered when using the model
#'                          to select the optimal parameters. Default is 1000.
#' @param compute.oob.predictions Whether OOB predictions on training set should be precomputed. Default is TRUE.
#' @param orthog.boosting (experimental) If TRUE, then when Y.hat = NULL or W.hat is NULL,
#'                 the missing quantities are estimated using boosted regression forests.
#'                 The number of boosting steps is selected automatically. Default is FALSE.
#' @param num.threads Number of threads used in training. By default, the number of threads is set
#'                    to the maximum hardware concurrency.
#' @param seed The seed of the C++ random number generator.
#'
#' @return A trained causal forest object. If tune.parameters is enabled,
#'  then tuning information will be included through the `tuning.output` attribute.
#'
#' @examples
#' \donttest{
#' # Train a causal forest.
#' n <- 500
#' p <- 10
#' X <- matrix(rnorm(n * p), n, p)
#' W <- rbinom(n, 1, 0.5)
#' Y <- pmax(X[, 1], 0) * W + X[, 2] + pmin(X[, 3], 0) + rnorm(n)
#' c.forest <- causal_forest(X, Y, W)
#'
#' # Predict using the forest.
#' X.test <- matrix(0, 101, p)
#' X.test[, 1] <- seq(-2, 2, length.out = 101)
#' c.pred <- predict(c.forest, X.test)
#'
#' # Predict on out-of-bag training samples.
#' c.pred <- predict(c.forest)
#'
#' # Predict with confidence intervals; growing more trees is now recommended.
#' c.forest <- causal_forest(X, Y, W, num.trees = 4000)
#' c.pred <- predict(c.forest, X.test, estimate.variance = TRUE)
#'
#' # In some examples, pre-fitting models for Y and W separately may
#' # be helpful (e.g., if different models use different covariates).
#' # In some applications, one may even want to get Y.hat and W.hat
#' # using a completely different method (e.g., boosting).
#' n <- 2000
#' p <- 20
#' X <- matrix(rnorm(n * p), n, p)
#' TAU <- 1 / (1 + exp(-X[, 3]))
#' W <- rbinom(n, 1, 1 / (1 + exp(-X[, 1] - X[, 2])))
#' Y <- pmax(X[, 2] + X[, 3], 0) + rowMeans(X[, 4:6]) / 2 + W * TAU + rnorm(n)
#'
#' forest.W <- regression_forest(X, W, tune.parameters = "all")
#' W.hat <- predict(forest.W)$predictions
#'
#' forest.Y <- regression_forest(X, Y, tune.parameters = "all")
#' Y.hat <- predict(forest.Y)$predictions
#'
#' forest.Y.varimp <- variable_importance(forest.Y)
#'
#' # Note: Forests may have a hard time when trained on very few variables
#' # (e.g., ncol(X) = 1, 2, or 3). We recommend not being too aggressive
#' # in selection.
#' selected.vars <- which(forest.Y.varimp / mean(forest.Y.varimp) > 0.2)
#'
#' tau.forest <- causal_forest(X[, selected.vars], Y, W,
#'   W.hat = W.hat, Y.hat = Y.hat,
#'   tune.parameters = "all"
#' )
#' tau.hat <- predict(tau.forest)$predictions
#' }
#'

#' @import data.table
#' @import Rcpp
#'
#' @export
balanced_causal_forest <- function(X,
                                   Y,
                                   W,
                                   Y.hat = NULL,
                                   W.hat = NULL,
                                   num.trees = 2000,
                                   sample.weights = NULL,
                                   target.weights = NULL,
                                   target.weight.penalty = 0,
                                   target.weight.bins.breaks = 256,
                                   target.weight.standardize = TRUE,
                                   target.avg.weights = NULL,
                                   target.weight.penalty.metric = "split_l2_norm_rate",  # <-- Changed : Added this
                                   clusters = NULL,
                                   equalize.cluster.weights = FALSE,
                                   sample.fraction = 0.5,
                                   mtry = min(ceiling(sqrt(ncol(X)) + 20), ncol(X)),
                                   min.node.size = 5,
                                   honesty = TRUE,
                                   honesty.fraction = 0.5,
                                   honesty.prune.leaves = TRUE,
                                   alpha = 0.05,
                                   imbalance.penalty = 0,
                                   stabilize.splits = TRUE,
                                   ci.group.size = 2,
                                   tune.parameters = "none",
                                   tune.num.trees = 200,
                                   tune.num.reps = 50,
                                   tune.num.draws = 1000,
                                   compute.oob.predictions = TRUE,
                                   orthog.boosting = FALSE,
                                   num.threads = NULL,
                                   seed = runif(1, 0, .Machine$integer.max)) {

  # :----  procedure ---:
  # 1. check input data
  # 2. get list of target weights: per column of X, bin X and get column-wise average of target weight
  # 3. (default): small forest to get Y_hat and W_hat, then Y-Y_hat, W - W_hat
  # 4. tune parameters via the same balanced_calsal_forest function
  #    a.tunnalbe parameters are in tune_balanced_causal_forest,
  #    b. prepare data and pass into tune_balanced_forest
  #       * random sample parameters from some distributions
  #       * fit small forest and get debiased.error
  #       * fit dice kriging: debiased.error ~ parameter values and find the smallest combinations (not necessary the tried version)
  #       * re-fit balanced_causal_forest by the best set and larger tress
  #       * return the best or default parameter set
  # 5. fit actual balanced_calsal_forest
  # +++++++++++++++++++++
  # to make penalty tunable, change tune_balance_causal_forest.R and tunning_balanced.R

  # target.avg.weights: array, [num target weight, observation, num x]
  # if null, create it internally


  # check input data
  has.missing.values <- validate_X(X, allow.na = TRUE)
  validate_sample_weights(sample.weights, X)
  Y <- validate_observations(Y, X)
  W <- validate_observations(W, X)
  stopifnot(target.weight.penalty >= 0)
  stopifnot(nrow(target.weights) == nrow(X))

  # standardize target weight
  if (!is.null(target.weights)) {
    stopifnot(dim(target.weights)[1] == dim(X)[1])

    if (isTRUE(target.weight.standardize)) {
      # mean 0 and sd 1 per column, defined in input_utilities.R
      # if de-mean only, penalty goes large when dim increaes
      target.weights = apply(target.weights, 2, standardize)
    }

  } else {
    # target.weights = as.matrix(replicate(NROW(X), 0))
    stop("If target.weight is missing, use casual_forest")
  }

  # verify penalty metric
  available_metrics = c("split_l2_norm_rate", # left, right: l2 norm(colmean target weight)* penalty rate * node decrease
                        "euclidean_distance_rate", # (left+right decrease) *  Euclidean distance (column mean target weight left, right ) * penalty rate
                        "cosine_similarity_rate", # (left+right decrease) *  (1-cos_sim(column mean target weight left, right )) * penalty rate

                        "split_l2_norm", #  sum(left,right l2 norm(colmean target weight))* penalty rate
                        "euclidean_distance", #  Euclidean distance (column mean target weight left, right ) * penalty rate
                        "cosine_similarity", #  (1-cos_sim(column mean target weight left, right )) * penalty rate
                        "custom.metric" # Changed for custom metric
  )

  # Changed Added following:

  if (target.weight.penalty.metric == "custom.metric") {
    message("⚠️ Using 'custom.metric' for penalty calculations.")
  }
  # End of addition


  #output list : [dim(X)[2]] [[num target weight feature, num rows obs]]
  # then  convert to 3d array: [dim(target weight), length(list)]
  if(is.null(target.avg.weights)){
    target.avg.weights = construct_target_weight_mean(x = X,
                                                      z = target.weights,
                                                      num_breaks = target.weight.bins.breaks)
  }
  stopifnot(is.array(target.avg.weights))
  stopifnot(dim(target.avg.weights) == c(dim(target.weights)[2], dim(X)[1], dim(X)[2]))


  # check args
  clusters <- validate_clusters(clusters, X)
  samples.per.cluster <-
    validate_equalize_cluster_weights(equalize.cluster.weights, clusters, sample.weights)
  num.threads <- validate_num_threads(num.threads)

  all.tunable.params <- c(
    "sample.fraction",
    "mtry",
    "min.node.size",
    "honesty.fraction",
    "honesty.prune.leaves",
    "alpha",
    "imbalance.penalty",
    "target.weight.penalty" # shall we include it? --> also change tune_balanced_causal_forest.R
  )

  # if(max(abs(target.weights)) == 0){
  #   all.tunable.params = all.tunable.params[all.tunable.params != 'target.weight.penalty']
  # }

  # used to pre-process Y and W, so we don't use target weights
  args.orthog <- list(
    X = X,
    num.trees = max(50, num.trees / 4),
    sample.weights = sample.weights,
    #target.weights =  target.weights,
    #target.weight.penalty=target.weight.penalty, # no target weight in center data
    clusters = clusters,
    equalize.cluster.weights = equalize.cluster.weights,
    sample.fraction = sample.fraction,
    mtry = mtry,
    min.node.size = 5,
    honesty = TRUE,
    honesty.fraction = 0.5,
    honesty.prune.leaves = honesty.prune.leaves,
    alpha = alpha,
    imbalance.penalty = imbalance.penalty,
    ci.group.size = 1,
    tune.parameters = tune.parameters,
    #num.threads = num.threads,
    seed = seed # ,
  )
  # use standard forest
  if (is.null(Y.hat) && !orthog.boosting) {
    # default setting goes here
    forest.Y <- do.call(regression_forest, c(Y = list(Y), args.orthog))
    Y.hat <- predict(forest.Y)$predictions
  } else if (is.null(Y.hat) && orthog.boosting) {
    forest.Y <-
      do.call(boosted_regression_forest, c(Y = list(Y), args.orthog))
    Y.hat <- predict(forest.Y)$predictions
  } else if (length(Y.hat) == 1) {
    Y.hat <- rep(Y.hat, nrow(X))
  } else if (length(Y.hat) != nrow(X)) {
    stop("Y.hat has incorrect length.")
  }

  if (is.null(W.hat) && !orthog.boosting) {
    forest.W <- do.call(regression_forest, c(Y = list(W), args.orthog))
    W.hat <- predict(forest.W)$predictions
  } else if (is.null(W.hat) && orthog.boosting) {
    forest.W <-
      do.call(boosted_regression_forest, c(Y = list(W), args.orthog))
    W.hat <- predict(forest.W)$predictions
  } else if (length(W.hat) == 1) {
    W.hat <- rep(W.hat, nrow(X))
  } else if (length(W.hat) != nrow(X)) {
    stop("W.hat has incorrect length.")
  }

  # orthog target weight matrix for multi reg
  args.target = copy(args.orthog)
  remove_args = c("ci.group.size", 'tune.parameters')
  for (i in remove_args) {
    args.target[i] = NULL
  }

  Y.centered <- Y - Y.hat
  W.centered <- W - W.hat

  # combine them into one matrix and track column index for splitting
  data <- create_train_matrices(X,outcome = Y.centered,treatment = W.centered, sample.weights = sample.weights)

  # for tunning and building forest
  args <- list(
    num.trees = num.trees,
    # nested list of matrix, one matrix per column of X
    target.avg.weights = target.avg.weights,
    target.weight.penalty = target.weight.penalty,
    target.weight.penalty.metric = target.weight.penalty.metric, # Changed from   target.weight.penalty.metric = "split_l2_norm_rate",
    clusters = clusters,
    samples.per.cluster = samples.per.cluster,
    sample.fraction = sample.fraction,
    mtry = mtry,
    min.node.size = min.node.size,
    honesty = honesty,
    honesty.fraction = honesty.fraction,
    honesty.prune.leaves = honesty.prune.leaves,
    alpha = alpha,
    imbalance.penalty = imbalance.penalty,
    stabilize.splits = stabilize.splits,
    ci.group.size = ci.group.size,
    compute.oob.predictions = compute.oob.predictions,
    num.threads = num.threads,
    seed = seed,
    reduced.form.weight = 0
  )

  # for tunning
  tuning.output <- NULL
  if (!identical(tune.parameters, "none")) {
    tuning.output <- tune_balanced_causal_forest(
      X,
      Y,
      W,
      Y.hat,
      W.hat,
      sample.weights = sample.weights,
      target.avg.weights = target.avg.weights,
      target.weight.penalty = target.weight.penalty,
      clusters = clusters,
      equalize.cluster.weights = equalize.cluster.weights,
      sample.fraction = sample.fraction,
      mtry = mtry,
      min.node.size = min.node.size,
      honesty = honesty,
      honesty.fraction = honesty.fraction,
      honesty.prune.leaves = honesty.prune.leaves,
      alpha = alpha,
      imbalance.penalty = imbalance.penalty,
      stabilize.splits = stabilize.splits,
      ci.group.size = ci.group.size,
      tune.parameters = tune.parameters,
      tune.num.trees = tune.num.trees,
      tune.num.reps = tune.num.reps,
      tune.num.draws = tune.num.draws,
      num.threads = num.threads,
      seed = seed
    )
    args <- modifyList(args, as.list(tuning.output[["params"]]))
  }

  # running actual forest
  forest <- do.call.rcpp(balanced_causal_train, c(data, args))

  # prepare output
  class(forest) <- c("balanced_causal_forest", "causal_forest", "grf")
  forest[["ci.group.size"]] <- ci.group.size
  forest[["X.orig"]] <- X
  forest[["Y.orig"]] <- Y
  forest[["W.orig"]] <- W
  forest[["Y.hat"]] <- Y.hat
  forest[["W.hat"]] <- W.hat
  forest[["clusters"]] <- clusters
  forest[["equalize.cluster.weights"]] <- equalize.cluster.weights
  forest[["sample.weights"]] <- sample.weights
  forest[["tunable.params"]] <- args[all.tunable.params]
  forest[["tuning.output"]] <- tuning.output
  forest[["has.missing.values"]] <- has.missing.values
  forest[["target.weight.penalty.metric"]] <- target.weight.penalty.metric # Changed this from forest[["target.weight.penalty.metric"]] <- "split_l2_norm_rate"

  forest
}

#' Predict with a balanced causal forest
#'
#' Gets estimates of tau(x) using a trained causal forest.
#'
#' @param object The trained forest.
#' @param newdata Points at which predictions should be made. If NULL, makes out-of-bag
#'                predictions on the training set instead (i.e., provides predictions at
#'                Xi using only trees that did not use the i-th training example). Note
#'                that this matrix should have the number of columns as the training
#'                matrix, and that the columns must appear in the same order.
#' @param linear.correction.variables Optional subset of indexes for variables to be used in local
#'                   linear prediction. If NULL, standard GRF prediction is used. Otherwise,
#'                   we run a locally weighted linear regression on the included variables.
#'                   Please note that this is a beta feature still in development, and may slow down
#'                   prediction considerably. Defaults to NULL.
#' @param ll.lambda Ridge penalty for local linear predictions. Defaults to NULL and will be cross-validated.
#' @param ll.weight.penalty Option to standardize ridge penalty by covariance (TRUE),
#'                  or penalize all covariates equally (FALSE). Penalizes equally by default.
#' @param num.threads Number of threads used in training. If set to NULL, the software
#'                    automatically selects an appropriate amount.
#' @param estimate.variance Whether variance estimates for hat{tau}(x) are desired
#'                          (for confidence intervals).
#' @param ... Additional arguments (currently ignored).
#'
#' @return Vector of predictions, along with estimates of the error and
#'         (optionally) its variance estimates. Column 'predictions' contains estimates
#'         of the conditional average treatent effect (CATE). The square-root of
#'         column 'variance.estimates' is the standard error of CATE.
#'         For out-of-bag estimates, we also output the following error measures.
#'         First, column 'debiased.error' contains estimates of the 'R-loss' criterion,
#          a quantity that is related to the true (infeasible) mean-squared error
#'         (See Nie and Wager 2017 for a justification). Second, column 'excess.error'
#'         contains jackknife estimates of the Monte-carlo error (Wager, Hastie, Efron 2014),
#'         a measure of how unstable estimates are if we grow forests of the same size
#'         on the same data set. The sum of 'debiased.error' and 'excess.error' is the raw error
#'         attained by the current forest, and 'debiased.error' alone is an estimate of the error
#'         attained by a forest with an infinite number of trees. We recommend that users grow
#'         enough forests to make the 'excess.error' negligible.
#'
#' @examples
#' \donttest{
#' # Train a causal forest.
#' n <- 100
#' p <- 10
#' X <- matrix(rnorm(n * p), n, p)
#' W <- rbinom(n, 1, 0.5)
#' Y <- pmax(X[, 1], 0) * W + X[, 2] + pmin(X[, 3], 0) + rnorm(n)
#' c.forest <- causal_forest(X, Y, W)
#'
#' # Predict using the forest.
#' X.test <- matrix(0, 101, p)
#' X.test[, 1] <- seq(-2, 2, length.out = 101)
#' c.pred <- predict(c.forest, X.test)
#'
#' # Predict on out-of-bag training samples.
#' c.pred <- predict(c.forest)
#'
#' # Predict with confidence intervals; growing more trees is now recommended.
#' c.forest <- causal_forest(X, Y, W, num.trees = 500)
#' c.pred <- predict(c.forest, X.test, estimate.variance = TRUE)
#' }
#' @method predict balanced_causal_forest
#' @export
predict.balanced_causal_forest <- function(object,
                                           newdata = NULL,
                                           linear.correction.variables = NULL,
                                           ll.lambda = NULL,
                                           ll.weight.penalty = FALSE,
                                           num.threads = NULL,
                                           estimate.variance = FALSE,
                                           ...) {
  local.linear <- !is.null(linear.correction.variables)
  allow.na <- !local.linear

  # If possible, use pre-computed predictions.
  if (is.null(newdata) &&
      !estimate.variance &&
      !local.linear &&
      !is.null(object$predictions)) {
    return(
      data.frame(
        predictions = object$predictions,
        debiased.error = object$debiased.error,
        excess.error = object$excess.error
      )
    )
  }

  num.threads <- validate_num_threads(num.threads)
  forest.short <- object[-which(names(object) == "X.orig")]
  X <- object[["X.orig"]]
  Y.centered <- object[["Y.orig"]] - object[["Y.hat"]]
  W.centered <- object[["W.orig"]] - object[["W.hat"]]
  target.weights <-
    train.data <-
    create_train_matrices(X, outcome = Y.centered, treatment = W.centered)

  if (local.linear) {
    linear.correction.variables <-
      validate_ll_vars(linear.correction.variables, ncol(X))

    if (is.null(ll.lambda)) {
      ll.regularization.path <- tune_ll_causal_forest(object,
                                                      linear.correction.variables,
                                                      ll.weight.penalty,
                                                      num.threads)
      ll.lambda <- ll.regularization.path$lambda.min
    } else {
      ll.lambda <- validate_ll_lambda(ll.lambda)
    }

    # Subtract 1 to account for C++ indexing
    linear.correction.variables <- linear.correction.variables - 1
  }
  args <- list(
    forest.object = forest.short,
    num.threads = num.threads,
    estimate.variance = estimate.variance
  )
  ll.args <- list(
    ll.lambda = ll.lambda,
    ll.weight.penalty = ll.weight.penalty,
    linear.correction.variables = linear.correction.variables
  )

  if (!is.null(newdata)) {
    validate_newdata(newdata, X, allow.na = allow.na)
    test.data <- create_test_matrices(newdata)
    if (!local.linear) {
      ret <- do.call.rcpp(causal_predict, c(train.data, test.data, args))
    } else {
      ret <-
        do.call.rcpp(ll_causal_predict,
                     c(train.data, test.data, args, ll.args))
    }
  } else {
    if (!local.linear) {
      ret <- do.call.rcpp(causal_predict_oob, c(train.data, args))
    } else {
      ret <-
        do.call.rcpp(ll_causal_predict_oob, c(train.data, args, ll.args))
    }
  }

  # Convert list to data frame.
  empty <- sapply(ret, function(elem)
    length(elem) == 0)
  do.call(cbind.data.frame, ret[!empty])
}



