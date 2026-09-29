#!/usr/bin/env Rscript

# Primary SRL transition-rate model for the gaze-claim robustness study.
#
# This research script is deliberately separate from the stable eyeprocesspy API.
# It fits one frozen estimator:
#
#   transition_count ~ prompt_indicator + factor(stimulus_id)
#                      + offset(log_exposure)
#                      + (1 | participant_id)
#
# using glmmTMB::nbinom2(link = "log").
#
# No alternative model is selected from the observed Prompt effect. Failed
# convergence is written as evidence and must remain in the planned universe.

args <- commandArgs(trailingOnly = TRUE)

arg_value <- function(flag, required = TRUE, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0L) {
    if (required) stop("Missing required argument: ", flag, call. = FALSE)
    return(default)
  }
  if (length(hit) != 1L || hit == length(args)) {
    stop("Argument must occur once and have a value: ", flag, call. = FALSE)
  }
  args[[hit + 1L]]
}

input_path <- arg_value("--input")
output_dir <- arg_value("--output-dir")
model_id <- arg_value("--model-id", required = FALSE, default = "primary_nb2_glmm")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

required_columns <- c(
  "participant_id",
  "stimulus_id",
  "experiment_condition",
  "prompt_indicator",
  "transition_count",
  "exposure_seconds",
  "log_exposure",
  "model_evaluable",
  "model_status"
)

data_all <- utils::read.csv(
  input_path,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  na.strings = c("", "NA", "NaN")
)

missing_columns <- setdiff(required_columns, names(data_all))
if (length(missing_columns)) {
  stop(
    "Model table is missing required columns: ",
    paste(missing_columns, collapse = ", "),
    call. = FALSE
  )
}

as_logical_strict <- function(x) {
  if (is.logical(x)) return(x)
  key <- tolower(trimws(as.character(x)))
  out <- rep(NA, length(key))
  out[key %in% c("true", "1")] <- TRUE
  out[key %in% c("false", "0")] <- FALSE
  if (anyNA(out)) {
    stop("model_evaluable must contain only TRUE/FALSE or 1/0.", call. = FALSE)
  }
  out
}

data_all$model_evaluable <- as_logical_strict(data_all$model_evaluable)

audit_all <- aggregate(
  rep(1L, nrow(data_all)),
  by = list(model_status = data_all$model_status),
  FUN = sum
)
names(audit_all)[[2L]] <- "rows"
utils::write.csv(
  audit_all,
  file.path(output_dir, paste0(model_id, "_row_audit.csv")),
  row.names = FALSE
)

d <- data_all[data_all$model_evaluable, , drop = FALSE]
if (nrow(d) == 0L) {
  stop("No model-evaluable rows remain. No zero-count substitution is permitted.", call. = FALSE)
}

d$transition_count <- suppressWarnings(as.numeric(d$transition_count))
d$prompt_indicator <- suppressWarnings(as.numeric(d$prompt_indicator))
d$exposure_seconds <- suppressWarnings(as.numeric(d$exposure_seconds))
d$log_exposure <- suppressWarnings(as.numeric(d$log_exposure))

if (any(!is.finite(d$transition_count)) || any(d$transition_count < 0)) {
  stop("transition_count must be finite and non-negative.", call. = FALSE)
}
if (any(abs(d$transition_count - round(d$transition_count)) > sqrt(.Machine$double.eps))) {
  stop("transition_count must contain integer counts.", call. = FALSE)
}
if (any(!is.finite(d$prompt_indicator)) ||
    any(!d$prompt_indicator %in% c(0, 1))) {
  stop("prompt_indicator must contain only 0 and 1.", call. = FALSE)
}
if (any(!is.finite(d$exposure_seconds)) || any(d$exposure_seconds <= 0)) {
  stop("exposure_seconds must be finite and > 0.", call. = FALSE)
}
if (any(!is.finite(d$log_exposure))) {
  stop("log_exposure must be finite.", call. = FALSE)
}
if (any(abs(d$log_exposure - log(d$exposure_seconds)) > 1e-10)) {
  stop("log_exposure does not equal log(exposure_seconds).", call. = FALSE)
}

condition_expected <- ifelse(d$prompt_indicator == 1, "Prompt", "Non-prompt")
if (any(trimws(d$experiment_condition) != condition_expected)) {
  stop(
    "experiment_condition disagrees with prompt_indicator for at least one row.",
    call. = FALSE
  )
}

d$participant_id <- factor(trimws(as.character(d$participant_id)))
d$stimulus_id <- factor(
  trimws(as.character(d$stimulus_id)),
  levels = sort(unique(trimws(as.character(d$stimulus_id))))
)

if (nlevels(d$participant_id) < 2L) {
  stop("At least two participants are required.", call. = FALSE)
}
if (nlevels(d$stimulus_id) < 2L) {
  stop("At least two Task identities are required.", call. = FALSE)
}
if (length(unique(d$prompt_indicator)) != 2L) {
  stop("Both Prompt and Non-prompt conditions must be represented.", call. = FALSE)
}

if (!requireNamespace("glmmTMB", quietly = TRUE)) {
  stop(
    "Package 'glmmTMB' is required for the frozen primary NB2 GLMM. ",
    "No substitute estimator is used.",
    call. = FALSE
  )
}

formula_primary <- stats::as.formula(
  "transition_count ~ prompt_indicator + factor(stimulus_id) + offset(log_exposure) + (1 | participant_id)"
)

fit_error <- NULL
fit <- tryCatch(
  glmmTMB::glmmTMB(
    formula = formula_primary,
    data = d,
    family = glmmTMB::nbinom2(link = "log"),
    na.action = stats::na.fail
  ),
  error = function(e) {
    fit_error <<- conditionMessage(e)
    NULL
  }
)

if (is.null(fit)) {
  failure <- data.frame(
    model_id = model_id,
    status = "fit_failed",
    error = fit_error,
    n_rows = nrow(d),
    n_participants = nlevels(d$participant_id),
    n_tasks = nlevels(d$stimulus_id),
    stringsAsFactors = FALSE
  )
  utils::write.csv(
    failure,
    file.path(output_dir, paste0(model_id, "_status.csv")),
    row.names = FALSE
  )
  stop("Primary glmmTMB model failed: ", fit_error, call. = FALSE)
}

summary_fit <- summary(fit)
coef_cond <- summary_fit$coefficients$cond
if (!"prompt_indicator" %in% rownames(coef_cond)) {
  stop("Fitted model did not return the frozen prompt_indicator term.", call. = FALSE)
}

estimate <- unname(coef_cond["prompt_indicator", "Estimate"])
std_error <- unname(coef_cond["prompt_indicator", "Std. Error"])
z_value <- unname(coef_cond["prompt_indicator", "z value"])
p_value <- unname(coef_cond["prompt_indicator", "Pr(>|z|)"])
z_crit <- stats::qnorm(0.975)
ci_low <- estimate - z_crit * std_error
ci_high <- estimate + z_crit * std_error

optimizer_code <- fit$fit$convergence
pd_hessian <- isTRUE(fit$sdr$pdHess)
converged <- isTRUE(optimizer_code == 0L) && pd_hessian

pearson <- stats::residuals(fit, type = "pearson")
pearson_dispersion <- sum(pearson^2, na.rm = TRUE) / stats::df.residual(fit)

prompt_result <- data.frame(
  model_id = model_id,
  term = "prompt_indicator",
  estimand_id = "prompt_transition_rate_ratio",
  estimate_log_rate_ratio = estimate,
  SE = std_error,
  CI_lower_log = ci_low,
  CI_upper_log = ci_high,
  rate_ratio = exp(estimate),
  CI_lower_rate_ratio = exp(ci_low),
  CI_upper_rate_ratio = exp(ci_high),
  z_value = z_value,
  p_value_reference_only = p_value,
  converged = converged,
  optimizer_convergence_code = optimizer_code,
  positive_definite_hessian = pd_hessian,
  n_rows = nrow(d),
  n_participants = nlevels(d$participant_id),
  n_tasks = nlevels(d$stimulus_id),
  AIC = stats::AIC(fit),
  logLik = as.numeric(stats::logLik(fit)),
  pearson_dispersion = pearson_dispersion,
  stringsAsFactors = FALSE
)

# p-value is retained only because glmmTMB returns it. It is not the robustness
# classification criterion and must not be counted across multiverse branches.
utils::write.csv(
  prompt_result,
  file.path(output_dir, paste0(model_id, "_prompt_coefficient.csv")),
  row.names = FALSE
)

fixed <- data.frame(
  term = rownames(coef_cond),
  estimate = coef_cond[, "Estimate"],
  SE = coef_cond[, "Std. Error"],
  z_value = coef_cond[, "z value"],
  p_value_reference_only = coef_cond[, "Pr(>|z|)"],
  row.names = NULL,
  stringsAsFactors = FALSE
)
utils::write.csv(
  fixed,
  file.path(output_dir, paste0(model_id, "_fixed_effects.csv")),
  row.names = FALSE
)

random_variance <- tryCatch(
  as.numeric(glmmTMB::VarCorr(fit)$cond$participant_id[1, 1]),
  error = function(e) NA_real_
)

status <- data.frame(
  model_id = model_id,
  status = if (converged) "ok" else "non_converged",
  family = "negative_binomial_2",
  link = "log",
  formula = paste(deparse(formula_primary), collapse = ""),
  n_rows = nrow(d),
  n_participants = nlevels(d$participant_id),
  n_tasks = nlevels(d$stimulus_id),
  optimizer_convergence_code = optimizer_code,
  positive_definite_hessian = pd_hessian,
  participant_random_intercept_variance = random_variance,
  glmmTMB_version = as.character(utils::packageVersion("glmmTMB")),
  stringsAsFactors = FALSE
)
utils::write.csv(
  status,
  file.path(output_dir, paste0(model_id, "_status.csv")),
  row.names = FALSE
)

if (!converged) {
  quit(status = 2L)
}

cat(
  sprintf(
    "Primary Prompt transition-rate ratio: %.6f (95%% Wald CI %.6f to %.6f); n=%d; participants=%d; tasks=%d\n",
    prompt_result$rate_ratio,
    prompt_result$CI_lower_rate_ratio,
    prompt_result$CI_upper_rate_ratio,
    nrow(d),
    nlevels(d$participant_id),
    nlevels(d$stimulus_id)
  )
)
