args <- commandArgs(trailingOnly = TRUE)
if (!requireNamespace("eyeprocess", quietly = TRUE)) stop("Frozen eyeprocess 0.11.1 must be installed for oracle generation.")
stopifnot(as.character(utils::packageVersion("eyeprocess")) == "0.11.1")
cat("eyeprocess R oracle ready: 0.11.1\n")
