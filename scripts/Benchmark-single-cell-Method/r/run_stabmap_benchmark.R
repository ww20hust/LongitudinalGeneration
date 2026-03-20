args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript run_stabmap_benchmark.R <prepared_dir> <output_dir>")
}

prepared_dir <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
repo_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
stabmap_r_dir <- file.path(repo_root, "baseline-model", "StabMap", "R")
for (path in list.files(stabmap_r_dir, pattern = "\\.R$", full.names = TRUE)) {
  source(path)
}

train_lab <- as.matrix(read.csv(file.path(prepared_dir, "train_lab_scaled.csv"), row.names = 1, check.names = FALSE))
train_metab <- as.matrix(read.csv(file.path(prepared_dir, "train_metab_scaled.csv"), row.names = 1, check.names = FALSE))
test_lab <- as.matrix(read.csv(file.path(prepared_dir, "test_lab_scaled.csv"), row.names = 1, check.names = FALSE))
test_metab <- as.matrix(read.csv(file.path(prepared_dir, "test_metab.csv"), row.names = 1, check.names = FALSE))

scaler_stats <- jsonlite::fromJSON(file.path(prepared_dir, "scaler_stats.json"))
metab_mean <- as.numeric(scaler_stats$metab$mean)
metab_scale <- as.numeric(scaler_stats$metab$scale)
metab_columns <- scaler_stats$metab_columns

paired_mat <- rbind(t(train_lab), t(train_metab))
query_mat <- t(test_lab)
colnames(paired_mat) <- rownames(train_lab)
colnames(query_mat) <- rownames(test_lab)

assay_list <- list(
  paired = paired_mat,
  query = query_mat
)

n_ref <- max(1, min(20, ncol(paired_mat) - 1))
n_sub <- max(1, min(20, ncol(query_mat), ncol(paired_mat) - 1))

embedding <- stabMap(
  assay_list,
  reference_list = c("paired"),
  ncomponentsReference = n_ref,
  ncomponentsSubset = n_sub,
  plot = FALSE,
  scale.center = FALSE,
  scale.scale = FALSE,
  suppressMessages = TRUE
)

imputed <- imputeEmbedding(
  assay_list,
  embedding,
  reference = colnames(assay_list[["paired"]]),
  query = colnames(assay_list[["query"]])
)

pred_scaled <- t(imputed[["paired"]][colnames(train_metab), , drop = FALSE])
colnames(pred_scaled) <- metab_columns
rownames(pred_scaled) <- rownames(test_lab)
pred_raw <- sweep(pred_scaled, 2, metab_scale, `*`)
pred_raw <- sweep(pred_raw, 2, metab_mean, `+`)

write.csv(pred_scaled, file.path(output_dir, "test_metab_pred_scaled.csv"), quote = FALSE)
write.csv(pred_raw, file.path(output_dir, "test_metab_pred.csv"), quote = FALSE)

mae <- mean(abs(as.matrix(test_metab) - pred_raw))
mse <- mean((as.matrix(test_metab) - pred_raw) ^ 2)
feature_corr <- sapply(seq_len(ncol(pred_raw)), function(i) {
  suppressWarnings(cor(test_metab[, i], pred_raw[, i], method = "pearson"))
})
feature_corr[is.na(feature_corr)] <- 0
metrics <- list(
  n_samples = nrow(test_metab),
  n_features = ncol(test_metab),
  mae = unname(mae),
  mse = unname(mse),
  pearson_mean = unname(mean(feature_corr)),
  pearson_median = unname(median(feature_corr)),
  pearson_min = unname(min(feature_corr)),
  pearson_max = unname(max(feature_corr))
)
jsonlite::write_json(metrics, file.path(output_dir, "metrics.json"), pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("StabMap benchmark completed: %s\n", output_dir))
