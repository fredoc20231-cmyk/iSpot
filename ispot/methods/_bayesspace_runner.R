#!/usr/bin/env Rscript
# BayesSpace runner — called by ispot/methods/bayesspace.py
# Usage: Rscript _bayesspace_runner.R <h5ad_path> <n_clusters> <dataset> <slide_id> <seed>
suppressWarnings(suppressMessages({
  .libPaths(c("/workspace/.Rlib", .libPaths()))
  library(SingleCellExperiment)
  library(SpatialExperiment)
  library(Matrix)
  library(BayesSpace)
  library(zellkonverter)
  library(mclust)
  library(scater)
}))

args <- commandArgs(trailingOnly=TRUE)
h5ad_path <- args[1]
n_clusters <- as.integer(args[2])
dataset <- args[3]
slide_id <- args[4]
seed <- as.integer(args[5])
set.seed(seed)

sce <- zellkonverter::readH5AD(h5ad_path)
if (!"counts" %in% names(assays(sce))) assayNames(sce) <- "counts"

# Preprocess
if ("in_tissue" %in% colnames(colData(sce))) sce <- sce[, colData(sce)$in_tissue==1]
if (!"annotation" %in% colnames(colData(sce)) && "ground_truth" %in% colnames(colData(sce)))
  colData(sce)$annotation <- colData(sce)$ground_truth
ann <- colData(sce)$annotation
ann[ann == "NA" | ann == "nan" | ann == ""] <- NA
colData(sce)$annotation <- ann
rownames(sce) <- make.unique(rownames(sce))
colData(sce)$slide_id <- slide_id

# Spatial coords
spatial <- reducedDim(sce, "spatial")
if (is.null(spatial)) {
  # Try obsm
  spatial <- as.matrix(colData(sce)[, c("array_col", "array_row")])
  reducedDim(sce, "spatial") <- spatial
}
colData(sce)$col <- round(spatial[,1])
colData(sce)$row <- round(spatial[,2])

# QC
n_spots <- ncol(sce)
n_genes <- nrow(sce)
cat("Loaded:", n_spots, "spots x", n_genes, "genes\n")

# BayesSpace requires SpatialExperiment
spe <- SpatialExperiment(
  assays = list(counts = assay(sce, "counts")),
  colData = colData(sce),
  spatialCoords = spatial
)

# Log-normalize
spe <- logNormCounts(spe)

# Run PCA for BayesSpace (requires reducedDim "PCA")
spe <- scater::runPCA(spe, ncomponents=15)

# Run BayesSpace clustering
t0 <- Sys.time()
result <- tryCatch({
  # q = number of clusters
  spe <- spatialCluster(spe, q=n_clusters, d=15, platform="Visium",
                        nrep=1000, gamma=2, model="t",
                        burn.in=100, init.method="mclust")
  runtime <- as.numeric(difftime(Sys.time(), t0, units="secs"))

  pred <- colData(spe)$spatial.cluster
  ann_valid <- colData(spe)$annotation
  valid <- !is.na(ann_valid) & !is.na(pred)
  ann_valid <- ann_valid[valid]; pred_valid <- pred[valid]

  ari <- mclust::adjustedRandIndex(ann_valid, pred_valid)

  # F1 with Hungarian matching
  source("/workspace/ispot/methods/_hungarian_f1.R")
  f1 <- calculate_F1_hungarian(ann_valid, pred_valid)

  list(ari=ari, macro_f1=f1$macro, weighted_f1=f1$weighted,
       runtime=runtime, n_spots=n_spots,
       n_clusters_pred=length(unique(pred)),
       n_clusters_true=n_clusters,
       labels=as.character(pred))
}, error=function(e) {
  list(error=conditionMessage(e))
})

output <- if (!is.null(result$error)) {
  list(error=result$error)
} else {
  result
}
cat("RESULT:", jsonlite::toJSON(output, auto_unbox=TRUE), "\n")
