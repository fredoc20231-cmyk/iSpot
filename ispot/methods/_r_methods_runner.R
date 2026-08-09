#!/usr/bin/env Rscript
# Shared R runner for BISON, SpaRTaCo, spatialMNN
# Usage: Rscript _r_methods_runner.R <h5ad_path> <n_clusters> <dataset> <slide_id> <seed> <method>
suppressWarnings(suppressMessages({
  .libPaths(c("/workspace/.Rlib", .libPaths()))
  library(SingleCellExperiment)
  library(SpatialExperiment)
  library(Matrix)
  library(dplyr)
  library(Seurat)
  library(scry)
  library(igraph)
  library(dbscan)
  library(RANN)
  library(clue)
  library(mclust)
  library(invgamma)
  library(ggplot2)
  library(ggnewscale)
  library(ggrepel)
  library(Rcpp)
  library(RcppArmadillo)
  library(zellkonverter)
  library(scater)
  library(scran)
  library(jsonlite)
  library(glmpca)

  # Source spatialMNN
  source("/mnt/shared-workspace/shared/models/spatialMNN/R/spatialMNN.R")
  source("/mnt/shared-workspace/shared/models/spatialMNN/R/utils.R")
  source("/mnt/shared-workspace/shared/models/spatialMNN/R/plotting_functions.R")

  # Source BISON (needs gnu++14 for C++ compilation)
  Sys.setenv("PKG_CXXFLAGS"="-std=gnu++14")
  setwd("/mnt/shared-workspace/shared/models/BISON")
  source("R/main.R")
  source("R/utils.R")

  # spartaco is an installed package
  library(spartaco)

  # Hungarian F1
  source("/workspace/ispot/methods/_hungarian_f1.R")
}))

args <- commandArgs(trailingOnly=TRUE)
h5ad_path <- args[1]
n_clusters <- as.integer(args[2])
dataset <- args[3]
slide_id <- args[4]
seed <- as.integer(args[5])
method <- args[6]
set.seed(seed)

# Load data
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
  spatial <- as.matrix(colData(sce)[, c("array_col", "array_row")])
  reducedDim(sce, "spatial") <- spatial
}
colData(sce)$col <- round(spatial[,1])
colData(sce)$row <- round(spatial[,2])

cat("Loaded:", ncol(sce), "spots x", nrow(sce), "genes\n")

# Helper functions
get_median_nn_distance <- function(coords, k=6) { nn <- RANN::nn2(coords, k=k+1); median(nn$nn.dists[,k+1]) }
rescale_spatial_coords <- function(sce, ref_dist, k=6) {
  coords <- reducedDim(sce, "spatial"); cur <- get_median_nn_distance(coords, k)
  reducedDim(sce, "spatial") <- coords * (ref_dist/cur); sce
}

compute_size_gene_factors <- function(count_mat) {
  list(sizefactor=colSums(count_mat)/mean(colSums(count_mat)),
       genefactor=rowMeans(count_mat)/mean(rowMeans(count_mat)))
}

sce2Seurat <- function(sce) {
  counts_mat <- as.matrix(assay(sce, "counts"))
  coords <- reducedDim(sce, "spatial")
  meta_df <- data.frame(coord_x=as.numeric(coords[,1]), coord_y=as.numeric(coords[,2]), row.names=colnames(sce))
  if ("annotation" %in% colnames(colData(sce))) meta_df$layer <- colData(sce)$annotation
  CreateSeuratObject(counts=counts_mat, meta.data=meta_df)
}

calculate_ARI <- function(sce, predicted) {
  ann <- colData(sce)$annotation; pred <- colData(sce)[[predicted]]
  valid <- !is.na(ann) & !is.na(pred)
  mclust::adjustedRandIndex(ann[valid], pred[valid])
}

# ---- Method runners ----

run_BISON <- function(sce, level, platform="Visium") {
  sce_log <- scater::logNormCounts(sce)
  dec <- scran::modelGeneVar(sce_log, assay.type="logcounts")
  top <- scran::getTopHVGs(dec, n=1000); sce_hvg <- sce[top,]
  Adj <- find_neighbors(sce_hvg, platform=platform, coordinate="lattice")
  neighbors <- find_neighbor_index(Adj, platform=platform)
  count_mat <- as.matrix(assay(sce_hvg, "counts")); P <- nrow(count_mat); N <- ncol(count_mat)
  factors <- compute_size_gene_factors(count_mat)
  sg <- matrix(rep(factors$sizefactor, each=P), P, N) * matrix(rep(factors$genefactor, N), P, N)
  assign("L", level, envir=.GlobalEnv)
  result <- BISON(count_mat=count_mat, sg=sg, neighbors=neighbors, K=level, R=level, f=1, n_iters=1000, seed=seed)
  predicted_key <- paste0("bison_spot_label_", level)
  colData(sce)[[predicted_key]] <- factor(result$pred_spot_label)
  list(sce=sce, predicted_key=predicted_key)
}

run_spatialMNN <- function(sce, level, sample_name) {
  n_spots <- ncol(sce)
  if (n_spots < 200) { nn<-3; cl_res<-0.5; cl_min<-2 } else { nn<-6; cl_res<-10; cl_min<-5 }
  seu <- sce2Seurat(sce); seu@meta.data[["orig.ident"]] <- sample_name
  seu_ls <- setNames(list(seu), sample_name)
  seu_ls <- stage_1(seu_ls, cor_threshold=0.6, nn=nn, cl_resolution=cl_res, top_pcs=8, cl_min=cl_min,
                    find_HVG=TRUE, hvg=2000, cor_met="PC", edge_smoothing=TRUE, use_glmpca=TRUE, verbose=FALSE)
  rtn_ls <- stage_2(seu_ls, cl_key="merged_cluster", rtn_seurat=TRUE, top_pcs=8, use_glmpca=TRUE,
                    method="louvain", resolution=1, find_HVG=TRUE, hvg=2000, cor_met="PC")
  seu_ls <- assign_label(seu_ls, rtn_ls$cl_df, anno="louvain", cor_threshold=0.6, cl_key="merged_cluster")
  predicted_key <- paste0("spatialMNN_spot_label_", level)
  colData(sce)[[predicted_key]] <- factor(seu_ls[[sample_name]]@meta.data[["sec_cluster_louvain"]])
  list(sce=sce, predicted_key=predicted_key)
}

run_spartaco <- function(sce, level) {
  counts <- as.matrix(assay(sce, "counts"))
  dev <- scry::devianceFeatureSelection(counts)
  keep <- order(dev, decreasing=TRUE)[seq_len(min(2000, length(dev)))]
  counts <- counts[keep,,drop=FALSE]
  counts <- counts[rowSums(counts)>0, colSums(counts)>0, drop=FALSE]
  coordinates <- reducedDim(sce, "spatial")[colnames(counts),,drop=FALSE]
  fit <- spartaco::spartaco(data=counts, coordinates=coordinates, K=10, R=level, max.iter=1000,
                            conv.criterion=list(epsilon=0.01, iterations=5))
  predicted_key <- paste0("spartaco_spot_label_", level)
  spot_labels <- setNames(factor(fit$Ds), colnames(counts))
  colData(sce)[[predicted_key]] <- spot_labels[colnames(sce)]
  list(sce=sce, predicted_key=predicted_key)
}

# ---- Run the requested method ----
t0 <- Sys.time()
result <- tryCatch({
  if (method == "spatialMNN") {
    if (dataset != "DLPFC") {
      ref <- zellkonverter::readH5AD("/mnt/shared-workspace/shared/data/dlpfc_h5ad/151507.h5ad")
      ref_dist <- get_median_nn_distance(reducedDim(ref, "spatial"))
      sce <- rescale_spatial_coords(sce, ref_dist)
    }
    res <- run_spatialMNN(sce, n_clusters, slide_id)
  } else if (method == "BISON") {
    platform <- if (dataset=="DLPFC") "Visium" else "ST"
    res <- run_BISON(sce, n_clusters, platform)
  } else if (method == "SpaRTaCo") {
    res <- run_spartaco(sce, n_clusters)
  } else {
    stop(paste("Unknown method:", method))
  }
  runtime <- as.numeric(difftime(Sys.time(), t0, units="secs"))

  sce <- res$sce; predicted_key <- res$predicted_key
  ari <- calculate_ARI(sce, predicted_key)

  ann_valid <- as.character(colData(sce)$annotation)
  pred_valid <- as.character(colData(sce)[[predicted_key]])
  valid <- !is.na(ann_valid) & !is.na(pred_valid)
  f1 <- calculate_F1_hungarian(ann_valid[valid], pred_valid[valid])

  all_labels <- as.character(colData(sce)[[predicted_key]])

  list(ari=ari, macro_f1=f1$macro, weighted_f1=f1$weighted,
       runtime=runtime, n_spots=ncol(sce),
       n_clusters_pred=length(unique(colData(sce)[[predicted_key]])),
       n_clusters_true=n_clusters,
       labels=all_labels)
}, error=function(e) {
  list(error=conditionMessage(e))
})

cat("RESULT:", jsonlite::toJSON(result, auto_unbox=TRUE), "\n")
