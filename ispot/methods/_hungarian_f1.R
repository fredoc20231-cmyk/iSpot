#!/usr/bin/env Rscript
# Hungarian F1 matching — shared by all R method runners
# Matches the Python ispot/metrics.py implementation exactly

match_clusters_to_labels <- function(true_labels, pred_labels) {
  true_labels <- as.character(true_labels); pred_labels <- as.character(pred_labels)
  true_classes <- unique(true_labels); pred_classes <- unique(pred_labels)
  cost_matrix <- matrix(0, nrow=length(pred_classes), ncol=length(true_classes))
  for (i in seq_along(pred_classes)) for (j in seq_along(true_classes))
    cost_matrix[i,j] <- -sum(pred_labels==pred_classes[i] & true_labels==true_classes[j])
  n <- max(nrow(cost_matrix), ncol(cost_matrix))
  padded <- matrix(0, n, n)
  padded[seq_len(nrow(cost_matrix)), seq_len(ncol(cost_matrix))] <- cost_matrix
  padded <- padded - min(padded)
  assignment <- clue::solve_LSAP(padded)
  mapping <- setNames(rep(NA_character_, length(pred_classes)), pred_classes)
  for (i in seq_along(pred_classes)) { j <- assignment[i]; if (j <= length(true_classes)) mapping[pred_classes[i]] <- true_classes[j] }
  mapping[is.na(mapping)] <- "unmatched"
  unname(mapping[pred_labels])
}

.f1_multiclass <- function(true, pred, weighted=FALSE) {
  classes <- union(unique(true), unique(pred))
  f1s <- numeric(length(classes)); support <- numeric(length(classes))
  for (i in seq_along(classes)) {
    cls <- classes[i]
    tp <- sum(pred==cls & true==cls); fp <- sum(pred==cls & true!=cls); fn <- sum(pred!=cls & true==cls)
    precision <- if(tp+fp==0) 0 else tp/(tp+fp)
    recall <- if(tp+fn==0) 0 else tp/(tp+fn)
    f1s[i] <- if(precision+recall==0) 0 else 2*precision*recall/(precision+recall)
    support[i] <- sum(true==cls)
  }
  if (weighted) sum(f1s*support)/sum(support) else mean(f1s)
}

calculate_F1_hungarian <- function(true_labels, pred_labels) {
  pred_mapped <- match_clusters_to_labels(true_labels, pred_labels)
  list(macro=.f1_multiclass(true_labels, pred_mapped, weighted=FALSE),
       weighted=.f1_multiclass(true_labels, pred_mapped, weighted=TRUE))
}
