import sys
import os
sys.path.append(os.path.dirname(__file__)+"/../src/")

import pandas as pd
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from pnet import Pnet
from util import util, sankey_diag

# Read params
tumor_type  = sys.argv[1]
gene_list   = sys.argv[2]
score_type  = sys.argv[3]

# Paths
input_dir           = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/pnet_gene/all_cancers/aggregated_scores"
cancer_genes_list   = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/resources/gene_lists/CancerGenesList.csv"
tumor_types_path    = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/resources/sample_cancer_type/all_cancers/cancer_type_"+tumor_type+".csv"
output_dir          = sys.argv[4]

# Load ALL scores and sample data
genetic_data = dict()
for agg_func in ["avg", "sd", "max", "min", "delta"]:
    scores_hap1, scores_hap2 = util.load_hap_scores(input_dir+"/"+score_type+"/"+agg_func+"/", agg_func, score_type)
    genetic_data[agg_func+"_"+score_type+"_hap1"] = scores_hap1
    genetic_data[agg_func+"_"+score_type+"_hap2"] = scores_hap2

# Load tumor types and shuffle labels
tumor_types =  pd.read_csv(tumor_types_path, sep=",").dropna().set_index("sample")
tumor_types[tumor_types.columns[0]] = tumor_types[tumor_types.columns[0]].astype(int)
tumor_types[tumor_types.columns[0]] = tumor_types.sample(frac=1).values

# Chose gene list to operate on
if gene_list == "cancer_genes":
    # Load cancer genes list
    selected_genes = list(pd.read_csv(cancer_genes_list)['hgnc'])
else:
    # Select all genes in dataset
    selected_genes = list(scores_hap1.columns)

# Get samples list
samples = np.array(tumor_types.index.tolist())

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_gene_feature_importances = []
all_additional_feature_importances = []
all_gene_importances = []
all_layer_importance_scores = []
all_y_true = []
all_pred_proba = []

for fold, (train_index, test_index) in enumerate(kf.split(samples, tumor_types[tumor_types.columns[0]])):
    train_sample = util.balance_split(samples[train_index].tolist(), tumor_types)
    test_sample = util.balance_split(samples[test_index].tolist(), tumor_types)

    model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(
        genetic_data, tumor_types, seed=0, dropout=0.2, lr=1e-3, weight_decay=1e-3,
        batch_size=128, epochs=3000, early_stopping=True, train_inds=train_sample,
        test_inds=test_sample, input_dropout=0.5, gene_set=selected_genes, shuffle_labels=False
    )
    # Perform evaluation, generate plots and importances csv and save
    plt.clf()
    results = Pnet.evaluate_and_interpret(
        model,
        test_dataset,
        tumor_types.columns.values
    )
    all_gene_feature_importances.append(results['gene_feature_importances'])
    all_additional_feature_importances.append(results['additional_feature_importances'])
    all_gene_importances.append(results['gene_importances'])
    all_layer_importance_scores.append(results['layer_importance_scores'])
    all_y_true.append(results['y_true'])
    all_pred_proba.append(results['pred_proba'])

# Save all_y_true and all_y_pred as CSV
all_y_true_df = pd.concat([pd.DataFrame(y.cpu().numpy()) for y in all_y_true])
all_pred_proba_df = pd.concat([pd.DataFrame(p.cpu().numpy()) for p in all_pred_proba])
all_y_true_df.to_csv(f"{output_dir}/all_y_true.csv", index=False)
all_pred_proba_df.to_csv(f"{output_dir}/all_y_pred_proba.csv", index=False)

if all_y_true and all_pred_proba:
    # F1 at 0.5 threshold
    f1_05_scores = [util.get_f1((pred_proba > 0.5), y_true.to(torch.int)) for y_true, pred_proba in zip(all_y_true, all_pred_proba)]
    f1_05_scores = [s.item() if isinstance(s, torch.Tensor) else s for s in f1_05_scores]
    mean_f1_05 = np.mean(f1_05_scores)
    std_f1_05 = np.std(f1_05_scores)

    # F1 at best threshold
    f1_best_data = [util.get_best_f1(pred_proba, y_true) for y_true, pred_proba in zip(all_y_true, all_pred_proba)]
    f1_best_scores = [x[0] for x in f1_best_data]
    best_thresholds = [x[1] for x in f1_best_data]
    mean_f1_best = np.mean(f1_best_scores)
    std_f1_best = np.std(f1_best_scores)
    mean_best_thresh = np.mean(best_thresholds)
    std_best_thresh = np.std(best_thresholds)

    # Generate and save mean ROC and PRC plots
    mean_roc_auc, std_roc_auc = util.plot_mean_roc_curve(all_y_true, all_pred_proba, tumor_types.columns.values, f"{output_dir}/roc_auc_curve.pdf")
    mean_prc_auc, std_prc_auc = util.plot_mean_prc_curve(all_y_true, all_pred_proba, f"{output_dir}/prc_auc_curve.pdf")
    
    # Save mean AUC scores
    pd.DataFrame({
        'mean_roc_auc': [mean_roc_auc],
        'std_roc_auc':  [std_roc_auc],
        'mean_prc_auc': [mean_prc_auc],
        'std_prc_auc':  [std_prc_auc],
        'mean_f1_05': [mean_f1_05],
        'std_f1_05':  [std_f1_05],
        'mean_f1_best': [mean_f1_best],
        'std_f1_best':  [std_f1_best],
        'mean_best_threshold': [mean_best_thresh],
        'std_best_threshold': [std_best_thresh]
    }).to_csv(f"{output_dir}/mean_auc_scores.csv", index=False)

# Average importance scores and folds
avg_gene_feature_importances = pd.concat(all_gene_feature_importances).groupby(level=0).mean()
avg_additional_feature_importances = pd.concat(all_additional_feature_importances).groupby(level=0).mean()
avg_gene_importances = pd.concat(all_gene_importances).groupby(level=0).mean()

avg_gene_feature_importances.to_csv(f"{output_dir}/gene_feature_importances.csv")
avg_additional_feature_importances.to_csv(f"{output_dir}/additional_feature_importances.csv")
avg_gene_importances.to_csv(f"{output_dir}/gene_importances.csv")

# Prepare importance dataframes for Sankey diagram
avg_layer_importance_scores = []
if all_layer_importance_scores and all_layer_importance_scores[0]:
    for i in range(len(all_layer_importance_scores[0])):
        layer_scores = [fold_scores[i] for fold_scores in all_layer_importance_scores]
        avg_layer_score = pd.concat(layer_scores).groupby(level=0).mean()
        avg_layer_score.to_csv(f"{output_dir}/layer_{i}_importances.csv")
        avg_layer_importance_scores.append(avg_layer_score)
