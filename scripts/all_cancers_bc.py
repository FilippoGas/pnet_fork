import sys
import os
sys.path.append(os.path.dirname(__file__)+"/../src/")

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from pnet import Pnet
from util import util, sankey_diag

# Read params
agg_func    = sys.argv[1] # Function used to aggregate transcript scores into gene scores
score_type  = sys.argv[2]
tumor_type  = sys.argv[3]
gene_list   = sys.argv[4]

# Paths
input_dir           = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/pnet_gene/all_cancers/aggregated_scores/"+score_type+"/"+agg_func+"/"
cancer_genes_list   = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/resources/gene_lists/CancerGenesList.csv"
output_dir          = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/pnet_gene/all_cancers/bc/"+gene_list+"/"+tumor_type+"/plots/"+score_type+"/"+agg_func
tumor_types_path    = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/resources/sample_cancer_type/all_cancers/cancer_type_"+tumor_type+".csv"

# Load scores and sample data
scores_hap1, scores_hap2 = util.load_hap_scores(input_dir, agg_func, score_type)
genetic_data = {'scores_hap1':scores_hap1,
                'scores_hap2':scores_hap2}

# Load tumor types
tumor_types =  pd.read_csv(tumor_types_path, sep=",").dropna().set_index("sample")

# Chose gene list to operate on
if gene_list == "cancer_genes":
    # Load cancer genes list
    selected_genes = list(pd.read_csv(cancer_genes_list)['hgnc'])
else:
    # Select all genes in dataset
    selected_genes = list(scores_hap1.columns)

samples = np.array(scores_hap1.index.tolist())
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_gene_feature_importances = []
all_additional_feature_importances = []
all_gene_importances = []
all_layer_importance_scores = []
all_y_true = []
all_pred_proba = []

for fold, (train_index, test_index) in enumerate(kf.split(samples)):
    train_sample = samples[train_index].tolist()
    test_sample = samples[test_index].tolist()

    model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(
        genetic_data, tumor_types, seed=0, dropout=0.2, lr=1e-3, weight_decay=1e-3,
        batch_size=128, epochs=3000, early_stopping=True, train_inds=train_sample,
        test_inds=test_sample, input_dropout=0.5, gene_set=selected_genes
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

# Aggregate and save final results
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save all_y_true and all_y_pred as pickle
pickle.dump(all_pred_proba , open(f"{output_dir}/all_y_proba.pickle", "wb"))
pickle.dump(all_y_true, open(f"{output_dir}/all_y_ture.pickle", "wb"))

if all_y_true and all_pred_proba:
# Generate and save mean ROC and PRC plots
    mean_roc_auc, std_roc_auc = util.plot_mean_roc_curve(all_y_true, all_pred_proba, tumor_types.columns.values, f"{output_dir}/roc_auc_curve.pdf")
    mean_prc_auc, std_prc_auc = util.plot_mean_prc_curve(all_y_true, all_pred_proba, f"{output_dir}/prc_auc_curve.pdf")
    
    # Save mean AUC scores
    pd.DataFrame({
        'mean_roc_auc': [mean_roc_auc],
        'std_roc_auc':  [std_roc_auc],
        'mean_prc_auc': [mean_prc_auc],
        'std_prc_auc':  [std_prc_auc]
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

layer_list          = [avg_gene_importances] + avg_layer_importance_scores
layer_list_names    = ['gene'] + [f'layer_{i}' for i in range(len(avg_layer_importance_scores))]
layer_list_dict     = dict(zip(layer_list_names, layer_list))

# Plot Sankey
sk  = sankey_diag.SankeyDiag(layer_list_dict, path='/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/resources/pathways_short_names.csv', runs=1)
fig = sk.get_sankey_diag(f"{output_dir}/sankey.html")
