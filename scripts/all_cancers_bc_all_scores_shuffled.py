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
tumor_type  = sys.argv[1]
gene_list   = sys.argv[2]
score_type  = sys.argv[3]

# Paths
input_dir           = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/pnet_gene/all_cancers/aggregated_scores"
cancer_genes_list   = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/resources/gene_lists/CancerGenesList.csv"
output_dir          = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/pnet_gene/all_cancers/bc/"+gene_list+"_all_scores/"+tumor_type+"/"+score_type+"/plots"
tumor_types_path    = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/resources/sample_cancer_type/all_cancers/cancer_type_"+tumor_type+".csv"

# Load ALL scores and sample data
genetic_data = dict()
for agg_func in ["avg", "sd", "max", "min", "delta"]:
    scores_hap1, scores_hap2 = util.load_hap_scores(input_dir+"/"+score_type+"/"+agg_func+"/", agg_func, score_type)
    genetic_data[agg_func+"_"+score_type+"_hap1"] = scores_hap1
    genetic_data[agg_func+"_"+score_type+"_hap2"] = scores_hap2

# Load tumor types
tumor_types =  pd.read_csv(tumor_types_path, sep=",").dropna().set_index("sample")
tumor_types[tumor_types.columns[0]] = tumor_types[tumor_types.columns[0]].astype(int)

# Chose gene list to operate on
if gene_list == "cancer_genes":
    # Load cancer genes list
    selected_genes = list(pd.read_csv(cancer_genes_list)['hgnc'])
else:
    # Select all genes in dataset
    selected_genes = list(scores_hap1.columns)

# Get samples list
samples = np.array(tumor_types.index.tolist())

# Run P-NET on shuffled labels and save layer importances 
# to compute differential importance for each node in each layer
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_gene_feature_importances = []
all_additional_feature_importances = []
all_gene_importances = []
all_layer_importance_scores = []

for fold, (train_index, test_index) in enumerate(kf.split(samples, tumor_types[tumor_types.columns[0]])):
    train_sample = samples[train_index].tolist()
    test_sample = samples[test_index].tolist()

    model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(
        genetic_data, tumor_types, seed=0, dropout=0.2, lr=1e-3, weight_decay=1e-3,
        batch_size=128, epochs=3000, early_stopping=True, train_inds=train_sample,
        test_inds=test_sample, input_dropout=0.5, gene_set=selected_genes, shuffle_labels=True
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

# Aggregate and save final results
if not os.path.exists(output_dir+"_shuffled_labels_run"):
    os.makedirs(output_dir+"_shuffled_labels_run")

# Average importance scores and folds
avg_gene_feature_importances = pd.concat(all_gene_feature_importances).groupby(level=0).mean()
avg_additional_feature_importances = pd.concat(all_additional_feature_importances).groupby(level=0).mean()
avg_gene_importances = pd.concat(all_gene_importances).groupby(level=0).mean()

avg_gene_feature_importances.to_csv(f"{output_dir+"_shuffled_labels_run"}/gene_feature_importances.csv")
avg_additional_feature_importances.to_csv(f"{output_dir+"_shuffled_labels_run"}/additional_feature_importances.csv")
avg_gene_importances.to_csv(f"{output_dir+"_shuffled_labels_run"}/gene_importances.csv")

if all_layer_importance_scores and all_layer_importance_scores[0]:
    for i in range(len(all_layer_importance_scores[0])):
        layer_scores = [fold_scores[i] for fold_scores in all_layer_importance_scores]
        avg_layer_score = pd.concat(layer_scores).groupby(level=0).mean()
        avg_layer_score.to_csv(f"{output_dir+"_shuffled_labels_run"}/layer_{i}_importances.csv")