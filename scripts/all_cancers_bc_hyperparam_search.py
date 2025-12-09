import sys
import os
import optuna
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# Add path
sys.path.append(os.path.dirname(__file__)+"/../src/")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from pnet import Pnet
from util import util, sankey_diag

# Read params
agg_func    = sys.argv[1] 
score_type  = sys.argv[2] 
tumor_type  = sys.argv[3]

# Paths
input_dir           = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/pnet_gene/all_cancers/aggregated_scores/"+score_type+"/"+agg_func+"/"
cancer_genes_list   = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/resources/gene_lists/CancerGenesList.csv"
output_dir          = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/pnet_gene/all_cancers/bc/hyperparam_tuning_5foldCV/"+tumor_type+"/plots/"+score_type+"/"+agg_func
tumor_types_path    = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/resources/sample_cancer_type/all_cancers/cancer_type_"+tumor_type+".csv"

# Load scores and sample data
print("Loading data...")
scores_hap1, scores_hap2 = util.load_hap_scores(input_dir, agg_func, score_type)
genetic_data = {'scores_hap1':scores_hap1, 'scores_hap2':scores_hap2}

# Load tumor subtypes
tumor_types = pd.read_csv(tumor_types_path, sep=",").dropna().set_index("sample")

# Load cancer genes list
canc_genes = list(pd.read_csv(cancer_genes_list)['hgnc'])

# Prepare Samples
samples = np.array(scores_hap1.index.tolist())

# Hyperparameter search

def objective(trial):

    # 1. Suggest Hyperparameters
    dropout = trial.suggest_float("dropout", 0.0, 0.6)
    input_dropout = trial.suggest_float("input_dropout", 0.0, 0.6)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    n_splits_search = 1 
    kf_search = KFold(n_splits=n_splits_search, shuffle=True, random_state=42)
    
    fold_roc = []
    fold_prc = []
    fold_f1  = []

    for train_index, test_index in kf_search.split(samples):
        train_sample = samples[train_index].tolist()
        test_sample = samples[test_index].tolist()

        
        model, _, _, _, test_dataset = Pnet.run(
            genetic_data, tumor_types, seed=0, 
            dropout=dropout, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, epochs=1500, early_stopping=True,
            train_inds=train_sample, test_inds=test_sample, 
            input_dropout=input_dropout, gene_set=canc_genes
        )

        results = Pnet.evaluate_and_interpret(model, test_dataset, tumor_types.columns.values)
        
        y_true = results['y_true']
        pred_proba = results['pred_proba']

        try:
            if isinstance(y_true, list): y_true = np.array(y_true)
            if isinstance(pred_proba, list): pred_proba = np.array(pred_proba)

            roc = roc_auc_score(y_true, pred_proba, average='macro')
            prc = average_precision_score(y_true, pred_proba, average='macro')
            
            preds_binary = (pred_proba > 0.5).int()
            f1 = f1_score(y_true, preds_binary, average='macro')
            
            fold_roc.append(roc)
            fold_prc.append(prc)
            fold_f1.append(f1)
        except ValueError:
            # Handle edge cases (e.g. only one class present in split)
            return 0.0, 0.0, 0.0

    # Return mean metrics across folds
    return np.mean(fold_roc), np.mean(fold_prc), np.mean(fold_f1)

print("Starting Hyperparameter Search...")
# Optimize for ROCAUC, PRCAUC, F1. 
study = optuna.create_study(directions=["maximize", "maximize", "maximize"])
study.optimize(objective, n_trials=20) 

print("Search complete.")

# Select best parameters

best_trial = max(study.best_trials, key=lambda t: t.values[1]) # index 1 is PRC 

print(f"Best Trial ID: {best_trial.number}")
print(f"Best Metrics: ROC={best_trial.values[0]:.4f}, PRC={best_trial.values[1]:.4f}, F1={best_trial.values[2]:.4f}")
print("Best Hyperparameters:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Extract params
final_dropout = best_trial.params['dropout']
final_input_dropout = best_trial.params['input_dropout']
final_lr = best_trial.params['lr']
final_wd = best_trial.params['weight_decay']
final_bs = best_trial.params['batch_size']


# Run actual CV with optimized parameters
print("\nRunning Final Analysis with Optimized Parameters...")

# CV
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

all_gene_feature_importances = []
all_additional_feature_importances = []
all_gene_importances = []
all_layer_importance_scores = []
all_y_true = []
all_pred_proba = []

for fold, (train_index, test_index) in enumerate(kf.split(samples)):
    print(f"Final Run - Fold {fold+1}/{n_splits}")
    train_sample = samples[train_index].tolist()
    test_sample = samples[test_index].tolist()

    # USE OPTIMIZED PARAMS HERE
    model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(
        genetic_data, tumor_types, seed=0, 
        dropout=final_dropout, 
        lr=final_lr, 
        weight_decay=final_wd,
        batch_size=final_bs, 
        input_dropout=final_input_dropout,
        epochs=3000, 
        early_stopping=True, 
        train_inds=train_sample,
        test_inds=test_sample, 
        gene_set=canc_genes
    )

    # Perform evaluation
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

# Save results

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save Hyperparameter log
with open(f"{output_dir}/hyperparameters.txt", "w") as f:
    f.write(str(best_trial.params))

if all_y_true and all_pred_proba:
    mean_roc_auc, std_roc_auc = util.plot_mean_roc_curve(all_y_true, all_pred_proba, tumor_types.columns.values, f"{output_dir}/roc_auc_curve.pdf")
    mean_prc_auc, std_prc_auc = util.plot_mean_prc_curve(all_y_true, all_pred_proba, f"{output_dir}/prc_auc_curve.pdf")
    
    pd.DataFrame({
        'mean_roc_auc': [mean_roc_auc],
        'std_roc_auc': [std_roc_auc],
        'mean_prc_auc': [mean_prc_auc],
        'std_prc_auc': [std_prc_auc]
    }).to_csv(f"{output_dir}/mean_auc_scores.csv", index=False)

avg_gene_feature_importances = pd.concat(all_gene_feature_importances).groupby(level=0).mean()
avg_additional_feature_importances = pd.concat(all_additional_feature_importances).groupby(level=0).mean()
avg_gene_importances = pd.concat(all_gene_importances).groupby(level=0).mean()

avg_gene_feature_importances.to_csv(f"{output_dir}/gene_feature_importances.csv")
avg_additional_feature_importances.to_csv(f"{output_dir}/additional_feature_importances.csv")
avg_gene_importances.to_csv(f"{output_dir}/gene_importances.csv")

avg_layer_importance_scores = []
if all_layer_importance_scores and all_layer_importance_scores[0]:
    for i in range(len(all_layer_importance_scores[0])):
        layer_scores = [fold_scores[i] for fold_scores in all_layer_importance_scores]
        avg_layer_score = pd.concat(layer_scores).groupby(level=0).mean()
        avg_layer_score.to_csv(f"{output_dir}/layer_{i}_importances.csv")
        avg_layer_importance_scores.append(avg_layer_score)

layer_list          = [avg_gene_feature_importances, avg_additional_feature_importances, avg_gene_importances] + avg_layer_importance_scores
layer_list_names    = ['gene_feature', 'gene_feature', 'gene'] + [f'layer_{i}' for i in range(len(avg_layer_importance_scores))]
layer_list_dict     = dict(zip(layer_list_names, layer_list))

sk  = sankey_diag.SankeyDiag(layer_list_dict, path='/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/resources/pathways_short_names.csv', runs=1)
fig = sk.get_sankey_diag(f"{output_dir}/sankey.html")