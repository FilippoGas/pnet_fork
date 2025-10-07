import sys
import os
sys.path.append(os.path.dirname(__file__)+"/../src/")

from pnet import Pnet
from util import util, sankey_diag
import pickle


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from random import sample


# agg_func = sys.argv[1]
# tumor_type = sys.argv[2]

# Pahts
input_dir = "/home/filippo.gastaldello/data/pnet_fork/pnet_gene/breast_subtype/aggregated_scores/"+agg_func+"/"
cancer_genes_list = "/home/filippo.gastaldello/data/pnet_fork/resources/gene_lists/CancerGenesList.csv"
tumor_subtypes_path = "/home/filippo.gastaldello/data/pnet_fork/resources/cancer_subtype_"+tumor_type+".csv"

# Load scores and sample data
scores_hap1, scores_hap2 = util.load_hap_scores(input_dir, agg_func)
genetic_data = {'scores_hap1':scores_hap1,
                'scores_hap2':scores_hap2}

# Load tumor subtypes
tumor_subtypes =  pd.get_dummies(pd.read_csv(tumor_subtypes_path, sep=",").dropna().set_index("samples"), prefix=None)

# Load cancer genes list
canc_genes = list(pd.read_csv(cancer_genes_list)['hgnc'])

# Split samples in train/test
samples = scores_hap1.index.tolist()
train_sample = sample(samples, round(len(samples)*0.7))
test_sample = list(set(samples)-set(train_sample))

model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(genetic_data, tumor_subtypes, seed=0, dropout=0.2, lr=1e-3, weight_decay=1e-3,
                                                                           batch_size=64, epochs=300, early_stopping=True, train_inds=train_sample,
                                                                           test_inds=test_sample, input_dropout=0.5, gene_set=canc_genes)

# Perform evaluation, generate plots and importances csv and save
plt.clf()
gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores = Pnet.evaluate_interpret_save(model, test_dataset, tumor_subtypes.columns.values, "/home/filippo.gastaldello/data/pnet_fork/pnet_gene/breast_subtype/mc/plots/"+agg_func)

# Prepare importance dataframes for Sankey diagram
layer_list = [gene_feature_importances, additional_feature_importances, gene_feature_importances] + layer_importance_scores
layer_list_names = ['gene_feature', 'additional_feature', 'gene'] + [f'layer_{i}' for i in range(5)]
layer_list_dict = dict(zip(layer_list_names, layer_list))

# Plot Sankey
sk = sankey_diag.SankeyDiag(layer_list_dict, runs=1)
fig = sk.get_sankey_diag("/home/filippo.gastaldello/data/pnet_fork/pnet_gene/breast_subtype/mc/plots/"+agg_func+"/sankey.html")