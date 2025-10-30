import sys
import os
sys.path.append(os.path.dirname(__file__)+"/../src/")
import pandas as pd
import matplotlib.pyplot as plt

from pnet import Pnet
from util import util, sankey_diag
from sklearn.metrics import roc_auc_score
from random import sample

agg_func    = sys.argv[1]
score_type  = sys.argv[2]

# Paths
input_dir           = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/pnet_gene/all_cancers/aggregated_scores/"+score_type+"/"+agg_func+"/"
cancer_genes_list   = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/resources/gene_lists/CancerGenesList.csv"
tumor_types_path    = "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/resources/sample_cancer_type/all_cancers/cancer_type_mc.csv"

# Load scores
scores_hap1, scores_hap2 = util.load_hap_scores(input_dir, agg_func, score_type)
genetic_data = {'score_hap1':scores_hap1,
                'score_hap2':scores_hap2}

# Load tumor types
tumor_types = pd.get_dummies(pd.read_csv(tumor_types_path, sep=',').dropna().set_index("sample"), prefix=None)

# Load cancer genes list
canc_genes = list(pd.read_csv(cancer_genes_list)['hgnc'])

# Split samples in train/test
samples         = scores_hap1.index.tolist()
train_sample    = sample(samples, round(len(samples)*0.7))
test_sample     = list(set(samples)-set(train_sample))

model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(genetic_data, tumor_types, seed=0, dropout=0.2, lr=1e-3, weight_decay=1e-3,
                                                                           batch_size=64, epochs=3000, early_stopping=True, train_inds=train_sample,
                                                                           test_inds=test_sample, input_dropout=0.5, gene_set=canc_genes)

# Perform evaluation, generate plots and importance csv, save
plt.clf()
gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores = Pnet.evaluate_interpret_save(model,
                                                                                                                                   test_dataset,
                                                                                                                                   tumor_types.columns.values,
                                                                                                                                   "/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/pnet_gene/all_cancers/mc/plots/"+score_type+"/"+agg_func)
# Prepare importance dataframes for sankey diagram
layer_list = [gene_feature_importances, additional_feature_importances, gene_feature_importances] + layer_importance_scores
layer_list_names = ['gene_feature', 'gene_feature', 'gene'] + [f'layer_{i}' for i in range(5)]
layer_list_dict = dict(zip(layer_list_names, layer_list))

# Plot sankey
sk = sankey_diag.SankeyDiag(layer_list_dict, path='/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/resources/pathways_short_names.csv', runs = 1) 
fig = sk.get_sankey_diag("/shares/CIBIO-Storage/BCG/scratch/fgastaldello/data/pnet_fork/pnet_gene/all_cancers/mc/plots/"+score_type+"/"+agg_func+"/sankey.html")