import sys
import os
sys.path.append(os.path.dirname(__file__)+"/../src/")

from pnet import pnet_loader, Pnet
from util import util, sankey_diag 

import torch
import seaborn as sns
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from random import sample
import scipy

# Transcript-gene map
transcript_map_dir = "/home/filippo.gastaldello/resources/pnet/transcript_gene_map.json"
# Load scores and sample data
input_dir = "/home/filippo.gastaldello/data/pnet-fork/breast_subtype_prediction/input/"
scores_hap1, scores_hap2, tumor_subtypes = util.load_hap_scores(input_dir)

genetic_data = {'scores_hap1':scores_hap1,
                'scores_hap2':scores_hap2}

# Load cancer genes list
canc_genes = list(pd.read_csv("/home/filippo.gastaldello/data/notebook_example_data/gene_lists/CancerGenesList.csv")['hgnc'])

# Split samples in train/test
samples = scores_hap1.index.tolist()
train_sample = sample(samples, round(len(samples)*0.7))
test_sample = list(set(samples)-set(train_sample))

model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(genetic_data, pd.get_dummies(tumor_subtypes),with_transcript=True , transcript_gene_map=transcript_map_dir, seed=0, dropout=0.2, lr=1e-3, weight_decay=1e-3,
                                                                           batch_size=16, epochs=300, early_stopping=True, train_inds=train_sample,
                                                                           test_inds=test_sample, input_dropout=0.5, gene_set=canc_genes)

plt.clf()
Pnet.evaluate_interpret_save(model, test_dataset, "/home/filippo.gastaldello/data/pnet-fork/breast_subtype_prediction/plot/breast_subtype_prediction")