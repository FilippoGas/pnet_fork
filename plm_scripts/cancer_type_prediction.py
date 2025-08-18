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

# Load scores and sample data
input_dir = "/home/filippo.gastaldello/data/pnet-fork/breast_subtypes_prediction/input/"
scores_hap1, scores_hap2, tumor_subtypes = util.load_hap_scores(input_dir)

genetic_data = {'scores_hap1':scores_hap1,
                'scores_hap2':scores_hap2}

# Load cancer genes list
canc_genes = list(pd.read_csv("/home/filippo.gastaldello/data/notebook_example_data/gene_lists/CancerGenesList.csv").values.reshape(-1))

# Split samples in train/test
samples = scores_hap1.index.tolist()
train_sample = sample(samples, round(len(samples)*0.7))
test_sample = list(set(samples)-set(train_sample))

