# Script version of the SKCM_purity notebook

import sys
import os
sys.path.append(os.path.dirname(__file__)+"/../src/")

from pnet import pnet_loader, Pnet
from util import util, sankey_diag
 
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import random
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')


datapath = '/home/filippo.gastaldello/data/notebook_example_data/'
fig_output = '/home/filippo.gastaldello/data/pnet-fork/SKCM_purity/plots/'

# LOAD INPUT

# The input in this case is a count matrix from rna expression and copy number amplifications on 
# a gene level of the SKCM dataset from TCGA. We use a custom function for some formatting, but 
# pandas would work fine too. 

# The data is then put into a dictionary shape required for pnet, this is especially useful when 
# using multiple data modalities. 
rna, cna, tumor_type = util.load_tcga_dataset(datapath+'skcm_tcga_pan_can_atlas_2018')
genetic_data = {'rna' : rna, 'cna' : cna}

# LOAD TARGET

# In this example notebook we use tumor purity as a target, which serves as a good example for a 
# regression task with gene level data. We use the tumor purity called by  Taylor et al. 2018, 
# available here https://github.com/cBioPortal/datahub/tree/master/public

# Import purity dataset
purity_TCGA = pd.read_csv(datapath+'TCGA_mastercalls.abs_tables_JSedit.fixed.txt', delimiter = '\t').set_index('array')
# Only keep purity values for which rna information are available
purity = pd.DataFrame(purity_TCGA.join(rna, how = 'inner')['purity'])

# RUN PNET

# We limit the number of genes used for predictions to a subset of informative cancer genes
canc_genes = list(pd.read_csv(datapath+'gene_lists/CancerGenesList.csv')['hgnc'])

# The Pnet model can then be trained by simply calling Pnet.run() with the input dictionairy 
# and the target dataframe. Additionally, here we use the cancer gene list. Many more options are
# available to specific training.
model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(genetic_data = genetic_data,
                                                                         target = purity,
                                                                         gene_set = canc_genes)

# The pnet model automatically generates a train and test dataset from the input. Here we can use 
# these for illustration of the results on the test dataset:
x_train = train_dataset.x
additional_train = train_dataset.additional
y_train = train_dataset.y
x_test = test_dataset.x
addition_test = test_dataset.additional
y_test = test_dataset.y


model.to('cpu')

y_pred = model.predict(test_dataset.x, test_dataset.additional).detach()

df = pd.DataFrame(index = test_dataset.input_df.index)
df['y_test'] = test_dataset.y
df['y_pred'] = y_pred

sns.regplot(data = df, x = 'y_test', y = 'y_pred', color='#41B6E6')
correlation_coefficient = round(df['y_test'].corr(df['y_pred']), 2)
plt.text(0.95, 0.05, f'Correlation: {correlation_coefficient}', ha = 'right', va = 'center', transform = plt.gca().transAxes)
plt.plot(y_test, y_pred, color = '#FFA300', linestyle = '--', label = 'Diagonal Line')
sns.despine()
plt.savefig(fname = fig_output+"test_pred_corr.pdf", format='pdf', dpi = 500)