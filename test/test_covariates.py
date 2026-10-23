import os
import pickle
import numpy as np
import pandas as pd
import torch

from pnet import pnet_loader, Pnet
from pnet.Pnet import PNET_NN
from pnet import ReactomeNetwork


def load_test_data():
    test_rna = pd.read_csv('data/test_data/rna.csv').set_index('sample_id')
    test_cna = pd.read_csv('data/test_data/cna.csv').set_index('sample_id')
    test_add = pd.read_csv('data/test_data/add.csv').set_index('sample_id')
    test_y = pd.read_csv('data/test_data/y.csv').set_index('sample_id')
    with open('data/test_data/gene_sublist.txt', 'rb') as fp:
        gene_list = pickle.load(fp)
    genetic_data = {'rna': test_rna, 'cna': test_cna}
    return genetic_data, test_add, test_y, gene_list


def test_loader_with_covariates():
    genetic_data, test_add, test_y, gene_list = load_test_data()
    cov = pd.DataFrame({'cov1': np.linspace(0, 1, len(test_y)), 'cov2': np.random.RandomState(0).rand(len(test_y))},
                       index=test_y.index)
    train_dataset, test_dataset = pnet_loader.generate_train_test(genetic_data,
                                                                  test_y,
                                                                  additional_data=test_add,
                                                                  covariates=cov,
                                                                  test_split=0.2,
                                                                  gene_set=gene_list)
    assert hasattr(train_dataset, 'covariates')
    assert train_dataset.covariates.shape[1] == 2
    batch = train_dataset[0]
    assert len(batch) == 4


def test_loader_without_covariates_positional_ok():
    genetic_data, test_add, test_y, gene_list = load_test_data()
    train_dataset, test_dataset = pnet_loader.generate_train_test(genetic_data,
                                                                  test_y,
                                                                  None,
                                                                  test_add,
                                                                  0.2,
                                                                  None,
                                                                  None,
                                                                  None,
                                                                  2)
    assert hasattr(train_dataset, 'covariates')
    assert train_dataset.covariates.shape[1] == 0
    batch = train_dataset[0]
    assert len(batch) == 4


def test_model_forward_with_and_without_covariates():
    genetic_data, test_add, test_y, gene_list = load_test_data()
    cov = pd.DataFrame({'cov1': np.linspace(0, 1, len(test_y))}, index=test_y.index)
    train_dataset, test_dataset = pnet_loader.generate_train_test(genetic_data,
                                                                  test_y,
                                                                  additional_data=test_add,
                                                                  covariates=cov,
                                                                  test_split=0.2,
                                                                  gene_set=gene_list)
    reactome = ReactomeNetwork.ReactomeNetwork(train_dataset.get_genes())
    model = PNET_NN(reactome_network=reactome, task='REG', nbr_gene_inputs=len(genetic_data),
                    additional_dims=train_dataset.additional_data.shape[1], output_dim=test_y.shape[1],
                    n_covariates=cov.shape[1])
    out = model(train_dataset.x[:4], train_dataset.additional[:4], train_dataset.covariates[:4])
    assert out[0].shape[0] == 4
    # model built without covariates should ignore covariates
    model_no = PNET_NN(reactome_network=ReactomeNetwork.ReactomeNetwork(train_dataset.get_genes()), task='REG',
                       nbr_gene_inputs=len(genetic_data), additional_dims=train_dataset.additional_data.shape[1],
                       output_dim=test_y.shape[1], n_covariates=0)
    out2 = model_no(train_dataset.x[:4], train_dataset.additional[:4], train_dataset.covariates[:4])
    assert out2[0].shape[0] == 4


def main():
    test_loader_with_covariates()
    test_loader_without_covariates_positional_ok()
    test_model_forward_with_and_without_covariates()
    print('All covariate tests passed!')


if __name__ == '__main__':
    main()
