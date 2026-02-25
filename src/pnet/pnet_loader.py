import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import numpy as np
import os


# DataLoader object for pytorch. Constructing single loader for all data input modalities.

class PnetDataset(Dataset):
    def __init__(self, genetic_data, target, indicies, additional_data=None, gene_set=None, covariates_data=None):
        """
        DataLoader initialization, builds object for pytorch data loading. Handles concatenantion of different
        genetic modalities, connection to target and batching.
        :param genetic_data: Dict(str: pd.DataFrame); requires a dict containing a pd.DataFrame for each data modality
         and the str identifier. Paired samples should have matching indicies across Dataframes.
        :param target: pd.DataFrame or pd.Series; requires a single pandas Dataframe or Series with target variable
         paired per sample index. Target can be binary or continuous.
        :param indicies: list(str); List of sample names to be used for joint Dataset
        :param additional_data: pd.DataFrame; Dataframe with additional information per sample. Sample IDs should match
         genetic data. Per default None is provided
        :param gene_set: list(str); List of genes to be considered, by default all overlapping genes are considered
        """
        assert isinstance(genetic_data, dict), f"input data expected to be a dict, got {type(genetic_data)}"
        for inp in genetic_data:
            assert isinstance(inp, str), f"input data keys expected to be str, got {type(inp)}"
            assert isinstance(genetic_data[inp], pd.DataFrame), f"input data values expected to be a dict, got" \
                                                                f" {type(genetic_data[inp])}"
        self.inds = indicies
        
        temp_input_df, self.genes = self._unpack_and_align_genes(genetic_data, gene_set)
        
        self.feature_names = temp_input_df.columns
        self.sample_names = temp_input_df.index

        self.x = torch.tensor(temp_input_df.values, dtype=torch.float)
        del temp_input_df

        target_subset = target.loc[self.inds]
        self.y = torch.tensor(target_subset.values, dtype=torch.float)
        
        if additional_data is not None:
            add_subset = additional_data.loc[self.inds]
            self.additional = torch.tensor(add_subset.values, dtype=torch.float)
            self.additional_data_columns = add_subset.columns
            self.additional_data_index = add_subset.index
        else:
            self.additional = torch.tensor(np.zeros((len(self.inds), 0)), dtype=torch.float)
            self.additional_data_columns = []
            self.additional_data_index = self.inds

        if covariates_data is not None:
            cov_subset = covariates_data.loc[self.inds].values
            self.covariates_tensor = torch.tensor(cov_subset).float()
        else:
            self.covariates_tensor = None
        
    def _unpack_and_align_genes(self, genetic_data, gene_set):
        local_genetic_data = {k: v.loc[:, ~v.columns.duplicated()].copy() for k, v in genetic_data.items()}
        
        gene_sets = [set(df.columns) for df in local_genetic_data.values()]
        if gene_set:
            gene_sets.append(set(gene_set))
        genes = list(set.intersection(*gene_sets))
        print(f'Found {len(genes)} overlapping genes')

        input_df = pd.DataFrame(index=self.inds)
        for inp, df in local_genetic_data.items():
            temp_df = df[genes]
            temp_df = temp_df.add_suffix(f'_{inp}')
            input_df = input_df.join(temp_df, how='inner')
            
        print(f'generated input DataFrame of size {input_df.shape}')
        return input_df.loc[self.inds], genes
    
    @property
    def additional_data(self):
        class Wrapper:
            def __init__(self, tensor, cols, idx):
                self.shape = tensor.shape
                self.columns = cols
                self.index = idx
                
        return Wrapper(self.additional, self.additional_data_columns, self.additional_data_index)
    
    @property
    def input_df(self):
        """
        Mimics the input DataFrame to satisfy Pnet.py's interpretation methods.
        Returns an object with .index and .columns.
        """
        class MetaWrapper:
            def __init__(self, idx, cols):
                self.index = idx
                self.columns = cols
        return MetaWrapper(self.sample_names, self.feature_names)

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        additional = self.additional[index]

        if self.covariates_tensor is not None:
            return x, additional, self.covariates_tensor[index], y
        else:
            return x, additional, y

    def get_genes(self):
        return self.genes

    def unpack_input(self):
        """
        Unpacks data modalities into one joint pd.DataFrame. Suffixing gene names by their modality name.
        :return: pd.DataFrame; containing n*m columns, where n is the number of modalities and m the number of genes
        considered.
        """
        input_df = pd.DataFrame(index=self.inds)
        for inp in self.genetic_data:
            temp_df = self.genetic_data[inp][self.genes]
            temp_df.columns = temp_df.columns + '_' + inp
            input_df = input_df.join(temp_df, how='inner', rsuffix='_' + inp)
        print('generated input DataFrame of size {}'.format(input_df.shape))
        return input_df.loc[self.inds]
    
    def save_indicies(self, path):
        d = '/'.join(path.split('/')[:-1])
        if not os.path.exists(d):
            os.makedirs(d)
        df = pd.DataFrame(data={"indicies": self.inds})
        df.to_csv(path, sep=',',index=False)


def get_indicies(genetic_data, target, additional_data=None, covariates_data=None):
    """
    Generates a list of indicies which are present in all data modalities. Drops duplicated indicies.
    :param genetic_data: Dict(str: pd.DataFrame); requires a dict containing a pd.DataFrame for each data modality
         and the str identifier. Paired samples should have matching indicies across Dataframes.
    :param target: pd.DataFrame or pd.Series; requires a single pandas Dataframe or Series with target variable
        paired per sample index. Target can be binary or continuous.
    :param additional_data: pd.DataFrame; Dataframe with additional information per sample. Sample IDs should match
     genetic data.
    :return: List(str); List of sample names found in all data modalities
    """
    for gd in genetic_data:
        genetic_data[gd].dropna(inplace=True)
    target.dropna(inplace=True)
    ind_sets = [set(genetic_data[inp].index.drop_duplicates(keep=False)) for inp in genetic_data]
    ind_sets.append(target.index.drop_duplicates(keep=False))
    if additional_data is not None:
        ind_sets.append(additional_data.index.drop_duplicates(keep=False))
    if covariates_data is not None:
        ind_sets.append(covariates_data.index.drop_duplicates(keep=False))
    inds = list(set.intersection(*ind_sets))
    print('Found {} overlapping indicies'.format(len(inds)))
    return inds


def generate_train_test(genetic_data, target, gene_set=None, additional_data=None, covariates_data=None, test_split=0.3, seed=None,
                        train_inds=None, test_inds=None, collinear_features=0, shuffle_labels=False):
    """
    Takes all data modalities to be used and generates a train and test DataSet with a given split.
    :param genetic_data: Dict(str: pd.DataFrame); requires a dict containing a pd.DataFrame for each data modality
         and the str identifier. Paired samples should have matching indicies across Dataframes.
    :param target: pd.DataFrame or pd.Series; requires a single pandas Dataframe or Series with target variable
        paired per sample index. Target can be binary or continuous.
    :param gene_set: List(str); List of genes to be considered, default is None and considers all genes found in every
        data modality.
    :param additional_data: pd.DataFrame; Dataframe with additional information per sample. Sample IDs should match
    :param test_split: float; Fraction of samples to be used for testing.
    :param seed: int; Random seed to be used for train/test splits.
    :return:
    """
    print('Given {} Input modalities'.format(len(genetic_data)))
    inds = get_indicies(genetic_data, target, additional_data, covariates_data)
    random.seed(seed)
    random.shuffle(inds)
    if train_inds and test_inds:
        train_inds = list(set(inds).intersection(train_inds))
        test_inds = list(set(inds).intersection(test_inds))
    elif train_inds:
        train_inds = list(set(inds).intersection(train_inds))
        test_inds = [i for i in inds if i not in train_inds]
    elif test_inds:
        test_inds = list(set(inds).intersection(test_inds))
        train_inds = [i for i in inds if i not in test_inds]
    else:
        test_inds = inds[int((len(inds) + 1) * (1 - test_split)):]
        train_inds = inds[:int((len(inds) + 1) * (1 - test_split))]
    print('Initializing Train Dataset')
    train_dataset = PnetDataset(genetic_data, target, train_inds, additional_data=additional_data, gene_set=gene_set, covariates_data=covariates_data)
    print('Initializing Test Dataset')
    test_dataset = PnetDataset(genetic_data, target, test_inds, additional_data=additional_data, gene_set=gene_set, covariates_data=covariates_data)
    
    # Positive control: Replace a gene's values with values collinear to the target
    train_dataset, test_dataset = add_collinear(train_dataset, test_dataset, collinear_features)
    # Positive control: Shuffle labels for prediction
    if shuffle_labels:
        train_dataset = shuffle_data_labels(train_dataset)
        test_dataset = shuffle_data_labels(test_dataset)
    return train_dataset, test_dataset


def to_dataloader(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory = True, prefetch_factor=2, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory = True, prefetch_factor=2, persistent_workers=True)
    return train_loader, val_loader

def add_collinear(train_dataset, test_dataset, collinear_features):
    if isinstance(collinear_features, list):
        for f in features:
            replace_collinear(train_dataset, test_dataset, f)
    else:
        for n in range(collinear_features):
            r = random.randint(0, len(train_dataset.input_df.columns))
            altered_input_col = train_dataset.input_df.columns[r]
            train_dataset, test_dataset = replace_collinear(train_dataset, test_dataset, altered_input_col)
    return train_dataset, test_dataset

def shuffle_data_labels(dataset):
    print('shuffling {} labels'.format(dataset.target.shape[0]))
    target_copy = dataset.target.copy()
    target_copy[target_copy.columns[0]] = dataset.target.sample(frac=1).reset_index(drop=True).values
    dataset.target = target_copy
    return dataset

def replace_collinear(train_dataset, test_dataset, altered_input_col):
    train_dataset.altered_inputs.append(altered_input_col)
    test_dataset.altered_inputs.append(altered_input_col)
    print('Replace input of: {} with collinear feature.'.format(altered_input_col))
    train_dataset.input_df[altered_input_col] = train_dataset.target
    test_dataset.input_df[altered_input_col] = test_dataset.target
    return train_dataset, test_dataset
