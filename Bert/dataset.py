import torch
import pickle
import pandas as pd
import pathlib
import numpy as np
from transformers import BertTokenizer


def tokenize_dataset(df:pd.DataFrame, tokenizer: BertTokenizer):
    examples = []
    # We assume a particular layout for our data
    #print(df)
    for idx,row in df.iterrows():
        # This give us dictionary
        inpt = tokenizer(row['source'])
        lbls = np.array(row.values[2:],dtype=np.float32)
        inpt['labels'] =  torch.tensor(lbls,dtype=torch.float64)
        # TODO: this wastes space with having dictionaries wre they are not needed. 
        examples.append(inpt)

    return examples

class LabelDataset(torch.utils.data.Dataset):

    # We assume data given is already a split of train or val
    def __init__(self, tokenizer:BertTokenizer,df : pd.DataFrame=None,bin_path: str = None):
        
        # If already exists 
        binlocal_path = pathlib.Path(bin_path)
        if binlocal_path.exists() and binlocal_path.is_file():
            local_file = binlocal_path.open('rb')
            self.text_labels,self.samples = pickle.load(local_file)
        elif isinstance(df,pd.DataFrame):
            # We have to create, we override what was previously wrriten
            local_file = binlocal_path.open('wb')
            self.samples = tokenize_dataset(df,tokenizer)
            self.text_labels = list(df.columns[2:])
            print('Labels we use : ',self.text_labels)
            pickle.dump((self.text_labels,self.samples),local_file,protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise Exception('Misuse of dataset. No DataFrame or Location provided.')

        # Get some basic info
        self.num_samples = len(self.samples)
        self.num_labels = len(self.samples[1])
        print('Initialiaze Dataset with {} number of samples and with {} num of labels'
              .format(self.num_samples, self.num_labels))
        
    def get_labels(self):
        return self.text_labels

    def __getitem__(self,idx):
        # Return data as a torch tensor.
        item = {k:torch.tensor(v) for k,v in self.samples[idx].items()}
        return item

    def __len__(self):
        return self.num_samples

