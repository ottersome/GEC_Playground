import torch
import pickle
import pandas as pd
import pathlib
import numpy as np
import torch.nn.functional as F
from transformers import BertTokenizer


def tokenize_dataset(df:pd.DataFrame, tokenizer: BertTokenizer):
    examples = []
    # We assume a particular layout for our data
    #print(df)
    # Go Through all DataFrame
    print('Tokenizing Dataset...')
    for sentence,label in zip(df['sentence'],df['label']):
        # This give us dictionary
        ### X ###
        ## Tokenizer returns a dictionary
        inpt = tokenizer('[CLS]'+sentence)
        ### Y ###
        #tlabel = torch.Tensor(label)
        tlabel = F.one_hot(torch.Tensor([label]).to(torch.int64), num_classes=2).to(torch.float32)
        tlabel = tlabel.squeeze()
        # To dictionary we add labels key
        inpt['labels'] = tlabel
        assert label != None 
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
            print('We found an existing binary file at:')
            print('\t',binlocal_path)
            self.samples = pickle.load(local_file)
        elif isinstance(df,pd.DataFrame):
            # We have to create, we override what was previously wrriten
            local_file = binlocal_path.open('wb')
            self.samples = tokenize_dataset(df,tokenizer)
            pickle.dump(self.samples,local_file,protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise Exception('Misuse of dataset. No DataFrame or Location provided.')

        # Get some basic info
        self.num_samples = len(self.samples)
        
    def get_labels(self):
        return ["Unacceptable","Acceptable"]

    def __getitem__(self,idx):
        # Return data as a torch tensor.
        item = {k:torch.tensor(v) for k,v in self.samples[idx].items()}
        return item

    def __len__(self):
        return self.num_samples

