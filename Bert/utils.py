import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_path',
                        dest='ds_path',
                        default='../wi+locness/classified.csv',
                        type=str
                        )
    parser.add_argument('--num_epochs',
                        dest='num_epochs',
                        default=10,
                        type=int
                        )
    parser.add_argument('--binary_ds_path',
                        dest='bin_ds_path',
                        default='./.my_cache/dataset.bin',
                        type=str
                        )

    return parser.parse_args()


def load_data(args,tokenizer):

    # Look For Local Binary First
    ds_lbt = pathlib.Path(args.bin_ds_path+'_train.bin')# Local Binary Train
    ds_lbv = pathlib.Path(args.bin_ds_path+'_test.bin')# Local Binary Test
    df_path = pathlib.Path(args.ds_path)
    # Assuming Those Work:
    if ds_lbt.exists() and ds_lbv.exists():
        print('Found Binary. Loading from binary...')
        # We just Load the datasets
        train_ds = LabelDataset(tokenizer,bin_path=str(ds_lbt))
        test_ds = LabelDataset(tokenizer,bin_path=str(ds_lbv))
    else:
        # We load the DataFrame
        print('No binary found, constructing from zero...')
        assert df_path.exists()
        df = pd.read_csv(args.ds_path)
        df_train, df_test = train_test_split(df)
        # print(df_train.head())
        train_ds = LabelDataset(tokenizer,df=df_train,bin_path=str(ds_lbt))
        test_ds = LabelDataset(tokenizer,df=df_test,bin_path=str(ds_lbv))
    
    return train_ds,test_ds
