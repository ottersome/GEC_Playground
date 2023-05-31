import argparse
import pandas as pd
import torch
import wandb
from datetime import date,datetime
from transformers import (AutoModelForSequenceClassification,
    TrainingArguments,
    TrainerState,
    Trainer,
    AutoTokenizer)
from typing import List,Dict
from utils import * 
from torch.nn.utils.rnn import pad_sequence



def collator(examples: List[Dict[str,torch.Tensor]]):
    batch = {}
    first = examples[0]
    # Collect Differently
    for k in first.keys():
        batch[k] = [e[k] for e in examples]
    # Now Just Pad it as we know how to
    padder = 0.0
    if tokenizer._pad_token != None:
        paddier= tokenizer._pad_token
    for k,v in batch.items():
        # TODO: Check if we are padding with the right token id or if we have to do so at all. 
        batch[k] = pad_sequence(v, batch_first=True, padding_value=tokenizer.pad_token_id)

    return batch


if __name__ == '__main__':

    # Load Arguments
    args = parse_args()
    
    # Load Dataset
    print('Loading Data')
    tokenizer=AutoTokenizer.from_pretrained('bert-base-cased')
    ds_train,ds_test = load_data(args,tokenizer)
    labels = ds_train.get_labels()
    print('Working with labels:\n', labels)

    # Setup Stuff We need For Later
    id2label = {idx:label for idx,label in enumerate(labels)}
    label2id = {label:idx for idx,label in enumerate(labels)}


    # Load Model
    model= AutoModelForSequenceClassification.from_pretrained(
            'bert-base-cased',
            problem_type='multi_label_classification',
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id
        )
    #
    # Initialize Reporter
    wandb.init(project ='multilabeler', config=model.config)
    ##################################
    # Train Loop
    ##################################
    run_name = datetime.strftime(datetime.now(),'%Y_%m_%d-%H-%M_%S_train_run')
    training_args = TrainingArguments(
            "chkpnts",
            evaluation_strategy="steps",
            learning_rate=1e-5,
            gradient_accumulation_steps=2,
            auto_find_batch_size=True,
            #per_device_train_batch_size=4,
            num_train_epochs=args.num_epochs,
            warmup_steps=100,
            logging_dir="runs",
            logging_steps=5,
            save_strategy="steps",
            report_to='wandb',
            eval_steps=12,
            save_steps=3000,
            run_name=run_name,
            save_total_limit=2
            )

    trainer = Trainer(
        model=model,
        data_collator=collator,
        args=training_args,
        # optimizers=(optimizer,scheduler),
        train_dataset=ds_train,
        #compute_metrics=my_compute_metrics,
        eval_dataset=ds_test
        )

    trainer.train()

    # Datasets
    #   Create Split
    #
    #   Train Dataset
    #   Val Dataset

