import sys
import torch
import logging
from datetime import datetime

from torch.nn.utils.rnn import pad_sequence
from transformers import utils
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

tokenizer=AutoTokenizer.from_pretrained('bert-base-cased')

config = AutoConfig.from_pretrained('bigscience/bloomz-7b1')

labels = (
"R:OTHER",
"M:PUNCT",
"M:PREP",
"R:PREP",
"U:PREP",
"R:SPELL",
"UNK",
"R:VERB:SVA",
"R:NOUN:NUM",
"R:NOUN",
"U:DET",
"R:VERB",
"R:MORPH",
"noop",
"M:DET",
"M:OTHER",
"R:PUNCT",
"R:ORTH",
"M:CONJ",
"M:VERB:FORM",
"R:VERB:FORM",
"R:ADJ",
"R:WO",
"R:VERB:TENSE",
"M:PRON",
"U:PUNCT",
"R:PRON",
"R:ADV",
"U:OTHER",
"R:DET",
"M:NOUN",
"M:ADV",
"U:CONTR",
"U:NOUN",
"U:NOUN:POSS",
"U:VERB:FORM",
"U:ADJ",
"U:VERB",
"U:CONJ",
"U:PRON",
"M:VERB:TENSE",
"R:PART",
"M:PART",
"M:VERB",
"U:VERB:TENSE",
"U:PART",
"M:NOUN:POSS",
"R:ADJ:FORM",
"R:CONTR",
"R:CONJ",
"M:ADJ",
"U:ADV",
"R:NOUN:POSS",
"R:VERB:INFL",
"R:NOUN:INFL",
"M:CONTR")

id2label = {idx:label for idx,label in enumerate(labels)}
label2id = {label:idx for idx,label in enumerate(labels)}

device = torch.device('cuda')

if len(sys.argv) > 1:
    print('We assume your argument {} means checkpoint'.format(sys.argv[1]))
    chkpnt = sys.argv[1]
    model= AutoModelForSequenceClassification.from_pretrained(
            'bert-base-cased',
            problem_type='multi_label_classification',
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        ).to(device)

model.eval()
# print('Conversation Starts:')
while True:
    with torch.no_grad():
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        user_input = input("User: ").strip()
        #user_input = "Who was the president of Mexico in 2014?"
        print('You said:\n',user_input)
        bot_input_ids = tokenizer.encode(user_input , return_tensors='pt').to('cuda')

        # generated a response while limiting the total chat history to 1000 tokens, 

        output = model(bot_input_ids)
        print("output is ",output)

        siggs = torch.sigmoid(output.logits)[0].tolist()
        print(siggs)
        tuppies = []
        
        # Create Ordered List
        for i in range(len(labels)):
            tuppies.append((i,siggs[i]))

        tuppies.sort(key=lambda x : x[1],reverse=True)

        print('Most likely Errors:')
        for i in range(3):
            print("Error : {} with probability {:2.1f}".format(labels[tuppies[i][0]],tuppies[i][1]*100))

        sys.stdout.flush()


