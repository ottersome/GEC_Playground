import sys
import os
import argparse
import pandas as pd
import pathlib 
import matplotlib.pyplot as plt
from collections import OrderedDict

hist = OrderedDict()

norm_dict = {
            ' .': '.',
            ' -': '-',
            '- ': '-',
            " '": "'",
            " n'": "n'",
            ' _': '_',
            ' ,': ',',
            ' :': ':',
            ' ;': ';',
            ' ?': '?',
            ' !': '!' }

def argsyparsy():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--data_directory',
            dest='data_directory',
            required=True
            )
    argparser.add_argument(
        '--dest_dir',
            dest='dest_dir',
            required=True
            )
    return argparser.parse_args()

def get_files_in_dir(dire:pathlib.Path):

    if not isinstance(dire, pathlib.Path):
        raise Exception('Not a directory')
    listo = []
    for f in dire.iterdir():
        # HARD CODING ABC After realizing my mistake
        #if 'train' in f.name and 'ABC' in f.name and f.name[0] != '.':
        if 'ABC' in f.name and f.name[0] != '.':
            print('Adding ',f.name, ' to our files')
            listo.append(f)
    return listo

def add_count(err_type):
    # err_type = err_type.replace(':','')
    if not err_type in hist.keys():
        hist[err_type] = 0
    else:
        hist[err_type] += 1

def form_false_true(errs):
    # Assuming Hist is already formed
    keys = hist.keys()
    woop = [0]*len(keys)
    for i,key in enumerate(keys):
        if key in errs: woop[i] = 1
    return woop



def collect_examples_in_file(file_path:pathlib.Path):
    # Given Initial Positio
    # Start Loop
    errs = []
    print('Reading file: ', file_path)
    with file_path.open() as f:
        # Get S
        while True:
            line = f.readline()
            if line == '': break # EOF 
            assert line[0] == 'S'
            # For some reason we get " ." in the sentences
            src_sentence = line[2:]
            for k,v in norm_dict.items(): 
                src_sentence = src_sentence.replace(k,v)
            src_sentence.strip()
            
            line = f.readline()
            err_types = []
            while line != '\n':
                assert line[0] == 'A', print('Line was {}'.format(line))
                # Break Line Into Consituent Parts
                err_type = line.split('|||')[1]
                err_types.append(err_type)
                add_count(err_type)
                #print('Error type : ',err_type)
                line = f.readline()
            err_types = '\t'.join(err_types)
            errs.append([src_sentence,err_types])
    return errs

def main(src_dir,save_path):
    # For Each File in Directory
    errs = []
    for file in get_files_in_dir(src_dir):
        file_errs = collect_examples_in_file(file)
        errs = errs+file_errs

    # We Now Set them up in the format we desire
    keys = list(hist.keys())
    print('Final Keys are :', keys)
    print('Total keys: ', len(keys))
    
    # print(errs)
    # We go through lists here because we need to parse them first to see all possible errors
    for err in errs:
        err_txt = err.pop()
        err+=form_false_true(err_txt)
        
    df = pd.DataFrame(data = errs, columns=['source']+keys)
    # Dump it all in a single File
    df.to_csv(save_path)

if __name__=='__main__':
    args = argsyparsy()
    if not pathlib.Path(args.data_directory).exists() or not pathlib.Path(args.data_directory).is_dir():
        print('Invalid Input Path. Not an Existing Directory')
        exit(-1)
    # if not pathlib.Path(args.dest_dir).():
        # print('Invalid Output Path. Not an Existing File')
        # exit(-1)
    # Else Create a Directory Object
    src = pathlib.Path(args.data_directory)
    dest = pathlib.Path(args.dest_dir)
    main(src,dest)

    # Show Distribution 
    plt.barh(range(len(hist)),hist.values())
    plt.yticks(range(len(hist)),labels=hist.keys())
    plt.title('Distribution of Errors')
    plt.legend()
    plt.show()
    print('Done.')
