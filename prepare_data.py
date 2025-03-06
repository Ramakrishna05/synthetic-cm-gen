'''
Preparing data for the alignment

This script does the following tasks:
    1. Read source and target corpus files
    2. Remove empty lines (if any)
    3. Tokenize the data
    4. Save the data (tokenized source, tokenized target, and the input to the alignment script)

Indic language data is tokenized using Indic-NLP-Library
    Ref: https://nbviewer.org/url/anoopkunchukuttan.github.io/indic_nlp_library/doc/indic_nlp_examples.ipynb

Other language data is tokenized using SacreMoses
    Ref: https://github.com/hplt-project/sacremoses

'''

# imports
import os
import argparse
from tqdm import tqdm
from sacremoses import MosesTokenizer
from indicnlp.tokenize import indic_tokenize

# reading the data
def read_data(fpath):
    data = []
    with open(fpath, 'r', encoding='utf-8') as fin:
        for line in fin:
            data.append(line.strip())
            
    return data


# remove empty lines
def remove_empty_lines(src_data, tgt_data):
    assert len(src_data) == len(tgt_data), "Length of source and target files are not the same. Please check the number of sentences in the input files!"
    
    clean_src_data = []
    clean_tgt_data = []

    for src_line, tgt_line in tqdm(zip(src_data, tgt_data), ascii=True, total=len(clean_src_data)):
        if len(src_line.split()) != 0 and len(tgt_line.split()) != 0:
            clean_src_data.append(src_line)
            clean_tgt_data.append(tgt_line)

    print(f"No. of empty lines in both source and target data are: {len(src_data) - len(clean_src_data)}. They are removed.\n")

    return clean_src_data, clean_tgt_data


# tokenize data
def tokenize_data(data, lang_code):
    tokenized_data = []

    indic_lang_codes = ['as', 'bn', 'gu', 'hi', 'kn', 'mai', 'ml', 'mr', 'or', 'pa', 'ta', 'te']

    # tokenization for indic languages
    if lang_code in indic_lang_codes:
        for line in tqdm(data, ascii=True):
            line = line.strip()
            tokenized_data.append(" ".join(tok for tok in indic_tokenize.trivial_tokenize(line, lang=lang_code)))

    # tokenization for non-indic languages
    if lang_code not in indic_lang_codes:
        tokenizer = MosesTokenizer(lang=lang_code)
        for line in tqdm(data, ascii=True):
            line = line.strip()
            tokenized_data.append(tokenizer.tokenize(line, return_str=True))
    
    return tokenized_data


# main function
def main():
    # argument parser
    arg_parser = argparse.ArgumentParser("Synthetic code-mixed data generation: Preprocessing")
    
    arg_parser.add_argument('--src_path', help='Path to the source corpus file', required=True, type=str, default=None)
    arg_parser.add_argument('--tgt_path', help='Path to the target corpus file', required=True, type=str, default=None)
    arg_parser.add_argument('--src_lang', help='Source language', required=True, type=str, default=None)
    arg_parser.add_argument('--tgt_lang', help='Target language', required=True, type=str, default=None)
    arg_parser.add_argument('--output_path', help='Path to the output folder', required=True, type=str, default=None)

    args = arg_parser.parse_args()

    # reading source and target corpora
    print("Reading source and target corpora\n")
    raw_src_data = read_data(args.src_path)
    raw_tgt_data = read_data(args.tgt_path)
    print(f"Length of source and target corpora: {len(raw_src_data), len(raw_tgt_data)}\n")

    # remove empty lines
    print("Removing empty lines\n")
    clean_src_data, clean_tgt_data = remove_empty_lines(raw_src_data, raw_tgt_data)

    # tokenize data
    print("Tokenizing the data\n")
    tokenized_src_data = tokenize_data(clean_src_data, args.src_lang)
    tokenized_tgt_data = tokenize_data(clean_tgt_data, args.tgt_lang)

    # preparing data for the alignment tool
    # format: src_sent ||| tgt_set (there is a space before and after ||| (three pipe symbols))
    print("Preparing the data for the alignment tool\n")
    alignment_tool_input = []
    for src_line, tgt_line in tqdm(zip(tokenized_src_data, tokenized_tgt_data), ascii=True, total=len(tokenized_src_data)):
        to_write = src_line + ' ||| ' + tgt_line
        alignment_tool_input.append(to_write)

    # saving the data
    print("Saving the data\n")

    # creating the Output directory
    os.makedirs(args.output_path) #, exist_ok=True)

    with open(args.output_path + 'src.tok', 'w', encoding='utf-8') as fout:
        for line in tokenized_src_data:
            fout.write(f"{line.strip()}\n")

    with open(args.output_path + 'tgt.tok', 'w', encoding='utf-8') as fout:
        for line in tokenized_tgt_data:
            fout.write(f"{line.strip()}\n")

    with open(args.output_path + 'src-tgt.align.in', 'w', encoding='utf-8') as fout:
        for line in alignment_tool_input:
            fout.write(f"{line.strip()}\n")

    print(f"Data is saved to {args.output_path} folder with the following file names:")
    print("1. Source tokenized data: src.tok\n2. Target tokenized data: tgt.tok\n3. Input to the alignment tool: src-tgt.align.in")


# calling main function
if __name__ == '__main__':
    main()
