'''
Generating synthetic code-mixed corpus

Steps:
    1. Finding and removing the one-to-many mapped source tokens
    2. Replacing the remaining source tokens with target tokens
    3. Skipping over the stopwords at both the source and target sides (optional)
    4. Saving the detokenized source, target, and both tokenized and detokenized generated code-mixed data

Script takes three files:
    1. Source
    2. Target
    3. Alignment

Source and target indices are starting from 0
Note: We observed that replacing stop words is generating less fluent sentences.
      That's why we are not replacing the stop words. But please feel to update
      the code by removing this step or adding other filtering techniques.

Indic language data is detokenized using Indic-NLP-Library

Other language data is detokenized using SacreMoses
'''

# imports
import argparse
from tqdm import tqdm
from sacremoses import MosesDetokenizer
from indicnlp.tokenize import indic_detokenize

# reading the data
def read_data(fpath):
    data = []
    with open(fpath, 'r', encoding='utf-8') as fin:
        for line in fin:
            data.append(line.strip())
            
    return data


# generating synthetic code-mixed data
def generate_cm_data(tokenized_src_data, tokenized_tgt_data, align_data, src_stopwords, tgt_stopwords):
    cm_data = []

    for src_line, tgt_line, alignment in tqdm(zip(tokenized_src_data, tokenized_tgt_data, align_data), ascii=True, total=len(tokenized_src_data)):
        align_dict = {}
        cm_seq = ''
        src_tokens = src_line.strip().split()
        tgt_tokens = tgt_line.strip().split()
        align_tokens = alignment.strip().split()

        # creating a dictionary using alignment data
        # keys: target token position
        # values: target token position
        for item in align_tokens:
            src_pos = int(item.split('-')[0])
            tgt_pos = int(item.split('-')[1])
            if src_pos in align_dict.keys():
                align_dict[src_pos].append(tgt_pos)
            else:
                align_dict[src_pos] = []
                align_dict[src_pos].append(tgt_pos)

        # removing both one-to-many and many-to-one mapped token alignments
        temp_dict_1 = {}
        one_to_one_dict = {}
        for item in align_dict.items():
            # item: (src_pos, [tgt_pos_1, tgt_pos_2])
            # if the length of value is > 1 then one-to-many mapped token hence ignoring
            if len(list(item[-1])) == 1:
                temp_dict_1[item[0]] = item[1]

        # temp_dict_1 contains one-to-one and many-to-one mapped tokens
        # removing the many-to-one mapped tokens
        only_values = [val for vals in list(temp_dict_1.values()) for val in vals]
        for item in temp_dict_1.items():
            if only_values.count(item[-1][0]) == 1:
                one_to_one_dict[item[0]] = item[-1][0]
        
        # replacing source tokens with corresponding target tokens from the alignments
        # optional: skipping over the stop words from both the source and target sides
        for src_tok in src_tokens:
            # checking if the source token is a stopword or not
            if src_tok in src_stopwords:
                cm_seq = cm_seq + src_tok + ' ' # no replacement
            else:
                src_tok_index = src_tokens.index(src_tok)
                # checking if the current source token is in one_to_onedict
                if src_tok_index in one_to_one_dict.keys():
                    tgt_tok_index = one_to_one_dict[src_tok_index]
                    if tgt_tokens[tgt_tok_index] in tgt_stopwords:
                        cm_seq = cm_seq + src_tok + ' ' # no replacement
                    else:
                        cm_seq = cm_seq + tgt_tokens[tgt_tok_index] + ' ' # replacing with target token
                else:
                    cm_seq = cm_seq + src_tok + ' ' # no replacement if the source token not in the one_to_one_dict
        
        # adding the generated code-mixed sentence to the return list
        cm_data.append(cm_seq)

    return cm_data


# detokenize data
def detokenize_data(data, lang_code):
    detokenized_data = []

    indic_lang_codes = ['as', 'bn', 'gu', 'hi', 'kn', 'mai', 'ml', 'mr', 'or', 'pa', 'ta', 'te']

    # detokenizationg for indic languages
    if lang_code in indic_lang_codes:
        for line in tqdm(data, ascii=True):
            line = line.strip()
            detokenized_data.append(indic_detokenize.trivial_detokenize(line, lang=lang_code))

    # detokenization for non-indic languages
    if lang_code not in indic_lang_codes:
        detokenizer = MosesDetokenizer(lang=lang_code)
        for line in tqdm(data, ascii=True):
            line = line.strip()
            detokenized_data.append(detokenizer.detokenize(line, return_str=True))

    return detokenized_data


# main function
def main():
    # argument parser
    arg_parser = argparse.ArgumentParser("Synthetic code-mixed data generation: Generation")
    
    arg_parser.add_argument('--tokenized_src_path', help='Path to the tokenized source corpus file', required=True, type=str, default=None)
    arg_parser.add_argument('--tokenized_tgt_path', help='Path to the tokenized target corpus file', required=True, type=str, default=None)
    arg_parser.add_argument('--align_out_path', help='Path to the output of word alignment script', required=True, type=str, default=None)
    arg_parser.add_argument('--src_stopwords_path', help='Path to the source language stopwords list', required=False, type=str, default=None)
    arg_parser.add_argument('--tgt_stopwords_path', help='Path to the source language stopwords list', required=False, type=str, default=None)
    arg_parser.add_argument('--src_lang', help='Source language', required=True, type=str, default=None)
    arg_parser.add_argument('--tgt_lang', help='Target language', required=True, type=str, default=None)
    arg_parser.add_argument('--output_path', help='Path to the output folder', required=True, type=str, default=None)

    args = arg_parser.parse_args()

    # reading source and target corpora
    print("Reading source and target corpora\n")
    tokenized_src_data = read_data(args.tokenized_src_path)
    tokenized_tgt_data = read_data(args.tokenized_tgt_path)
    align_data = read_data(args.align_out_path)

    if args.src_stopwords_path is not None:
        src_stopwords = read_data(args.src_stopwords_path)
    else:
        src_stopwords = []
    
    if args.tgt_stopwords_path is not None:
        tgt_stopwords = read_data(args.tgt_stopwords_path)
    else:
        tgt_stopwords = []

    print(f"Length of source and target corpora: {len(tokenized_src_data), len(tokenized_tgt_data)}\n")

    # generating the synthetic code-mixed data
    tokenized_cm_data = generate_cm_data(tokenized_src_data, tokenized_tgt_data, align_data, src_stopwords, tgt_stopwords)

    # detokenizing the data
    print("Detokenizing the data\n")
    detokenized_src_data = detokenize_data(tokenized_src_data, args.src_lang)
    detokenized_tgt_data = detokenize_data(tokenized_tgt_data, args.tgt_lang)
    detokenized_cm_data = detokenize_data(tokenized_cm_data, args.src_lang)

    # saving the data
    print("Saving the data\n")

    with open(args.output_path + 'src.detok', 'w', encoding='utf-8') as fout:
        for line in detokenized_src_data:
            fout.write(f"{line.strip()}\n")

    with open(args.output_path + 'tgt.detok', 'w', encoding='utf-8') as fout:
        for line in detokenized_tgt_data:
            fout.write(f"{line.strip()}\n")

    with open(args.output_path + 'src-tgt.cm.tok', 'w', encoding='utf-8') as fout:
        for line in tokenized_cm_data:
            fout.write(f"{line.strip()}\n")

    with open(args.output_path + 'src-tgt.cm.detok', 'w', encoding='utf-8') as fout:
        for line in detokenized_cm_data:
            fout.write(f"{line.strip()}\n")

    print(f"Data is saved to {args.output_path} folder with the following file names:")
    print("1. Source detokenized data: src.detok\n2. Target detokenized data: tgt.detok\n3. Generated synthetic tokenized code-mixed data: src-tgt.cm.tok\n4. Generated synthetic detokenized code-mixed data: src-tgt.cm.detok")

    print("Done..!")


# calling main function
if __name__ == '__main__':
    main()
