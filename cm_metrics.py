'''
Quantifying the amount of code-mixing

The metrics used are:
    1. Code-mixing Index (CMI) (https://aclanthology.org/L16-1292/)
    2. Switch-point Fraction (SPF) (https://aclanthology.org/P18-1143/)
    3. Entropy (Shannon's entropy)

These metrics are computed for each sentence as:
    1. CMI: (1 - (Dominent language word count) / (Total word count)) * 100
    2. SPF: (Number of switch points) / (Total word count - 1)
    3. Entropy: -(p log(p) + q log(q)) (p, q are percentage of  number of words from language 1 and language 2 respectively)
'''

# imports
import argparse
import math
from tqdm import tqdm

# reading the data
def read_data(fpath):
    data = []
    with open(fpath, 'r', encoding='utf-8') as fin:
        for line in fin:
            data.append(line.strip())

    return data


# metrics
def metrics(src_data, tgt_data, cm_data):
    cmi = 0
    entropy = 0
    spf = 0

    for src_line, tgt_line, cm_line in tqdm(zip(src_data, tgt_data, cm_data), ascii=True, total=len(src_data)):
        src_tokens = src_line.split()
        tgt_tokens = tgt_line.split()
        cm_tokens = cm_line.split()

        if len(cm_tokens) == 1:
            continue

        # common token count between source, target and code-mixed sentences
        common_tokens = list(set(src_tokens).intersection(cm_tokens, tgt_tokens))
        common_count = len(common_tokens)

        # common token count between source and code-mixed sentences
        src_cm_count = len(list(set(src_tokens).intersection(cm_tokens))) - common_count
        src_cm_percent = src_cm_count / len(cm_tokens)

        # common token count between target and code-mixed sentences
        tgt_cm_count = len(list(set(tgt_tokens).intersection(cm_tokens))) - common_count
        tgt_cm_percent = tgt_cm_count / len(cm_tokens)

        # if all tokes are identical across source and target then considering them as source tokens
        if src_cm_count == 0 and tgt_cm_count == 0:
            src_cm_count = common_count
            src_cm_percent = src_cm_count / len(cm_tokens)

        # finding the switching points
        switch_points = 0
        prev_lang = 'src'
        for tok in cm_tokens:
            # for the first token
            if cm_tokens.index(tok) == 0:
                if tok not in common_tokens and tok in src_tokens:
                    prev_lang = 'src'
                elif tok not in common_tokens and tok in tgt_tokens:
                    prev_lang = 'tgt'
                elif tok in common_tokens:
                    prev_lang = prev_lang
            else:
                if tok in common_tokens:
                    prev_lang = prev_lang
                elif tok in src_tokens and prev_lang == 'tgt':
                    switch_points += 1
                elif tok in src_tokens and prev_lang == 'src':
                    prev_lang = 'src'
                elif tok in tgt_tokens and prev_lang == 'src':
                    switch_points += 1
                elif tok in tgt_tokens and prev_lang == 'tgt':
                    prev_lang = 'tgt'

        # finding the dominant count
        if src_cm_count > tgt_cm_count:
            dominant_count = src_cm_count
        else:
            dominant_count = tgt_cm_count

        # cmi
        sent_cmi = (1 - (dominant_count/len(cm_tokens))) * 100
        cmi += sent_cmi

        # entropy
        if tgt_cm_percent == 0:
            sent_entropy = -(src_cm_percent * math.log2(src_cm_percent))
        elif src_cm_percent == 0:
            sent_entropy = -(tgt_cm_percent * math.log2(tgt_cm_percent))
        else:
            sent_entropy = -((src_cm_percent * math.log2(src_cm_percent)) + (tgt_cm_percent * math.log2(tgt_cm_percent)))

        entropy += sent_entropy

        # spf
        sent_spf = (switch_points / (len(cm_tokens) - 1))
        spf += sent_spf

    return (cmi / len(cm_data)), (entropy / len(cm_data)), (spf / len(cm_data))


# main function
def main():
    # argument parser
    arg_parser = argparse.ArgumentParser("Synthetic code-mixed data generation: Metrics")
    arg_parser.add_argument('--tokenized_src_path', help='Path to the tokenized source corpus file', required=True, type=str, default=None)
    arg_parser.add_argument('--tokenized_tgt_path', help='Path to the tokenized target corpus file', required=True, type=str, default=None)
    arg_parser.add_argument('--tokenized_cm_path', help='Path to the tokenized code-mixed corpus file', required=True, type=str, default=None)

    args = arg_parser.parse_args()

    # reading source and target corpora
    print("Reading source, target, and code-mixed corpora\n")
    tokenized_src_data = read_data(args.tokenized_src_path)
    tokenized_tgt_data = read_data(args.tokenized_tgt_path)
    tokenized_cm_data = read_data(args.tokenized_cm_path)

    # calculating the metrics
    print("Calculating the metrics\n")
    cmi, entropy, spf =  metrics(tokenized_src_data, tokenized_tgt_data, tokenized_cm_data)

    print(f"CMI: {cmi:,.2f}\nEntropy: {entropy:,.2f}\nSPF: {spf:,.2f}\n")


# calling the main function
if __name__ == '__main__':
    main()
