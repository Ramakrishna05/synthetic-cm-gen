# Bash script for generating synthetic code-mixed data
set -e

# Global variables
SRC_PATH=~/synthetic-cm-gen/data/sample.hi # path to the source corpus file
TGT_PATH=~/synthetic-cm-gen/data/sample.en # path to the target corpus file
OUT_PATH=~/synthetic-cm-gen/cm_hi-en/ # path to the save directory

# : '
# Step 1: Preprocessing the data
echo -e "\n-------- Preparing data for alignment --------\n"

python prepare_data.py \
	--src_path=$SRC_PATH \
	--tgt_path=$TGT_PATH \
	--src_lang 'hi' \
	--tgt_lang 'en' \
	--output_path=$OUT_PATH

# Step 2: Extracting the word-level alignments
# Please choose the method of getting word alignments and update the following script (only Step 2 needs to be updated)

# Getting word alignments using awesome-align
echo -e "\n-------- Running alignment --------\n"

CUDA_VISIBLE_DEVICES=0 awesome-align \
	--output_file="${OUT_PATH}/src-tgt.align.out" \
	--model_name_or_path=bert-base-multilingual-cased \
	--data_file="${OUT_PATH}/src-tgt.align.in" \
	--extraction 'softmax' \
	--batch_size 128

# getting word alignments using fast_align
# ./fast_align -i '${OUT_PATH}/src-tgt.align.in' -d -o -v >'${OUT_PATH}/src-tgt.align.out'

# Step 3: Generating the synthetic code-mixed data
echo -e "\n-------- Generating synthetic code-mixed data --------\n"

python generate_cm_data.py \
	--tokenized_src_path="${OUT_PATH}/src.tok" \
	--tokenized_tgt_path="${OUT_PATH}/tgt.tok" \
	--align_out_path="${OUT_PATH}/src-tgt.align.out" \
	--src_stopwords_path ~/synthetic-cm-gen/stopwords/hi.txt \
	--tgt_stopwords_path ~/synthetic-cm-gen/stopwords/en.txt \
	--src_lang 'hi' \
	--tgt_lang 'en' \
	--output_path=$OUT_PATH

# Step 4: Calculating the metrics
echo -e "\n-------- Calculating the metrics --------\n"
python cm_metrics.py \
	--tokenized_src_path="${OUT_PATH}/src.tok" \
	--tokenized_tgt_path="${OUT_PATH}/tgt.tok" \
	--tokenized_cm_path="${OUT_PATH}/src-tgt.cm.tok"
