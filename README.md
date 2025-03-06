## Synthetic Code-Mixed Corpus Generation using Word Alignments

<p align="center">
  <image src="alignment.png"/>
</p>
<p align="center">
  <em> Generating synthetic code-mixed Hinglish sentence from Hindi and English. </em>
</p>
    
This repo contains the code for generating synthetic code-mixed data from a given parallel corpus.  We follow the [Matrix Language Frame model (MLF)](https://en.wikipedia.org/wiki/Code-switching#Matrix_language-frame_model) to generate the code-mixed data. The words in the source language (considered the matrix language) are replaced with the target language (considered the embedded language) words based on the word alignment information. The following three steps describe the data generation process:

1. Preprocessing: The parallel corpus is pre-processed by removing the empty lines, tokenizing the data, and preparing the data to give as input to the alignment tool.
2. Word Alignments: The word-level alignments are extracted either by using [Awesome-Align](https://github.com/neulab/awesome-align) or [fast_align](https://github.com/clab/fast_align)
3. Generating the Synthetic Code-Mixed Data: The source words were replaced with the mapped target words based on the word alignments extracted. We only consider one-to-one mapped tokens for the replacement.

**Note:** The input to the code should be a parallel corpus.

### Dependencies
We tested the code on Python 3.7. The following tools are required to run the code.

1. Alignment tools: Word alignments can be extracted from Awesome-Align or the fast_align. Please refer to [Awesome-Align](https://github.com/neulab/awesome-align) or [fast_align](https://github.com/clab/fast_align) for the installation instructions.
2. Indic-NLP-Library: We use the Indic-NLP-Library to tokenize and detokenize the Indic language data. Please refer to [Indic-NLP-Library](https://github.com/anoopkunchukuttan/indic_nlp_library) for the installation instructions.
3. SacreMoses: We use the SacreMoses library to tokenize and detokenize the non-indic language data. Please refer to [Sacremoses](https://github.com/hplt-project/sacremoses) for the installation instructions.

### Files
There are a total of four files in this repo.

1. `prepare_data.py`: Preprocess (removing the empty lines, tokenization, and aligning the data for the alignment tool) and save the data.
2. `generate_cm_data.py`: Generates the synthetic code-mixed data based on the one-to-one word alignment information.
3. `cm_metrics.py`: Calculates the amount of code-mixing. The metrics used are: [Code-Mixing Index (CMI)](https://aclanthology.org/L16-1292/), [Switch-Point Fraction (SPF)](https://aclanthology.org/P18-1143/), [Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)#Definition).
4. `cm_gen.sh`: A shell file that runs the entire process, i.e., preprocessing the data, running word alignment, generating the synthetic code-mixed data, and calculating the code-mixing metrics.

### Language Support
We tested our code on the following languages:
1. English (en)
2. German (de)
3. French (fr)
4. Spanish (es)
5. Czech (cz)
6. Italian (it)
7. Russian (ru)
8. Hindi (hi)
9. Bengali (bn)
10. Assamese (as)
11. Marathi (mr)
12. Maithili (mai)
13. Gujarati (gu)
14. Punjabi (pa)
15. Odia (or)
16. Tamil (ta)
17. Telugu (te)
18. Kannada (kn)
19. Malayalam (ml)

Other languages are also supported, but please check the tokenizer and the detokenizer support for your language pair.

**Note:** Please use the ISO 639-1 code (the alpha-2 code contains only two letter language code) for the language code (Please refer to [ISO 639-2](https://www.loc.gov/standards/iso639-2/php/code_list.php) for more information).

### Optional Step: Stop Words
We noticed that replacing the stop words generates less fluent code-mixed outputs. Due to this reason, we are not replacing the stop words (i.e., if either the source or the target word is a stop word, then no replacement). The `stopwords` subfolder contains the stopword lists for some of the languages. The `links.txt` file in the `stopwords` subfolder contains the links from which we extracted these lists. This step is optional, but the code runs without these stopword lists.

### Citation
Please kindly cite the following paper if you found our code or approach helpful:

```
@inproceedings{appicharla2021iitp,
  title={IITP-MT at CALCS2021: English to Hinglish neural machine translation using unsupervised synthetic code-mixed parallel corpus},
  author={Appicharla, Ramakrishna and Gupta, Kamal Kumar and Ekbal, Asif and Bhattacharyya, Pushpak},
  booktitle={Proceedings of the Fifth Workshop on Computational Approaches to Linguistic Code-Switching},
  pages={31--35},
  year={2021}
}
```
