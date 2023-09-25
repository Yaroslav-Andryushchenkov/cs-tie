# Overview

The script parses the text into sentences and, for all possible phrases formed
from the words of the sentence, suggests a replacement from the list of terms.
For each phrase, a replacement is chosen with the term having the highest cosine
similarity of the embedding. As a result, replacements will be suggested for which
the cosine similarity exceeds a specified threshold. By default, the threshold is 0.7,
however, the user can set their own threshold in the script parameters.
Embeddings are computed using the pretrained transformer 'all-mpnet-base-v2'.
The results are output to both the console and a file.

# Installation

You need python 3.8 to run the script.

To install all necessary dependencies run:
    
    pip install -r requirements.txt


# Script run

## Script parameters
    --text Path to a text file to imporove. Default value is 'sample_text.txt'.
    --terms Path to a csv file with terms. Default value is 'terms.csv'.
    --result Path to a file to save replacement suggestions in sentences. Default value is 'imporved-text.txt'.
    --threshold Only terms with similarity above threshold value are suggested as replacements. Default value is 0.7.

## Script run example:
default parameters:

    python -m replace-terms

custom parameters:

    python -m replace-terms --threshold 0.69 --result delme.txt







