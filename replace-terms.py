import os
import argparse
import csv
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import namedtuple

# turn off GPU, we don't need it
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PhraseEmbedding = namedtuple('PhraseEmbedding', ['phrase', 'embedding'])
Replacement = namedtuple('Replacement', ['phrase', 'term', 'similarity'])


def get_sentence_phrase_list(sentence: str) -> List[str]:
    words = sentence.split()
    phrase_list = []

    for i in range(len(words)):
        for j in range(i, len(words)):
            phrase_list.append(" ".join(words[i:j + 1]))

    return phrase_list


def get_phrase_embedding_list(phrase_list: List[str]) -> List[PhraseEmbedding]:
    model = SentenceTransformer('all-mpnet-base-v2')
    embedding_list = model.encode(phrase_list)
    return [PhraseEmbedding(phrase, embedding) for phrase, embedding in zip(phrase_list, embedding_list)]


def get_sentence_replacement_list(sentence: List[PhraseEmbedding],
                                  term_list: List[PhraseEmbedding], threshold: float) -> List[Replacement]:
    replacement_list: List[Replacement] = []
    sentance_similarity_list = cosine_similarity([phrase.embedding for phrase in sentence],
                                                 [term.embedding for term in term_list])

    for phrase_index, phrase_similarity_list in enumerate(sentance_similarity_list):
        most_similar_term_index = np.argmax(phrase_similarity_list)
        best_similarity = phrase_similarity_list[most_similar_term_index]
        if best_similarity >= threshold:
            replacement_list.append(Replacement(sentence[phrase_index].phrase,
                                                term_list[most_similar_term_index].phrase, best_similarity))

    return replacement_list


def output_result(result_path: str, sentence_list: List[str], replacement_list: List[Replacement]) -> None:
    with open(result_path, 'w') as f:
        for sentence, sentence_replacemnet_list in zip(sentence_list, replacement_list):
            f.write(sentence + "\n")
            print(sentence + "\n")
            for replacement in sentence_replacemnet_list:
                f.write(f"\n")
                f.write(f"\t phrase: {replacement.phrase} \n")
                f.write(f"\t replacement: {replacement.term} \n")
                f.write(f"\t similarity {replacement.similarity} \n")
                print(f"\n")
                print(f"\t phrase: {replacement.phrase} \n")
                print(f"\t replacement: {replacement.term} \n")
                print(f"\t similarity {replacement.similarity} \n")

            print(f"\n")
            f.write(f"\n")


def find_replacement(text_path: str, terms_path: str, result_path: str, threshold: int) -> None:
    with open(text_path, 'r') as f:
        text = f.read()

    sentence_list = [sentence.strip() for sentence in text.split('.') if sentence]
    sentence_phrase_list = [get_sentence_phrase_list(sentence) for sentence in sentence_list]
    sentence_phrase_embedding_list = [get_phrase_embedding_list(candidate_list)
                                      for candidate_list in sentence_phrase_list]

    with open(terms_path, 'r') as f:
        term_list = [row[0].strip() for row in csv.reader(f)]

    term_embedding_list = get_phrase_embedding_list(term_list)
    replacement_list = [get_sentence_replacement_list(sentence_phrase_list, term_embedding_list, threshold) for
                        sentence_phrase_list in sentence_phrase_embedding_list]

    output_result(result_path, sentence_list, replacement_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze text based on terms and save the result.")

    default_text_path = "sample_text.txt"
    default_terms_path = "terms.csv"
    default_result_path = "improved-text.txt"
    default_threshold = 0.7

    parser.add_argument("--text", type=str, default=default_text_path, help="Path to the text file.")
    parser.add_argument("--terms", type=str, default=default_terms_path, help="Path to the terms file.")
    parser.add_argument("--result", type=str, default=default_result_path, help="Path to save the result.")
    parser.add_argument("--threshold", type=float, default=default_threshold, help="Threshold.")

    args = parser.parse_args()
    print('Start processing. Please wait')
    find_replacement(args.text, args.terms, args.result, args.threshold)
    print('Finished')
