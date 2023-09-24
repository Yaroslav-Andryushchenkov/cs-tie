import os
import argparse
import csv
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# turn off GPU, we don't need it
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def get_sentence_phrase_list(sentence: str) -> List[str]:
    words = sentence.split()
    phrase_list = []

    for i in range(len(words)):
        for j in range(i, len(words)):
            phrase_list.append(" ".join(words[i:j+1]))

    return phrase_list



from collections import namedtuple

PhraseEmbedding = namedtuple('PhraseEmbedding', ['phrase', 'embedding'])
Replacement = namedtuple('Replacement', ['phrase', 'term', 'similarity'])


def get_phrase_embedding_list(phrase_list: List[str]) -> List[PhraseEmbedding]:
    model = SentenceTransformer('all-mpnet-base-v2').to('cpu')
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
                                                term_list[most_similar_term_index].phrase,best_similarity))

    return replacement_list




def process_files(text_path: str, terms_path: str, result_path: str) -> None:
    with open(text_path, 'r') as f:
        text = f.read()

    sentence_list = [sentence.strip() for sentence in text.split('.') if sentence]
    sentence_candidate_list = [get_sentence_phrase_list(sentence) for sentence in sentence_list]
    sentence_candidate_embedding_list = [get_phrase_embedding_list(candidate_list)
                                         for candidate_list in sentence_candidate_list]

    with open(terms_path, 'r') as f:
        term_list = [row[0].strip() for row in csv.reader(f)]

    term_embedding_list = get_phrase_embedding_list(term_list)
    replacement_list = [get_sentence_replacement_list(sentence_phrase_list, term_embedding_list, 0.8) for
                        sentence_phrase_list in sentence_candidate_embedding_list]


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
                print(f"\t similarity {replacement.similarity } \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze text based on terms and save the result.")

    default_text_path = "sample_text.txt"
    default_terms_path = "terms.csv"
    default_result_path = "improved-text.txt"

    parser.add_argument("--text", type=str, default=default_text_path, help="Path to the text file.")
    parser.add_argument("--terms", type=str, default=default_terms_path, help="Path to the terms file.")
    parser.add_argument("--result", type=str, default=default_result_path, help="Path to save the result.")

    args = parser.parse_args()

    process_files(args.text, args.terms, args.result)
