import argparse


def process_files(text_path: str, terms_path: str, result_path: str) -> None:
    with open(text_path, 'r') as f:
        content = f.read()

    with open(terms_path, 'r') as f:
        terms = f.readlines()

    with open(result_path, 'w') as f:
            f.write(content)


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
