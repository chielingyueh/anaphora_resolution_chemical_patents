import argparse
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def read_tsv_to_df(input_file):
    """Reads a tab separated value file to dataframe."""
    colnames = ['label', 'sent']
    df = pd.read_csv(input_file, sep='\t', names=colnames, header=None)
    return df


def create_answer_keys(args):
    df = read_tsv_to_df(args.path_to_input_file)

    df.index = np.arange(8001, len(df) + 8001)
    final_answers = df[['label']]
    final_answers.to_csv('./eval/' + args.output_file, sep="\t", header=False, encoding="utf-8")

    print(f"Answer keys file created for {args.path_to_input_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_input_file",
        default="./data/full_train.tsv",
        type=str,
        help="Path to the file where you want to create answer keys file for.",
    )

    parser.add_argument(
        "--output_file",
        default="answer_keys.txt",
        type=str,
        help="Name of output file.",
    )

    answers_config = parser.parse_args()
    create_answer_keys(answers_config)
