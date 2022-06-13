import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def read_tsv_to_df(input_file):
    """Reads a tab separated value file to dataframe."""
    colnames = ['label', 'sent']
    df = pd.read_csv(input_file, sep='\t', names=colnames, header=None)
    return df


def create_kfold_data(kfold_config):
    """Creates input dataframes for k-fold cross-validation, and creates ground truth file for the evaluation."""

    df = read_tsv_to_df(kfold_config.path_to_input_file)
    skf = StratifiedKFold(n_splits=kfold_config.k, shuffle=True)
    X = df.drop('label', axis=1)
    y = df.label

    for i, item in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[item[0]], X.iloc[item[1]]
        y_train, y_test = y.iloc[item[0]], y.iloc[item[1]]
        train = pd.concat([y_train, X_train], axis=1)
        test = pd.concat([y_test, X_test], axis=1)

        train.to_csv('./data/train-' + str(i) + '.tsv', sep="\t", header=None, index=False, encoding="utf-8")
        test.to_csv('./data/val-' + str(i) + '.tsv', sep="\t", header=None, index=False, encoding="utf-8")

        train.index = np.arange(8001, len(train) + 8001)
        train_answers = train[['label']]
        train_answers.to_csv('./eval/answer_keys_train-' + str(i) + '.txt', sep="\t", header=False, encoding="utf-8")

        test.index = np.arange(8001, len(test) + 8001)
        test_answers = test[['label']]
        test_answers.to_csv('./eval/answer_keys_val-' + str(i) + '.txt', sep="\t", header=False, encoding="utf-8")

    print(f"{kfold_config.k}-fold CV data is created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_input_file",
        default="./data/full_train.tsv",
        type=str,
        help="Path to the train file on which the kfold CV is done. ",
    )

    parser.add_argument(
        "--k",
        default="5",
        type=int,
        help="Number of folds",
    )

    kfold_config = parser.parse_args()
    create_kfold_data(kfold_config)
