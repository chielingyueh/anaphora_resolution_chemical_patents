# Anaphora Resolution in Chemical Patents

## Model Architecture

[Stress Testing BERT Anaphora Resolution Models for Reaction Extraction in Chemical Patents](https://arxiv.org/abs/2306.13379)

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68673458-1b090d00-0597-11ea-96b1-7c1453e6edbb.png" />  
</p>

## Dependencies

- python>=3.6
- torch==1.6.0
- transformers==3.3.1

## BERT models used for experiments:

1. Bert-base: "bert-base-uncased" 
2. Bert-large: "bert-large-uncased"
3. BioBert: "dmis-lab/biobert-v1.1"
4. Clinical bert: "emilyalsentzer/Bio_ClinicalBERT"
5. Pubmed bert: "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16"

## How to run

### Running the k-fold cross-validation

First, we prepare data for the k-fold CV (default is 5-fold):
```bash
python kfold.py --path_to_input_file "./data/full_train.tsv"
```
Second, we run the 5-fold CV:

```bash
python main.py --do_train --do_eval --train_file 'train-0.tsv' --test_file 'val-0.tsv' --eval_dir "./eval/fold_1" --fold "fold1"
python main.py --do_train --do_eval --train_file 'train-1.tsv' --test_file 'val-1.tsv' --eval_dir "./eval/fold_2" --fold "fold2"
python main.py --do_train --do_eval --train_file 'train-2.tsv' --test_file 'val-2.tsv' --eval_dir "./eval/fold_3" --fold "fold3"
python main.py --do_train --do_eval --train_file 'train-3.tsv' --test_file 'val-3.tsv' --eval_dir "./eval/fold_4" --fold "fold4"
python main.py --do_train --do_eval --train_file 'train-4.tsv' --test_file 'val-4.tsv' --eval_dir "./eval/fold_5" --fold "fold5"
```

### Running test prediction

First, we create answer_keys files for the evaluation:

```bash
python answer_keys.py --path_to_input_file "./data/full_train.tsv" --output_file "answer_keys_train.txt"
python answer_keys.py --path_to_input_file "./data/test.tsv" --output_file "answer_keys_test.txt"
```

Second, we run model for test predictions:

```bash
python main.py --do_train --do_eval --train_file 'full_train.tsv' --test_file 'test.tsv' --eval_dir "./eval/test"
```

Note: one can use the ```running-model.ipynb``` 

## References

- [Stress Testing BERT Anaphora Resolution Models for Reaction Extraction in Chemical Patents](https://arxiv.org/abs/2306.13379)
