from mmap import ACCESS_COPY

import torch, sys
import pandas as pd
from extra import DATA_PATH
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

GPT_known = { 
             'amazon-var' : (0.00, 0.945),
             'absc-rest': (0.7057, 0.8385),
             'stanford': (0.8032, 0.8032)
             }


DATASETS= {'sst2': 'sst2_test_gpt.tsv',
           'sent-eng': 'sent-english-gpt.csv',
           'sent-twi': 'sent-twi-gpt.csv',
           'mixed': 'MixedVal.tsv',#'MixedVal_val.tsv',
           'absc-laptop': 'ABSC-laptop-trial_test_gpt.tsv',#'ABSC-laptop-trial_test.tsv',
           'absc-rest' : 'ABSC-restaurants-trial_test.tsv',
           'stanford': 'stanfordnlp_test.tsv',
           'amazon-var' : 'amazon_rev_test.tsv',
}

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def get_sentiment_perf(dataset, classifier):
    pred_y = []
    print(dataset)
    val_data = Dataset.from_pandas(pd.read_csv(f'{DATA_PATH}{DATASETS[dataset]}', header=0, sep='\t'))

    y_true = val_data['labels']

    for item in classifier(val_data['text']):
        #Need to cast prediction labels to speccific numbers since those portion do not comply with the label2id of the model
        target = 2
        if item['label'].lower() == 'negative':
            target = 0
        elif item['label'].lower() == 'positive':
            target = 1
        pred_y.append(target)
    f1 = f1_score(y_true, pred_y, average='macro')
    acc = accuracy_score(y_true, pred_y)

    if 'gpt4' in val_data.column_names:
        y_gpt = val_data['gpt4']
        f1_gpt = f1_score(y_true, y_gpt, average='macro'
        acc_gpt = accuracy_score(y_true, y_gpt)
    else:
        f1_gpt, acc_gpt = GPT_known[dataset]

    print(f'Model - F1: {f1:.4f}, Acc: {acc:.4f}')
    print(f'-GPT4 - F1: {f1_gpt:.4f}, Acc: {acc_gpt:.4f}')
    print('-'*31)
print('Starting')

print(f'model loaded to {device}')

task = "sentiment-analysis"
classifier = pipeline(task, model='alexander-sh/mDeBERTa-v3-multi-sent', device=device, batch_size=64)

for dname in  DATASETS:
    get_sentiment_perf(dname, classifier)
