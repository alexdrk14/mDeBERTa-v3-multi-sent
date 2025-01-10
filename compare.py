from mmap import ACCESS_COPY

import torch, sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import pandas as pd
#from load_data import DATA_PATH#, filename

from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

DATA_PATH = './datasets/'
"""
GPT results

Amazon variety 94.5 Accuracy
SST-2 (val portion) 93.12 acc
ABSC laptop acc: 76.42 F1:  66.79   (macro F1)
ABSC restaurant acc: 83.85 , F1: 70.57 (macro F1)
"""
GPT_known = {'sst2': (0.00, 0.931),
             'amazon-var' : (0.00, 0.945),
             'absc-laptop' : (0.6679, 0.7642),
             'absc-rest': (0.7057, 0.8385),
             'stanford': (0.8032, 0.8032)}


DATASETS= {'sst2': 'sst2_val.tsv',
           'sent-eng': 'sent-english-gpt.csv',
           'sent-twi': 'sent-twi-gpt.csv',
           'mixed': 'MixedVal.tsv',#'MixedVal_val.tsv',
           'absc-laptop': 'ABSC-laptop-trial_test.tsv',
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

    #for item in classifier(val_data['text'], ['positive', 'neutral', 'negative'], multi_label=False):

    for item in classifier(val_data['text']):
        #print(item)
        target = 2
        if item['label'].lower() == 'negative':
            target = 0
        elif item['label'].lower() == 'positive':
            target = 1
        pred_y.append(target)
    #print(len(pred_y))
    #print(set(pred_y))
    #print(len(y_true))
    f1 = f1_score(y_true, pred_y, average='macro')#'weighted')
    #precision = precision_score(y_true, pred_y, average='weighted')
    #recall = recall_score(y_true, pred_y, average='weighted')
    acc = accuracy_score(y_true, pred_y)
    #print(f'Model - F1: {f1:.4f}, Acc: {acc:.4f}')

    if 'gpt4' in val_data.column_names:
        y_gpt = val_data['gpt4']
        f1_gpt = f1_score(y_true, y_gpt, average='macro')#'weighted')
        #precision = precision_score(y_true, y_gpt, average='weighted')
        #recall = recall_score(y_true, y_gpt, average='weighted')
        acc_gpt = accuracy_score(y_true, y_gpt)
        #print(f'-GPT4 - F1: {f1:.4f}, Acc: {acc:.4f}')
    else:
        f1_gpt, acc_gpt = GPT_known[dataset]

    print(f'Model - F1: {f1:.4f}, Acc: {acc:.4f}')
    print(f'-GPT4 - F1: {f1_gpt:.4f}, Acc: {acc_gpt:.4f}')
    print('-'*31)
print('Starting')

#model = AutoModelForSequenceClassification.from_pretrained("mrm8488/deberta-v3-ft-financial-news-sentiment-analysis")#"./results/checkpoint-324/")#"MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")#"mrm8488/deberta-v3-ft-financial-news-sentiment-analysis")#"MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#model.to(device)
print(f'model loaded to {device}')
#val_data = Dataset.from_pandas(pd.read_csv(f'{DATA_PATH}sst2_test.tsv', header=0, sep='\t'))
#y_true = val_data['labels']
task = "sentiment-analysis"#"text-classification"#"text-classification"#"zero-shot-classification"#"text-classification"
model_id = "./results/checkpoint-650440/"#"./results/checkpoint-25920/"#"MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"#"mrm8488/deberta-v3-ft-financial-news-sentiment-analysis"

classifier = pipeline(task, model_id, tokenizer=tokenizer, device=device, batch_size=64)

for dname in  DATASETS:
    get_sentiment_perf(dname, classifier)