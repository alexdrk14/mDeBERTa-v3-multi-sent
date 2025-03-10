"""Download datasets"""
import requests, os
import pandas as pd

DATA_PATH = './datasets/'

def github_raw(url, filename):
    if os.path.exists(f'{DATA_PATH}{filename}'):
        return True
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Check for HTTP request errors

        with open(f'{DATA_PATH}{filename}', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        return False

    if filename.endswith('tsv') and 'SST-2-sentiment-analysis' in url:
        df = pd.read_csv(f'{DATA_PATH}{filename}', sep='\t', header=None)
        df.columns = ['labels', 'text']
        df['labels'] = [0 if item == 0 else 2 for item in df['labels']]
        df.to_csv(f'{DATA_PATH}{filename}', sep='\t', index=False, header=True)
    return True

SST2 = [("https://raw.githubusercontent.com/YJiangcm/SST-2-sentiment-analysis/refs/heads/master/data/test.tsv", 'sst2_test.tsv'),
        ("https://raw.githubusercontent.com/YJiangcm/SST-2-sentiment-analysis/refs/heads/master/data/dev.tsv", 'sst2_val.tsv'),
        ("https://raw.githubusercontent.com/YJiangcm/SST-2-sentiment-analysis/refs/heads/master/data/train.tsv", 'sst2_train.tsv')]

for file_url , filename in SST2:
    github_raw(file_url, filename)

#-----------------------------------
from datasets import load_dataset
ment = load_dataset('sentiment140')  # source: https://huggingface.co/datasets/sentiment140/viewer/sentiment140/test

def parse(data, portion, filename):
    text = []
    labels = []
    for item in data[portion]:
        text.append(item['text'])
        label = item['sentiment']
        label = int(label/2)

        labels.append(label)
    df = pd.DataFrame({'labels': labels, 'text': text})
    df.to_csv(f'{DATA_PATH}{filename}_{portion}.tsv', sep='\t', index=False, header=True)


parse(ment, 'train', 'stanfordnlp')
parse(ment, 'test', 'stanfordnlp')


#https://osf.io/6pnb2/