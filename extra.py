DATA_PATH = './datasets/'
OUT_PATH_SENT = './sent_checkpoints/'
OUT_PATH_XNLI = './xnli_checkpoints/'
MODEL_NAME = "microsoft/mdeberta-v3-base"

id2label = {0: "negative", 1: "neutral", 2: 'positive'} 
label2id = {"negative": 0, "neutral": 1, 'positive': 2}


# Convert labels into numerical format
def convert_labels(example):
    example["label"] = label2id[example["label"]]
    return example

dataset_names = ["tyqiangz/multilingual-sentiments", #multi
                 "cardiffnlp/tweet_sentiment_multilingual", #multi
                 "mteb/tweet_sentiment_multilingual", #multi
                 "Sp1786/multiclass-sentiment-analysis-dataset", #eng
                 ]

local_datasets = [f'{DATA_PATH}amazon_rev_train.tsv',
                 '{DATA_PATH}sst2_train.tsv']

Default_Class_Names = {#"stanfordnlp/sentiment140": ['positive', 'neutral', 'negative'],# according to https://huggingface.co/datasets/stanfordnlp/sentiment140/blob/main/sentiment140.py line 52.}
                       "Sp1786/multiclass-sentiment-analysis-dataset": ['negative', 'neutral', 'positive'], #according to https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset
                       "mteb/tweet_sentiment_multilingual": ['negative', 'neutral', 'positive'] # https://huggingface.co/datasets/mteb/tweet_sentiment_multilingual
                       }