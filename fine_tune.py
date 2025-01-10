import torch, os
import numpy as np
import evaluate, wandb
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import pandas as pd

from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DATA_PATH = './datasets/'

id2label = {0: "NEGATIVE", 1: "POSITIVE", 2: 'NEUTRAL'}
label2id = {"NEGATIVE": 0, "POSITIVE": 1, 'NEUTRAL': 2}

device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
#torch.set_default_device('cuda')

model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base",
        num_labels=3, id2label=id2label, label2id=label2id)
model.config.classifier_dropout = 0.2  # Set classifier dropout rate

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

run = wandb.init(project="DeBERTa-v3-Sentiment", name=datetime.now().strftime('%m/%d/%Y'))

metric = evaluate.load("accuracy")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    #accuracy = metric.compute(predictions=predictions, references=labels)
    #return metric.compute(predictions=predictions, references=labels)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
"""Load Training dataset portions and cast them into dataset format"""
train_data = pd.concat([pd.read_csv(f'{DATA_PATH}{filename}', header=0, sep='\t') for filename in os.listdir(DATA_PATH) if filename.endswith('_train.tsv')])
#Drop duplicated into two stages 1st: drop 1 of the duplicates where text and label is similar. 2nd drop any instances with identical text but different labels.
train_data.drop_duplicates(keep='first', inplace=True) #Keep only single sample from similar samples with identical targets.
train_data = train_data.iloc[train_data['text'].drop_duplicates(keep=False).index] #Remove any identical samples with different targets
train_data = Dataset.from_pandas(train_data.sample(frac=1).reset_index(drop=True))
"""Similarly read and cast validation portions"""
val_data = pd.concat([pd.read_csv(f'{DATA_PATH}{filename}', header=0, sep='\t') for filename in os.listdir(DATA_PATH) if filename.endswith('_val.tsv') ])
#Drop duplicated texts
val_data.drop_duplicates(keep='first', inplace=True)
val_data = Dataset.from_pandas(val_data.iloc[val_data['text'].drop_duplicates(keep=False).index])


tokenized_train_datasets = train_data.map(tokenize_function, batched=True)
tokenized_val_datasets = val_data.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    num_train_epochs=5,
    learning_rate=2e-5,
    optim="adamw_torch",
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    weight_decay=0.01,
    adam_epsilon = 1e-6,
    max_grad_norm=1.0, # clipping
    lr_scheduler_type='linear',
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1, #the default gradient accumulation step is 2 with batch size of 16, making gradient update after 64 samples
    warmup_ratio=0.06,
    fp16=False,
    output_dir="./results",
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="f1",
    save_total_limit=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_val_datasets,
    compute_metrics=compute_metrics,
)
trainer.train()
