
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
import torch, os, evaluate, wandb


from datasets import Dataset, load_dataset, concatenate_datasets, Features, ClassLabel, Value
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, get_cosine_schedule_with_warmup, TrainerCallback

DATA_PATH = '/shevtsov/sent_datasets/'
OUT_PATH = '/shevtsov/sent_results'

id2label = {0: "negative", 1: "neutral", 2: 'positive'}
label2id = {"negative": 0, "neutral": 1, 'positive': 2}

dataset_names = ["tyqiangz/multilingual-sentiments", #multi
                 "cardiffnlp/tweet_sentiment_multilingual", #multi
                 "mteb/tweet_sentiment_multilingual", #multi
                 #"stanfordnlp/sentiment140", #eng
                 "Sp1786/multiclass-sentiment-analysis-dataset", #eng
                 ]

Default_Class_Names = {#"stanfordnlp/sentiment140": ['positive', 'neutral', 'negative'],# according to https://huggingface.co/datasets/stanfordnlp/sentiment140/blob/main/sentiment140.py line 52.}
                       "Sp1786/multiclass-sentiment-analysis-dataset": ['negative', 'neutral', 'positive'], #according to https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset
                       "mteb/tweet_sentiment_multilingual": ['negative', 'neutral', 'positive'] # https://huggingface.co/datasets/mteb/tweet_sentiment_multilingual
                       }
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
#torch.set_default_device('cuda')

target_features = Features({
        'text': Value('string'),
        'labels': ClassLabel(names=list(id2label.values()))
        })

model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base",
        num_labels=3, id2label=id2label, label2id=label2id)

model.config.classifier_dropout = 0.3  # Set classifier dropout rate
model.config.hidden_dropout_prob = 0.2  # Add hidden layer dropout
model.config.attention_probs_dropout_prob = 0.2  # Add attention dropout

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")



metric = evaluate.load("accuracy")

"""Custom trainer in order to take into the consideration during the training class imbalance between pos/neg/neu samples."""
class CustomTrainer(Trainer):
    def __init__(self, *args, tensor_class_w=None, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        #store the class weights
        self.tensor_class_w = tensor_class_w.float() if tensor_class_w is not None else None
        self.gamma = gamma


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits.float()

        # compute custom loss (suppose one has 2 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.tensor_class_w, reduction='none')
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        pt = torch.exp(-loss)
        focal_loss = ((1-pt)**self.gamma*loss).mean()
        return (focal_loss, outputs) if return_outputs else focal_loss

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

def label_standarization(sample, mapper):
    """Convert label to integer"""
    sample['label'] = mapper[sample['label']]
    return sample

def get_dataset_split(split):
    assert split in ['train', 'validation']
    data_split = []
    for dname in dataset_names:
        print(f'Loading dataset: {dname}')
        if dname in ['stanfordnlp/sentiment140', 'Sp1786/multiclass-sentiment-analysis-dataset']:
            data = load_dataset(dname, split=split)
            data = Dataset.from_dict({
                                    'text': data['text'],
                                    'label': data['sentiment'] if 'label' not in data.features.keys() else data['label'],
                                    },
                                    features=Features({
                                                        'text': Value('string'),
                                                        'label': ClassLabel(names=Default_Class_Names[dname])
                                    })
                                    )

        else:
            try:
                data = load_dataset(dname, 'all', split=split)
            except:
                data = load_dataset(dname, 'default', split=split)
                data = Dataset.from_dict({
                                            'text': data['text'],
                                            'label': [int(label) for label in data['label']],
                                        },
                                            features=Features({
                                                'text': Value('string'),
                                                'label': ClassLabel(names=Default_Class_Names[dname])
                                            })
                                        )

        """Remove unnecessary columns from the dataset. Only text and label columns are needed."""
        to_drop = list(set(data.features.keys()) - set(['text', 'label']))
        if len(to_drop) != 0:
            data = data.remove_columns(to_drop)
        """Create translator (aka mapper) that correspond dataset labels to the standard label type."""
        mapper = {id: label2id[label.lower()] for label, id in zip(data.features['label'].names, label2id.values())}
        #data = data.map(lambda x:label_standarization(x, mapper), batched=False)
        # Convert dataset2 to use the same schema
        data = Dataset.from_dict({
                                  'text': data['text'],
                                  'labels': [mapper[label] for label in data['label']]
                                 }, features=target_features)

        data_split.append(data)

    return concatenate_datasets(data_split)



def data_loader():
    """Load Training dataset portions and cast them into dataset format"""
    #train_data = pd.concat([pd.read_csv(f'{DATA_PATH}{filename}', header=0, sep='\t') for filename in os.listdir(DATA_PATH) if filename.endswith('_train.tsv')])
    train_data = get_dataset_split('train')
    val_data = get_dataset_split('validation')

    """Drop duplicated into two stages 1st: drop 1 of the duplicates where text and label is similar. 
       2nd drop any instances with identical text but different labels."""
    #train_data.drop_duplicates(keep='first', inplace=True) #Keep only single sample from similar samples with identical targets.
    #train_data = train_data.iloc[train_data['text'].drop_duplicates(keep=False).index] #Remove any identical samples with different targets
    #train_data = Dataset.from_pandas(train_data.sample(frac=1).reset_index(drop=True))
    """Similarly read and cast validation portions"""
    #val_data = pd.concat([pd.read_csv(f'{DATA_PATH}{filename}', header=0, sep='\t') for filename in os.listdir(DATA_PATH) if filename.endswith('_val.tsv') ])

    "k""Drop duplicated texts"""
    #val_data.drop_duplicates(keep='first', inplace=True)
    #val_data = Dataset.from_pandas(val_data.iloc[val_data['text'].drop_duplicates(keep=False).index])

    return train_data, val_data

train_data, val_data = data_loader()
tokenized_train_datasets = train_data.map(tokenize_function, batched=True)
tokenized_val_datasets = val_data.map(tokenize_function, batched=True)

parser = argparse.ArgumentParser()
parser.add_argument('--cont', action='store_true')
args = parser.parse_args()
if __name__ == '__main__':
    if args.cont:
        print('Continue previous fine-tuning ...')
    # Calculate total training steps
    training_args = TrainingArguments(
        label_smoothing_factor=0.1,  # Add label smoothing
        #load_best_model_at_end=True,
        evaluation_strategy="epoch",
        greater_is_better=True,
        # Adding weight decay
        weight_decay=0.02,
        num_train_epochs=10 if not args.cont else 5 + 5,
        learning_rate=5e-6,#1e-5,
        optim="adamw_torch",
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        #weight_decay=0.01,
        adam_epsilon = 1e-6,
        max_grad_norm=0.5,#1.0, # clipping
        #lr_scheduler_type='linear',
        lr_scheduler_type='cosine',
        per_device_train_batch_size=24,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        fp16=False,
        output_dir=OUT_PATH,
        #eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="f1",
        save_total_limit=3,
        resume_from_checkpoint=args.cont,
    )
    if args.cont:
        checkpoints = [ (checkpoint, int(checkpoint.split('-')[-1])) for checkpoint in os.listdir(OUT_PATH) if checkpoint.startswith('checkpoint-')]
        checkpoints.sort(key=lambda t: t[1], reverse=True)
        model = AutoModelForSequenceClassification.from_pretrained(f"{OUT_PATH}/{checkpoints[0][0]}")

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_val_datasets,
        compute_metrics=compute_metrics,
        tensor_class_w=torch.tensor(compute_class_weight(class_weight='balanced',
                                                         classes=np.unique(train_data['labels']),
                                                         y=train_data['labels']), device='cuda'),
        gamma=2.0,
    )

    run = wandb.init(project="DeBERTa-v3-Sentiment", name=datetime.now().strftime('%m/%d/%Y'))
    trainer.train(resume_from_checkpoint=args.cont)
