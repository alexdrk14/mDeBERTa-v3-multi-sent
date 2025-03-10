import numpy as np
import pandas as pd
import argparse
from copy import deepcopy
from datetime import datetime
import torch, os, evaluate, wandb
from extra import *


from datasets import Dataset, load_dataset, concatenate_datasets, Features, ClassLabel, Value
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, get_cosine_schedule_with_warmup, TrainerCallback

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

target_features = None

model = None

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
        labels = labels.to(model.device)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits.float()
        logits = logits.to(model.device)
        # compute custom loss (suppose one has 2 labels with different weights)
        loss = torch.nn.CrossEntropyLoss(weight=self.tensor_class_w, reduction='none')
        loss = loss.to(model.device)
        if self.tensor_class_w is not None:
            """In case of imbalance data compute focal loss"""
            loss = loss(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            pt = torch.exp(-loss)
            loss = ((1-pt)**self.gamma*loss).mean()
        return (loss, outputs) if return_outputs else loss

class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

def tokenize_function_sent(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")

def tokenize_function_xnli(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length", max_length=512)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
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
        data = Dataset.from_dict({
                                  'text': data['text'],
                                  'labels': [mapper[label] for label in data['label']]
                                 }, features=target_features)

        data_split.append(data)
     
    for datasetname in local_datasets:
        if split == 'validation':
            datasetname = datasetname.replace('train', 'val')
        df = pd.read_csv(datasetname, header=0, sep='\t')
        df['labels'] = [0 if item == 0 else 2 for item in df['labels']]
        df = df.dropna()
        df = Dataset.from_dict({
                                   'text': df['text'],
                                   'labels': df['labels'],
                                }, features=target_features)
        
        data_split.append(df)
    
    return concatenate_datasets(data_split).shuffle(seed=42)



def data_loader():
    class_weights = None

    """Load Training dataset portions and cast them into dataset format"""
    train_data = get_dataset_split('train')
    class_weights = torch.tensor(
                        compute_class_weight(
                                    class_weight='balanced',
                                    classes=np.unique(train_data['labels']),
                                    y=train_data['labels']),
                        device='cuda')
    train_data = train_data.map(tokenize_function_sent, batched=True)
    val_data = get_dataset_split('validation').map(tokenize_function_sent, batched=True)

    return train_data, val_data, class_weights

if __name__ == '__main__':
    train_tokens, val_tokens, class_w = data_loader()
    
    print(f'\n\nFine-tuning of the Sentiment task\n\n')
    target_features = Features({
        'text': Value('string'),
        'labels': ClassLabel(names=list(id2label.values()))
    })

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                                num_labels=3,
                                                                id2label=id2label,
                                                                label2id=label2id)

    model.config.classifier_dropout = 0.3  # Set classifier dropout rate
    model.config.hidden_dropout_prob = 0.2  # Add hidden layer dropout
    model.config.attention_probs_dropout_prob = 0.2  # Add attention dropout

    training_args = TrainingArguments(
        label_smoothing_factor=0.1,  # Add label smoothing
        evaluation_strategy="epoch",
        greater_is_better=True,
        # Adding weight decay
        weight_decay=0.02,
        num_train_epochs=10,
        learning_rate=5e-6,  # 1e-5,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-6,
        max_grad_norm=0.5,  # 1.0, # clipping
        lr_scheduler_type='cosine',
        per_device_train_batch_size=48,
        per_device_eval_batch_size=48,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        fp16=False,
        output_dir=OUT_PATH_SENT,
        logging_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="f1",
        save_total_limit=3,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokens,
        eval_dataset=val_tokens,
        compute_metrics=compute_metrics,
        tensor_class_w=class_w,
        gamma=2.0,
    )
    
    trainer.add_callback(CustomCallback(trainer))
    run = wandb.init(project="mDeBERTa-v3-Sentiment", name=f'SENT-{datetime.now().strftime("%m/%d/%y")}')
    trainer.train()
