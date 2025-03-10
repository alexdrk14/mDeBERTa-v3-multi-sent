# Model 

Multi-language sentiment classification model developed over the multi-language Microsoft [mDeBERTa-v3 base model](https://huggingface.co/microsoft/mdeberta-v3-base). 
This model where originally trained over [CC100](https://huggingface.co/datasets/statmt/cc100) multi-lingual dataset with more that 100+ languages. In this repo we provide fine-tuned model towards the multi-language sentiment analysis.
Model where trained on mulitple datasets with multiple languages with additional weights over class (sentiment categories: Negative, Positive, Neutral).
In order to train the model the following dataset where used:
 - tyqiangz/multilingual-sentiments
 - cardiffnlp/tweet_sentiment_multilingual
 - mteb/tweet_sentiment_multilingual
 - Sp1786/multiclass-sentiment-analysis-dataset
 - ABSC amazon review
 - SST2

# Evaluation and comparison with Vanilla and GPT-4o model:

| Dataset          | Model  | F1     | Accuracy |
|------------------|--------|--------|----------|
|                  | Vanilla| 0.0000 | 0.0000   |
| **sst2**         | Our    | 0.6161 | 0.9231   |
|                  | GPT-4  | 0.6113 | 0.8605   |
|---|---|---|---|
|                  | Vanilla| 0.2453 | 0.5820   |
| **sent-eng**     | Our    | 0.6289 | 0.6470   |
|                  | GPT-4  | 0.4611 | 0.5870   |
|---|---|---|---|
|                  | Vanilla| 0.0889 | 0.1538   |
| **sent-twi**     | Our    | 0.3368 | 0.3488   |
|                  | GPT-4  | 0.5049 | 0.5385   |
|---|---|---|---|
|                  | Vanilla| 0.0000 | 0.0000   |
| **mixed**        | Our    | 0.5644 | 0.7786   |
|                  | GPT-4  | 0.5336 | 0.6863   |
|---|---|---|---|
|                  | Vanilla| 0.1475 | 0.2842   |
| **absc-laptop**  | Our    | 0.5513 | 0.6682   |
|                  | GPT-4  | 0.6679 | 0.7642   |
|---|---|---|---|
|                  | Vanilla| 0.1045 | 0.1858   |
| **absc-rest**    | Our    | 0.6149 | 0.7726   |
|                  | GPT-4  | 0.7057 | 0.8385   |
|---|---|---|---|
|                  | Vanilla| 0.1455 | 0.2791   |
| **stanford**     | Our    | 0.8352 | 0.8353   |
|                  | GPT-4  | 0.8045 | 0.8032   |
|---|---|---|---|
|                  | Vanilla| 0.0000 | 0.0000   |
| **amazon-var**   | Our    | 0.6432 | 0.9647   |
|                  | GPT-4  | -----  | 0.9450   |

F1 score is measured with macro average computation parameter. 

# Execution 

Create python virtual enviroment and install the requirement libraries:
```bash
python3.12 -m venv pyvenv
source pyvenv/bin/activate # activates venv with installed requirements 
python3.12 -m pip install -r requirements.txt
```

In case of single GPU or CPU training execute:
```bash
source pyvenv/bin/activate # activates venv with installed requirements 
python3 fine_tune.py 
```
In case of multiple GPUs and DDP training execute:
```bash
source pyvenv/bin/activate # activates venv with installed requirements 
python -m torch.distributed.launch --nproc_per_node XX fine_tune.py # Where XX is number of available GPUs instances
```

Execution will report to the [wandb](https://wandb.ai) account during the model fine-tuning at each epoch.

# HuggingFace model checkpoints
[alexander-sh/mDeBERTa-v3-multi-sent](https://huggingface.co/alexander-sh/mDeBERTa-v3-multi-sent)

# Author 

Alexander Shevtsov alex.drk14[at]gmail.com
