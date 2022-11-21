import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import nltk
nltk.download("all")
import matplotlib.pyplot as plt
import transformers
from datasets import Dataset
dataset = Dataset.from_pandas(df)

###load data


def getR8():
  r8train = pandas.read_csv('/zhp/289g/R8/train.txt', sep='\t', names=['label', 'sentence'])
  r8test = pandas.read_csv('/zhp/289g/R8//test.txt', sep='\t', names=['label', 'sentence'])
  return r8train, r8test

def getMovieReview():
  trainX = pandas.read_csv('/zhp/289g/MovieReview/text_train.txt', sep='\t', names=['sentence'])
  trainY = pandas.read_csv('/zhp/289g/MovieReview/label_train.txt', sep='\t', names=['label'])
  train = pandas.concat([trainX, trainY], axis=1)
  testX = pandas.read_csv('/zhp/289g/MovieReview/text_test.txt', sep='\t', names=['sentence'])
  testY = pandas.read_csv('/zhp/289g/MovieReview/label_test.txt', sep='\t', names=['label'])
  test = pandas.concat([testX, testY], axis=1)
  return train, test

r8train, r8test = getR8()
r8 = pd.DataFrame({"train":r8train, "test":r8test})
mr_train, mr_test = getMovieReview()
mr = pd.DataFrame({"train":mr_train, "test":mr_test})
mr_dataset = Dataset.from_pandas(mr)
r8_dataset = Dataset.from_pandas(r8)


### load model
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_MODEL = "bert-base-uncased"
from transformers.tokenization_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)
tokenized_mr = mr_dataset.map(preprocess_function, batched=True)
tokenized_r8 = r8_dataset.map(preprocess_function, batched=True)



from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import BertForSequenceClassification,TrainingArguments, Trainer
model = BertForSequenceClassification.from_pretrained(BERT_MODEL)
model.to(device)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=20,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_mr["train"], #change dataset here for R8
    eval_dataset=tokenized_mr["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()


### evaluation, load trained model and get accuracy on testset
from evaluate import evaluator
from datasets import load_dataset
SAVE_CHECK = "/zhp/289g/MovieReview/results/checkpoint",
model_check = BertForSequenceClassification.from_pretrained(SAVE_CHECK)
task_evaluator = evaluator("text-classification")
data = load_dataset("mr_dataset", split="test[:2]")
results = task_evaluator.compute(
    model_or_pipeline=model,
    data=data,
    metric="accuracy",
    label_mapping={"LABEL_0": 0.0, "LABEL_1": 1.0},
    strategy="bootstrap",
    n_resamples=10,
    random_state=0
)
print(results)

 

