import os
from datasets import Dataset, ClassLabel
from transformers import Trainer
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import AutoTokenizer, DataCollatorWithPadding
import pandas as pd
import sys
sys.path.append('/opt/project/KG_Assist_LLM')

# %%


# %%


if not os.path.exists('/opt/project/KG_Assist_LLM/data/bert_train/data_for_train/data_train.csv'):
    data = pd.read_csv(
        '/opt/project/KG_Assist_LLM/data/pand/datas/pand_clean_process2-3.csv', index_col=0)

    def replace(x):
        x = x.upper()
        if 'YES' in x:
            return int(1)
        elif 'NO' in x:
            return int(0)
        else:
            return

    data = data[['body', 'is_mbti']]
    data.columns = ['body', 'labels']
    data.labels = data.labels.map(replace)
    data.dropna(inplace=True)
    data.labels = data.labels.astype(int)

    data.head()
    """
        body	labels
    0	i'm honestly surprised Alexandria isn't on thi...	1.0
    1	You really need to quit going to her	0.0
    2	Mother, look! He thinks he's people!!	1.0
    3	i love how the first one you linked is newer t...	0.0
    4	in July 2003 on my dad's (and grandpa's) farm,...	0.0
    """
    data_train, data_eval = train_test_split(
        data, test_size=0.1, random_state=42)
    data_train.to_csv(
        '/opt/project/KG_Assist_LLM/data/bert_train/data_for_train/data_train.csv')
    data_eval.to_csv(
        '/opt/project/KG_Assist_LLM/data/bert_train/data_for_train/data_eval.csv')
else:
    data_train = pd.read_csv(
        '/opt/project/KG_Assist_LLM/data/bert_train/data_for_train/data_train.csv', index_col=0)
    data_eval = pd.read_csv(
        '/opt/project/KG_Assist_LLM/data/bert_train/data_for_train/data_eval.csv', index_col=0)


data_train = Dataset.from_pandas(
    df=data_train
)
data_eval = Dataset.from_pandas(
    df=data_eval
)


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["body"], truncation=True, padding="max_length", max_length=512)


tokenized_train = data_train.map(tokenize_function, batched=True)
tokenized_eval = data_eval.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=2, trust_remote_code=True)


data_train = data_train.cast_column("labels", ClassLabel(num_classes=2))
data_eval = data_eval.cast_column("labels", ClassLabel(num_classes=2))

home = '/opt/project/KG_Assist_LLM/'

# tensorboard --logdir=/opt/project/KG_Assist_LLM/logs
training_args = TrainingArguments(
    output_dir=f"{home}data/model_train/bert_train",
    logging_dir=f"{home}logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    report_to="tensorboard",
    per_device_train_batch_size=16,
    num_train_epochs=15,
    learning_rate=2e-6,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()

# Step 8: 保存模型
trainer.save_model(
    "/opt/project/KG_Assist_LLM/data/pand/bert_train/model_save")
metrics = trainer.evaluate()

trainer.log_metrics("eval", metrics)
print(metrics)
