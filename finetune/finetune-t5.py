from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

import torch
device = torch.device("cpu")

checkpoint = "/Users/hhwang/models/t5-small"
# checkpoint = "/Users/hhwang/models/flan-t5-small"

print('********* before finetune ***********')
tokenizer = AutoTokenizer.from_pretrained(checkpoint,use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
# print(model.config)
inputs = tokenizer.encode("translate English to Chinese: That is good", return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=20)
print('result: ',tokenizer.batch_decode(outputs))

data = [
    {"question": "今天天真好", "answer": "那一起打篮球去吧"},
    {"question": "translate English to Chinese: That is good", "answer": "Not bad"}
]

def preprocess_function(examples):
    inputs = tokenizer(examples["question"], max_length=32, truncation=True)
    labels = tokenizer(examples["answer"], max_length=32, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

from datasets import Dataset, load_dataset
dataset = Dataset.from_list(data)
dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
print(dataset)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
# print(data_collator([dataset[0], dataset[1]]))

training_args = Seq2SeqTrainingArguments(
    output_dir="checkpoints",
    overwrite_output_dir=True,
    use_cpu=True,
    do_train=True,
    do_eval=True,
    learning_rate=1e-3,
    lr_scheduler_type="constant",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    save_steps=10,
    save_total_limit=5,
    logging_first_step=True,
    logging_steps=1,
    # logging_dir="./",
    eval_steps=1,
    evaluation_strategy="steps",
    load_best_model_at_end=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=data_collator,
    # compute_metrics=compute_metrics
)

print('begin train')
trainer.train()
print('done train')

finetune_mode = "/tmp/outputs/t5-small"
trainer.save_model(finetune_mode)

print('********* after finetune ***********')
prompt = "translate English to Chinese: That is good"
model = AutoModelForSeq2SeqLM.from_pretrained(finetune_mode)
generator = pipeline("summarization", model=model, tokenizer=tokenizer)
print(generator(prompt))
