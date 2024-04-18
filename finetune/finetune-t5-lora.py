
import torch
device = torch.device("cpu")

checkpoint = "/Users/hhwang/models/t5-small"
checkpoint = "/Users/hhwang/models/flan-t5-small"
prompt = "translate English to German: That is good"
print('********* before finetune ***********')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained(checkpoint,use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
# print(model.config)
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=20)
print('prompt:', prompt)
print('result: ',tokenizer.batch_decode(outputs))

print('***************** begin lora finetune *********************')
from peft import LoraConfig, TaskType
from peft import get_peft_model

lora_config = LoraConfig(
    r=16,
    target_modules=["q", "v"],
    task_type=TaskType.SEQ_2_SEQ_LM,
    lora_alpha=32,
    lora_dropout=0.05
)
# print(model)
lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

data = [
    {"question": "今天天真好", "answer": "那一起打篮球去吧"},
    {"question": prompt, "answer": "Not bad"}
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

from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
# print(data_collator([dataset[0], dataset[1]]))

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
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
    num_train_epochs=100,
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
    model=lora_model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=data_collator,
    # compute_metrics=compute_metrics
)
lora_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

print('begin train')
trainer.train()
print('done train')

lora_checkpoint = "/tmp/outputs/t5-small-lora"
lora_model.save_pretrained(lora_checkpoint)
print('Save', lora_checkpoint)

print('***************** after lora finetune *********************')
from peft import PeftModel, PeftConfig
config = PeftConfig.from_pretrained(lora_checkpoint)
# print(config)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
lora_model = PeftModel.from_pretrained(model, lora_checkpoint)
# inputs = tokenizer.encode(prompt, return_tensors="pt")
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")
# outputs = lora_model.generate(inputs)
outputs = lora_model.generate(input_ids=input_ids,max_length=100, temperature=0.7, do_sample=True)
# print('result: ',tokenizer.batch_decode(outputs))
print('prompt:', prompt)
print('result: ',tokenizer.decode(outputs[0]))
