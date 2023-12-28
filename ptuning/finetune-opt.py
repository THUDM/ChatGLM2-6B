# https://github.com/jesusoctavioas/Finetune_opt_bnb_peft

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cpu")
print("device:", device)

checkpoint = "/Users/hhwang/models/opt-125m"
checkpoint = "/Users/hhwang/models/opt-350m"
# checkpoint = "/Users/hhwang/models/gpt2"

print('checkpoint:', checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model = AutoModelForCausalLM.from_pretrained(checkpoint, load_in_8bit=True, device_map='auto')
model = AutoModelForCausalLM.from_pretrained(checkpoint)

batch = tokenizer("Two things are infinite: ", return_tensors='pt')
output_tokens = model.generate(**batch, max_new_tokens=50)
print('result:', tokenizer.decode(output_tokens[0], skip_special_tokens=True))

from datasets import load_dataset
# data = load_dataset("Abirate/english_quotes")
dataset = load_dataset("/Users/hhwang/models/dataset/english_quotes")
print('dataset', dataset)
dataset = dataset.map(lambda samples: tokenizer(samples['quote']), batched=True)
train_ds = dataset['train'].select(range(100))
print('train_ds', train_ds)

trainer = Trainer(
    model=model, 
    train_dataset=train_ds,
    args=TrainingArguments(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4, 
        # fp16=True, # only works on cuda
        logging_steps=1, 
        output_dir='outputs'
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
print('begin train')
trainer.train()
print('done train')