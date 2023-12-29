import torch

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("device:", device)

checkpoint = "/Users/hhwang/models/opt-350m"
checkpoint = "/Users/hhwang/models/opt-125m"

prompt = "No matter how plain a woman may be"
print('***************** before lora finetune *********************')
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
batch = tokenizer(prompt, return_tensors='pt')
output_tokens = model.generate(**batch, max_new_tokens=50)
print('prompt:', prompt)
print('result:', tokenizer.decode(output_tokens[0], skip_special_tokens=True))

print('***************** begin lora finetune *********************')
from peft import LoraConfig, TaskType
from peft import get_peft_model
print(model)
lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.05
)
lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()


from datasets import load_dataset
dataset = load_dataset("/Users/hhwang/models/dataset/english_quotes")
dataset = dataset.map(lambda samples: tokenizer(samples['quote']), batched=True)
train_ds = dataset['train'].select(range(100))

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
trainer = Trainer(
    model=lora_model,
    train_dataset=train_ds,
    args=TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=3,
        max_steps=10,
        learning_rate=2e-4, 
        # fp16=True, # only works on cuda
        logging_steps=1, 
        output_dir='outputs'
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
lora_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

print('begin train')
trainer.train()
print('done train')

lora_checkpoint = "/tmp/outputs/opt-350m-lora"
lora_model.save_pretrained(lora_checkpoint)
print('Save', lora_checkpoint)

print('***************** after lora finetune *********************')
from peft import PeftModel, PeftConfig
config = PeftConfig.from_pretrained(lora_checkpoint)
# print(config)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
lora_model = PeftModel.from_pretrained(model, lora_checkpoint)
batch = tokenizer(prompt, return_tensors='pt')
output_tokens = lora_model.generate(**batch, max_new_tokens=50)
print('prompt:', prompt)
print('result:', tokenizer.decode(output_tokens[0], skip_special_tokens=True))

