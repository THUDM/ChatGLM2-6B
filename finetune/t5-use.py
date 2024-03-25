from transformers import  AutoTokenizer, AutoModelForSeq2SeqLM

import torch
device = torch.device("cpu")

checkpoint = "/Users/hhwang/models/t5-small"
checkpoint = "/Users/hhwang/models/flan-t5-small"

print('********* case 1 ***********')
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
# # print(model.config)
# inputs = tokenizer.encode("translate English to German: That is good", return_tensors="pt")
# outputs = model.generate(inputs, max_new_tokens=20)
# print('result: ',tokenizer.batch_decode(outputs))

print('********* case 2 ***********')

from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
prompt = "translate English to German: That is good?"
generator = pipeline("summarization", model=model, tokenizer=tokenizer)
print(generator(prompt))
