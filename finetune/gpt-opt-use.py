
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import pipeline

import torch
device = torch.device("cpu")

checkpoint = "bigscience/mt0-large"
checkpoint = "/Users/hhwang/models/gpt2"
# checkpoint = "/Users/hhwang/models/opt-125m"
# checkpoint = "/Users/hhwang/models/opt-350m"

# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
# inputs = tokenizer.encode("Write a short story", return_tensors="pt")
# outputs = model.generate(inputs)
# print(tokenizer.decode(outputs[0]))

# case 1
# pipe = pipeline(task='text-generation', model=checkpoint)
# print(pipe)
# result = pipe("tell me a joke")
# print('result: ',result)

# case 2
print('********* case 2 ***********')
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
model = GPT2LMHeadModel.from_pretrained(checkpoint)
text = "Replace me by any text you'd like."
encoded_input = tokenizer.encode(text, return_tensors='pt')
outputs = model.generate(encoded_input, max_length=50, num_return_sequences=1)
print('outputs:', outputs)
print(outputs.shape)
generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for i, generated_text in enumerate(generated_texts):
    print(f"Generated text {i + 1}: {generated_text}")

# case 3
print('********* case 3 ***********')
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
model = GPT2Model.from_pretrained(checkpoint)
print('config', model.config)
# print('model', model)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
outputs = model(**encoded_input)
# print(outputs)
last_hidden_states = outputs.last_hidden_state
print('last_hidden_states', last_hidden_states)
print(last_hidden_states.shape)
print(len(last_hidden_states[0][0]))
import torch.nn as nn
lm_head = nn.Linear(model.config.n_embd, model.config.vocab_size, bias=False)
lm_logits = lm_head(last_hidden_states)
print('lm_logits', lm_logits)
print(lm_logits.shape)

# case 4
# print('********* case 4 ***********')
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModel.from_pretrained(checkpoint)
# encoded_input = tokenizer.encode("Write a short story", return_tensors="pt")
# model = model.eval()
# print('config', model.config)
# print('model', model)
# print('inputs', encoded_input)
# outputs = model(encoded_input)
# print(outputs)

# case 5
# print('********* case 5 ***********')
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForCausalLM.from_pretrained(checkpoint)
# inputs = tokenizer.encode("Write a short story", return_tensors="pt")
# outputs = model.generate(inputs)
# print('result: ',tokenizer.batch_decode(outputs))

# case 6
# print('********* case 6 ***********')
# from transformers import GPT2Tokenizer, OPTForCausalLM
# model = OPTForCausalLM.from_pretrained(checkpoint)
# tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
# prompt = "Anti Vaccine Movemenet"
# inputs = tokenizer(prompt, return_tensors="pt").input_ids

# gen_tokens = model.generate(inputs,do_sample=True,temperature=0.9,max_length=100)
# gen_text = tokenizer.batch_decode(gen_tokens)[0]
# print('gen_text', gen_text)
# generate_ids = model.generate(inputs,max_length=2000,early_stopping= True,do_sample=True,min_length=2000,top_k=125,top_p=0.92,temperature= 0.85,repetition_penalty=1.5,num_return_sequences=3)
# for i, sample_output in enumerate(generate_ids):
#     result = tokenizer.decode(sample_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#     print(result) 

# case 7
# print('********* case 7 ***********')
# generator = pipeline('text-generation', model=checkpoint, device="cpu")
# text_inputs = ["tell me joke", "How do you", "Would you help", "I like apple", "This is something"]
# sent_gen = generator(text_inputs, max_length=50, num_return_sequences=2, repetition_penalty=1.3, top_k = 20) 
# #返回的sent_gen 形如#[[{'generated_text':"..."},{}],[{},{}]]
# for i in sent_gen:
#     print(i)

# case 8
# print('********* case 8 ***********')
# from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextGenerationPipeline
# tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
# model = GPT2LMHeadModel.from_pretrained(checkpoint)
# text_generator = TextGenerationPipeline(model, tokenizer, batch_size=3, device="cpu")
# text_generator.tokenizer.pad_token_id = model.config.eos_token_id
# text_inputs = ["tell me joke", "How do you", "Would you help", "I like apple", "This is something"]
# gen = text_generator(text_inputs, max_length=50, repetition_penalty=10.0, do_sample=True,  num_beams=5, top_k=10)
# for sent in gen:
#     gen_seq = sent[0]["generated_text"]
#     print("")
#     print(gen_seq)

# case 9
# print('********* case 9 ***********')
# from transformers import AutoTokenizer, AutoModelWithLMHead
# import torch
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelWithLMHead.from_pretrained(checkpoint)
# config=model.config
# # print('config', config)
# print(model)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = model.to(device)
# texts = ["tell me joke", "How do you", "Would you help", "I like apple", "This is something"]
# #用batch输入的时候一定要设置padding
# tokenizer.pad_token = tokenizer.eos_token
# encoding = tokenizer(texts, return_tensors='pt', padding=True).to(device)
# with torch.no_grad():
#     generated_ids = model.generate(**encoding, max_length=50, do_sample=True, top_k=20, repetition_penalty=3.0) 
# generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
# for l in generated_texts:
#     print(l)