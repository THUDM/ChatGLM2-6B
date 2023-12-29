
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import pipeline

checkpoint = "bigscience/mt0-large"
checkpoint = "/Users/hhwang/models/gpt2"
checkpoint = "/Users/hhwang/models/opt-125m"
checkpoint = "/Users/hhwang/models/opt-350m"

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
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
# model = GPT2LMHeadModel.from_pretrained(checkpoint)
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer.encode(text, return_tensors='pt')
# outputs = model.generate(encoded_input, max_length=50, num_return_sequences=1)
# generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# for i, generated_text in enumerate(generated_texts):
#     print(f"Generated text {i + 1}: {generated_text}")

# # case 3
# from transformers import GPT2Tokenizer, GPT2Model
# tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
# model = GPT2Model.from_pretrained(checkpoint)
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# outputs = model(**encoded_input)
# print(outputs)
# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states)

# case 4
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModel.from_pretrained(checkpoint)
# inputs = tokenizer.encode("Write a short story", return_tensors="pt")
# model = model.eval()
# print(inputs)
# outputs = model(inputs)
# print(outputs)

# case 5
print('********* case 5 ***********')
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
inputs = tokenizer.encode("Write a short story", return_tensors="pt")
outputs = model.generate(inputs)
print('result: ',tokenizer.batch_decode(outputs))

# case 6
print('********* case 6 ***********')
from transformers import GPT2Tokenizer, OPTForCausalLM
model = OPTForCausalLM.from_pretrained(checkpoint)
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
prompt = "Anti Vaccine Movemenet"
inputs = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(inputs,do_sample=True,temperature=0.9,max_length=100)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print('gen_text', gen_text)
# generate_ids = model.generate(inputs,max_length=2000,early_stopping= True,do_sample=True,min_length=2000,top_k=125,top_p=0.92,temperature= 0.85,repetition_penalty=1.5,num_return_sequences=3)
# for i, sample_output in enumerate(generate_ids):
#     result = tokenizer.decode(sample_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#     print(result) 
