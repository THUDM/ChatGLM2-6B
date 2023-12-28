#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence for P-Tuning v2
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

# CUDA_VISIBLE_DEVICES=-1 python finetune-p-tuning-v2.py

# accelerate launch --cpu --num_machines=1 --num_processes=1 --num_cpu_threads_per_process=1 finetune-p-tuning-v2.py
# accelerate launch --cpu --num_machines=1 --num_processes=4 --num_cpu_threads_per_process=1 finetune-p-tuning-v2.py

# import logging
import os
import sys
import json

import numpy as np
from datasets import load_dataset
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

# import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    # HfArgumentParser,
    # Seq2SeqTrainingArguments,
    # set_seed,
)

# from typing import Any, Dict, List, Optional, Tuple, Union

# import torch
# from torch import nn
# from torch.utils.data import Dataset

# from transformers.deepspeed import is_deepspeed_zero3_enabled
# from trainer import PrefixTrainer
# from transformers.trainer_utils import PredictionOutput
# from transformers.utils import logging

# import os
# from typing import Optional
from transformers import Trainer

# import torch
# from transformers.modeling_utils import PreTrainedModel, unwrap_model
# from transformers.utils import logging

# from trainer_seq2seq import Seq2SeqTrainer

# from arguments import ModelArguments, DataTrainingArguments

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

def main():
    # print(torch.backends.mps.is_available())
    # print(torch.backends.mps.is_built())
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("device:", device)
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # else:
    #     model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    # )

    # if training_args.should_log:
    #     # The default of training_args.log_level is passive, so we set log level at info here to have that default.
    #     transformers.utils.logging.set_verbosity_info()

    # log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    # logger.warning(
    #     f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    #     + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    # )
    # logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    # set_seed(training_args.seed)

    # Load dataset
    data_files_path = "/tmp/hub/dataset/shibing624/AdvertiseGen"
    print('data_files_path:', data_files_path)
    data_files = {}
    data_files["train"] = "train.json"
    data_files["validation"] = "dev.json"
    # data_files["test"] = "test.json"
    print('data_files:', data_files)
    raw_datasets = load_dataset(data_files_path, data_files=data_files)
    print("raw_datasets:", raw_datasets)
    
    # Load pretrained model and tokenizer
    model_name_or_path = '/tmp/hub/chatglm2-6b'
    print('model_name_or_path', model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    print('load autoconfig done')
    # soft prompt 长度
    PRE_SEQ_LEN=128 
    config.pre_seq_len = PRE_SEQ_LEN
    config.prefix_projection = False

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    print('load AutoTokenizer done')

    # if model_args.ptuning_checkpoint is not None:
    #     # Evaluation
    #     # Loading extra state dict of prefix encoder
    #     model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
    #     prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
    #     new_prefix_state_dict = {}
    #     for k, v in prefix_state_dict.items():
    #         if k.startswith("transformer.prefix_encoder."):
    #             new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    #     model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    # else:
    #     model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    model = AutoModel.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
    #print('load AutoModel done')
    # model = model.quantize(4)
    
    # if model_args.quantization_bit is not None:
    #     print(f"Quantized to {model_args.quantization_bit} bit")
    #     model = model.quantize(model_args.quantization_bit)
    # if model_args.pre_seq_len is not None:
    #     # P-tuning v2
    #     model = model.half()
    #     model.transformer.prefix_encoder.float()
    # else:
    #     # Finetune
    #     model = model.float()
    
    # P-tuning v2, do not work for accelerate
    model = model.half()
    model.transformer.prefix_encoder.float()
    
    # finetune, work for accelerate
    # model = model.float()
    
    print('model half done')

    prefix = ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    # if training_args.do_train:
    #     column_names = raw_datasets["train"].column_names
    # elif training_args.do_eval:
    #     column_names = raw_datasets["validation"].column_names
    # elif training_args.do_predict:
    #     column_names = raw_datasets["test"].column_names
    # else:
    #     logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
    #     return
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    prompt_column = 'content'
    response_column = 'summary'
    history_column = None
    
    # Temporarily set max_target_length for training.
    max_source_length = 64
    max_target_length = 128
    ignore_pad_token_for_loss = True
    
    def preprocess_function_eval(examples):
        inputs, targets = [], []
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query = examples[prompt_column][i]
                history = examples[history_column][i] if history_column is not None else None
                prompt = tokenizer.build_prompt(query, history)
                inputs.append(prompt)
                targets.append(examples[response_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True, padding=True)
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
        
        if ignore_pad_token_for_loss:
            labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_function_train(examples):
        max_seq_length = max_source_length + max_target_length + 1
        model_inputs = { "input_ids": [], "labels": [] }
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]

                history = examples[history_column][i] if history_column is not None else None
                prompt = tokenizer.build_prompt(query, history)

                prompt = prefix + prompt
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True, max_length=max_source_length)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True, max_length=max_target_length)

                context_length = len(a_ids)
                input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
                labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]
                
                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                if ignore_pad_token_for_loss:
                    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs
    
    def print_dataset_example(example):
        print('*******************  print_dataset_example ******************************')
        print("input_ids:", example["input_ids"])
        print("inputs:", tokenizer.decode(example["input_ids"]))
        print("label_ids:", example["labels"])
        print("labels:", tokenizer.decode(example["labels"]))

    max_train_samples = 5
    do_train = True
    if do_train:
        train_dataset = raw_datasets["train"]
        max_train_samples = min(len(train_dataset), max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function_train,
            batched=True,
            num_proc=5,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
        )
        # print_dataset_example(train_dataset[0])

    max_eval_samples = 5
    do_eval = True
    if do_eval:
        eval_dataset = raw_datasets["validation"]
        max_eval_samples = min(len(eval_dataset), max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function_eval,
            batched=True,
            num_proc=5,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on validation dataset",
        )
        # print_dataset_example(eval_dataset[0])

    # if training_args.do_predict:
    #     max_target_length = data_args.val_max_target_length
    #     if "test" not in raw_datasets:
    #         raise ValueError("--do_predict requires a test dataset")
    #     predict_dataset = raw_datasets["test"]
    #     if data_args.max_predict_samples is not None:
    #         max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
    #         predict_dataset = predict_dataset.select(range(max_predict_samples))
    #     with training_args.main_process_first(desc="prediction dataset map pre-processing"):
    #         predict_dataset = predict_dataset.map(
    #             preprocess_function_eval,
    #             batched=True,
    #             num_proc=data_args.preprocessing_num_workers,
    #             remove_columns=column_names,
    #             load_from_cache_file=not data_args.overwrite_cache,
    #             desc="Running tokenizer on prediction dataset",
    #         )
    #     print_dataset_example(predict_dataset[0])

    # Data collator
    label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )
    print("data_collator done")
    
    # Metric
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
            result = scores[0]
            
            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    # Override the decoding parameters of Seq2SeqTrainer
    # training_args.generation_max_length = (
    #     training_args.generation_max_length
    #     if training_args.generation_max_length is not None
    #     else data_args.val_max_target_length
    # )
    # training_args.generation_num_beams = (
    #     data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    # )
    
    # Initialize our Trainer
    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     # args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    #     save_changed=PRE_SEQ_LEN is not None
    # )
    
    trainer = Trainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print('build trainer done')
    
    # Training
    if do_train:
        # checkpoint = False
        # if training_args.resume_from_checkpoint is not None:
        #     checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print("begin trainning")
        # train_result = trainer.train(resume_from_checkpoint=checkpoint)
        train_result = trainer.train()
        # trainer.save_model()  # Saves the tokenizer too for easy upload
        print("done trainning")
        metrics = train_result.metrics
        max_train_samples = len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print("save state")
        # trainer.save_model("tmp_trainer/ptuning")
        print("save model")

    # # Evaluation
    # results = {}
    # max_seq_length = data_args.max_source_length + data_args.max_target_length + 1
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7, max_length=max_seq_length, temperature=0.95)
    #     max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #     metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # if training_args.do_predict:
    #     logger.info("*** Predict ***")
    #     predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_length=max_seq_length, do_sample=True, top_p=0.7, temperature=0.95)
    #     metrics = predict_results.metrics
    #     max_predict_samples = (
    #         data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    #     )
    #     metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    #     trainer.log_metrics("predict", metrics)
    #     trainer.save_metrics("predict", metrics)

    #     if trainer.is_world_process_zero():
    #         if training_args.predict_with_generate:
    #             predictions = tokenizer.batch_decode(
    #                 predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #             )
    #             predictions = [pred.strip() for pred in predictions]
    #             labels = tokenizer.batch_decode(
    #                 predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #             )
    #             labels = [label.strip() for label in labels]
    #             output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
    #             with open(output_prediction_file, "w", encoding="utf-8") as writer:
    #                 for p, l in zip(predictions, labels):
    #                     res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
    #                     writer.write(f"{res}\n")
    # return results

# WEIGHTS_NAME = "pytorch_model.bin"
# TRAINING_ARGS_NAME = "training_args.bin"

# class PrefixTrainer(Trainer):
#     def __init__(self, *args, save_changed=False, **kwargs):
#         self.save_changed = save_changed
#         super().__init__(*args, **kwargs)

#     def _save(self, output_dir: Optional[str] = None, state_dict=None):
#         # If we are executing this function, we are the process zero, so we don't check for that.
#         output_dir = output_dir if output_dir is not None else self.args.output_dir
#         os.makedirs(output_dir, exist_ok=True)
#         logger.info(f"Saving model checkpoint to {output_dir}")
#         # Save a trained model and configuration using `save_pretrained()`.
#         # They can then be reloaded using `from_pretrained()`
#         if not isinstance(self.model, PreTrainedModel):
#             if isinstance(unwrap_model(self.model), PreTrainedModel):
#                 if state_dict is None:
#                     state_dict = self.model.state_dict()
#                 unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
#             else:
#                 logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
#                 if state_dict is None:
#                     state_dict = self.model.state_dict()
#                 torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
#         else:
#             if self.save_changed:
#                 print("Saving PrefixEncoder")
#                 state_dict = self.model.state_dict()
#                 filtered_state_dict = {}
#                 for k, v in self.model.named_parameters():
#                     if v.requires_grad:
#                         filtered_state_dict[k] = state_dict[k]
#                 self.model.save_pretrained(output_dir, state_dict=filtered_state_dict)
#             else:
#                 print("Saving the whole model")
#                 self.model.save_pretrained(output_dir, state_dict=state_dict)
#         if self.tokenizer is not None:
#             self.tokenizer.save_pretrained(output_dir)

#         # Good practice: save your training arguments together with the trained model
#         torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

# class Seq2SeqTrainer(PrefixTrainer):
#     def evaluate(
#         self,
#         eval_dataset: Optional[Dataset] = None,
#         ignore_keys: Optional[List[str]] = None,
#         metric_key_prefix: str = "eval",
#         **gen_kwargs
#     ) -> Dict[str, float]:
#         """
#         Run evaluation and returns metrics.

#         The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
#         (pass it to the init `compute_metrics` argument).

#         You can also subclass and override this method to inject custom behavior.

#         Args:
#             eval_dataset (`Dataset`, *optional*):
#                 Pass a dataset if you wish to override `self.eval_dataset`. If it is an [`~datasets.Dataset`], columns
#                 not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
#                 method.
#             ignore_keys (`List[str]`, *optional*):
#                 A list of keys in the output of your model (if it is a dictionary) that should be ignored when
#                 gathering predictions.
#             metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
#                 An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
#                 "eval_bleu" if the prefix is `"eval"` (default)
#             max_length (`int`, *optional*):
#                 The maximum target length to use when predicting with the generate method.
#             num_beams (`int`, *optional*):
#                 Number of beams for beam search that will be used when predicting with the generate method. 1 means no
#                 beam search.
#             gen_kwargs:
#                 Additional `generate` specific kwargs.

#         Returns:
#             A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
#             dictionary also contains the epoch number which comes from the training state.
#         """

#         gen_kwargs = gen_kwargs.copy()
#         if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
#             gen_kwargs["max_length"] = self.args.generation_max_length
#         gen_kwargs["num_beams"] = (
#             gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
#         )
#         self._gen_kwargs = gen_kwargs

#         return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

#     def predict(
#         self,
#         test_dataset: Dataset,
#         ignore_keys: Optional[List[str]] = None,
#         metric_key_prefix: str = "test",
#         **gen_kwargs
#     ) -> PredictionOutput:
#         """
#         Run prediction and returns predictions and potential metrics.

#         Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
#         will also return metrics, like in `evaluate()`.

#         Args:
#             test_dataset (`Dataset`):
#                 Dataset to run the predictions on. If it is a [`~datasets.Dataset`], columns not accepted by the
#                 `model.forward()` method are automatically removed. Has to implement the method `__len__`
#             ignore_keys (`List[str]`, *optional*):
#                 A list of keys in the output of your model (if it is a dictionary) that should be ignored when
#                 gathering predictions.
#             metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
#                 An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
#                 "eval_bleu" if the prefix is `"eval"` (default)
#             max_length (`int`, *optional*):
#                 The maximum target length to use when predicting with the generate method.
#             num_beams (`int`, *optional*):
#                 Number of beams for beam search that will be used when predicting with the generate method. 1 means no
#                 beam search.
#             gen_kwargs:
#                 Additional `generate` specific kwargs.

#         <Tip>

#         If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
#         padding in a token classification task) the predictions will be padded (on the right) to allow for
#         concatenation into one array. The padding index is -100.

#         </Tip>

#         Returns: *NamedTuple* A namedtuple with the following keys:

#             - predictions (`np.ndarray`): The predictions on `test_dataset`.
#             - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
#             - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
#               labels).
#         """

#         gen_kwargs = gen_kwargs.copy()
#         if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
#             gen_kwargs["max_length"] = self.args.generation_max_length
#         gen_kwargs["num_beams"] = (
#             gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
#         )
#         self._gen_kwargs = gen_kwargs


#         return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

#     def prediction_step(
#         self,
#         model: nn.Module,
#         inputs: Dict[str, Union[torch.Tensor, Any]],
#         prediction_loss_only: bool,
#         ignore_keys: Optional[List[str]] = None,
#     ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
#         """
#         Perform an evaluation step on `model` using `inputs`.

#         Subclass and override to inject custom behavior.

#         Args:
#             model (`nn.Module`):
#                 The model to evaluate.
#             inputs (`Dict[str, Union[torch.Tensor, Any]]`):
#                 The inputs and targets of the model.

#                 The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
#                 argument `labels`. Check your model's documentation for all accepted arguments.
#             prediction_loss_only (`bool`):
#                 Whether or not to return the loss only.

#         Return:
#             Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
#             labels (each being optional).
#         """

#         if not self.args.predict_with_generate or prediction_loss_only:
#             return super().prediction_step(
#                 model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
#             )

#         has_labels = "labels" in inputs
#         inputs = self._prepare_inputs(inputs)

#         # XXX: adapt synced_gpus for fairscale as well
#         gen_kwargs = self._gen_kwargs.copy()
#         if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
#             gen_kwargs["max_length"] = self.model.config.max_length
#         gen_kwargs["num_beams"] = (
#             gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
#         )
#         default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
#         gen_kwargs["synced_gpus"] = (
#             gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
#         )

#         if "attention_mask" in inputs:
#             gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
#         if "position_ids" in inputs:
#             gen_kwargs["position_ids"] = inputs.get("position_ids", None)
#         if "global_attention_mask" in inputs:
#             gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

#         # prepare generation inputs
#         # some encoder-decoder models can have varying encoder's and thus
#         # varying model input names
#         if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
#             generation_inputs = inputs[self.model.encoder.main_input_name]
#         else:
#             generation_inputs = inputs[self.model.main_input_name]

#         gen_kwargs["input_ids"] = generation_inputs
#         generated_tokens = self.model.generate(**gen_kwargs)
#         generated_tokens = generated_tokens[:, generation_inputs.size()[-1]:]

#         # in case the batch is shorter than max length, the output should be padded
#         if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
#             generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
#         elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[-1] < (
#             gen_kwargs["max_new_tokens"] + 1
#         ):
#             generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"] + 1)

#         loss = None

#         if self.args.prediction_loss_only:
#             return (loss, None, None)

#         if has_labels:
#             labels = inputs["labels"]
#             if gen_kwargs.get("max_length") is not None and labels.shape[-1] < gen_kwargs["max_length"]:
#                 labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
#             elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
#                 gen_kwargs["max_new_tokens"] + 1
#             ):
#                 labels = self._pad_tensors_to_max_len(labels, (gen_kwargs["max_new_tokens"] + 1))
#         else:
#             labels = None

#         return (loss, generated_tokens, labels)

#     def _pad_tensors_to_max_len(self, tensor, max_length):
#         if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
#             # If PAD token is not defined at least EOS token has to be defined
#             pad_token_id = (
#                 self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
#             )
#         else:
#             if self.model.config.pad_token_id is not None:
#                 pad_token_id = self.model.config.pad_token_id
#             else:
#                 raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

#         padded_tensor = pad_token_id * torch.ones(
#             (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
#         )
#         padded_tensor[:, : tensor.shape[-1]] = tensor
#         return padded_tensor


if __name__ == "__main__":
    main()
