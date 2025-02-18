#!/usr/bin/env python
# coding: utf-8

# import the libraries
import os
import torch
import random
import evaluate
import numpy as np
import pandas as pd
from dataclasses import dataclass
from time import perf_counter
from datasets import concatenate_datasets, Dataset, DatasetDict
import argparse
from sklearn.utils import shuffle
from transformers import (
    AutoTokenizer,
    MBart50Tokenizer,
    NllbTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    MBartForConditionalGeneration
)
# use argparse to let the user provides values for variables at runtime

def DataTrainingArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', 
        type=str, required=True, help='Load a LLM as model checkpoint for translation')
    parser.add_argument('--cache_dir', 
        type=str, required=True, help='Directory for saving the pre-trained translation model')
    parser.add_argument('--file_name', 
        type=str, required=True, help='File containing parallel sentences for training data')
    parser.add_argument('--data_dir', 
        type=str, required=True, help='Directory of the dataset files')
    args = parser.parse_args()
    return args

#create the configuration class
@dataclass
class Config:
    lang: str= "Eng2Lad" # Here We set English-Ladin as translation system 
    batch_size: int = 8
    seed: int = 42
    max_source_length: int = 256 # the maximum length in number of tokens for tokenizing the input sentence
    max_target_length: int = 256 # the maximum length in number of tokens for tokenizing the target sentence

    lr: float = 0.0001  #learning rate
    weight_decay: float = 0.01 # Weight decay (L2 penalty) is eight regularization to reduce the overfitting
    epochs: int = 5
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set random seed to ensure that results are reproducible
    def __post_init__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

def main():
    data_train_args=DataTrainingArguments() #call the arguments
    config = Config()
    #Load the training dataset from tsv files
    dataset = pd.read_csv(data_train_args.data_dir+ f'/{data_train_args.file_name}.csv')
    dataset = shuffle(dataset, random_state=42)

    # Divide the data into training and validation set
    tds = Dataset.from_pandas(dataset[:round(len(dataset)*0.85)])
    vds = Dataset.from_pandas(dataset[round(len(dataset)*0.85):])
    
    #Create dataset dict for training and validation set
    dataset_dict = DatasetDict()
    dataset_dict['train'] = tds
    dataset_dict['val'] = vds

    print(dataset_dict)
    #Load the evaluation metrics
    bleu_score = evaluate.load("bleu")
        
    # Load the initial model checkpoint from the pre-trained model to perform fine-tuning translation
    model_name = data_train_args.model_checkpoint.split("/")[-1] #the name of pre-trained model
    # The directory of the fine-tuned translation model
    fine_tuned_model_checkpoint = os.path.join(
        data_train_args.cache_dir,
        f"{model_name}_{config.lang}"
    )
    
    # Load the Model
    if 'mbart' in data_train_args.model_checkpoint: # If the set model is MBART
        do_train = True
        model = MBartForConditionalGeneration.from_pretrained(data_train_args.model_checkpoint, cache_dir=data_train_args.cache_dir)

    else: # If the set model is not MBART
        do_train = True
        model = AutoModelForSeq2SeqLM.from_pretrained(data_train_args.model_checkpoint, cache_dir=data_train_args.cache_dir)
    
    # Fixing the tokenizer by adding Ladin language
    def fix_tokenizer(model, tokenizer, new_lang, sim_lang):
        """ Add a new language token to the tokenizer vocabulary (this should be done each time after its initialization) """
        old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
        tokenizer.lang_code_to_id[new_lang] = old_len-1
        tokenizer.id_to_lang_code[old_len-1] = new_lang
        # always move "mask" to the last position
        tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset

        tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
        tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
        if new_lang not in tokenizer._additional_special_tokens:
            tokenizer._additional_special_tokens.append(new_lang)
        # clear the added token encoder; otherwise a new token may end up there by mistake
        tokenizer.added_tokens_encoder = {}
        tokenizer.added_tokens_decoder = {}

        # patching 'lad_Latn' 
        model.resize_token_embeddings(len(tokenizer))
        # fixing the new/moved token embeddings in the model
        added_token_id = tokenizer.convert_tokens_to_ids(new_lang)
        similar_lang_id = tokenizer.convert_tokens_to_ids(sim_lang)

        # set the ladin embedding same with friulian embedding since both are in same family
        # moving the embedding for "mask" to its new position
        model.model.shared.weight.data[added_token_id+1] = model.model.shared.weight.data[added_token_id]
        # initializing new language token with a token of a similar language
        model.model.shared.weight.data[added_token_id] = model.model.shared.weight.data[similar_lang_id]
        return model, tokenizer
    
    # Load the tokenizer from pre-trained model to perform fine-tuning translation
    # HERE we set Ladin
    if 'nllb' in  data_train_args.model_checkpoint: # If the set model is NLLB
        eng_lang = 'eng_Latn'
        lad_lang = 'lad_Latn'
        # YOU SHOULD SET WHICH LANGUAGE THAT SIMILAR TO THE UNSEEN (LANUGUAGE) TO GET ITS EMBEDDINGS
        # Since our new language is Ladin, we set Friulian as the similar language of ladin
        sim_lang = 'fur_Latn'
        print("fix the tokenizer configuration")
        tokenizer = NllbTokenizer.from_pretrained(data_train_args.model_checkpoint)
        model, tokenizer = fix_tokenizer(model, tokenizer,  new_lang=lad_lang, sim_lang=sim_lang)
        print('Ids of ladin is', tokenizer.convert_tokens_to_ids('lad_Latn'))
        

    elif 'mbart' in data_train_args.model_checkpoint: # If the set model is MBART
        eng_lang = 'en_EN'
        lad_lang = 'ld_LD'
        print("fix the tokenizer configuration")
        tokenizer = MBart50Tokenizer.from_pretrained(data_train_args.model_checkpoint)
        model, tokenizer = fix_tokenizer(model, tokenizer, new_lang=lad_lang, sim_lang=eng_lang)
        print('Ids of ladin is', tokenizer.convert_tokens_to_ids('ld_LD'))

        tokenizer.init_kwargs["src_lang"]= "en_EN"
        tokenizer.init_kwargs["tgt_lang"] = "ld_LD"
        print(len(tokenizer))
        print(tokenizer.vocab_size)
        print(len(model.model.shared.weight.data))
        print('Ids of ladin is', tokenizer.convert_tokens_to_ids('ld_LD'))

    else: 
        tokenizer = AutoTokenizer.from_pretrained(data_train_args.model_checkpoint)
    
    print("number of parameters:", model.num_parameters())
    
    def batch_tokenize_fn(examples, source_lang, target_lang):
        """
        Generate the input_ids and labels field for dataset dict of training data.
        """
        if source_lang in ['eng_Latn', 'en_EN']:
            sources = examples["english"]
            targets = examples["ladin"]
        else:
            sources = examples["ladin"]
            targets = examples["english"]
        # tokenizing the input sentences

        tokenizer.src_lang = source_lang
        model_inputs = tokenizer(sources, max_length=config.max_source_length, truncation=True)
    
        # tokenizing the target sentences
        # tokenized ids of the target are stored as the labels field
        tokenizer.src_lang = target_lang
        labels = tokenizer(targets, max_length=config.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        # set the language target tag 
        model_inputs["forced_bos_token_id"] = [tokenizer.convert_tokens_to_ids(target_lang)] * len(sources)
  
        return model_inputs
    
    # Create a Wrapper Function
    def batch_tokenize_fn_wrapper(examples, source_lang, target_lang):
        return batch_tokenize_fn(examples, source_lang, target_lang)
  
    # Tokenizing sentence pair of english and Ladin dataset
    i2l_dataset_dict_tokenized = dataset_dict.map(
        lambda examples: batch_tokenize_fn_wrapper(examples, source_lang=eng_lang, target_lang=lad_lang),
        batched=True,
        remove_columns=dataset_dict["train"].column_names
        )
    
    l2i_dataset_dict_tokenized = dataset_dict.map(
        lambda examples: batch_tokenize_fn_wrapper(examples, source_lang=lad_lang, target_lang=eng_lang),
        batched=True,
        remove_columns=dataset_dict["train"].column_names
        )
    
    #concatenate all tokenized datasets
    l2i=l2i_dataset_dict_tokenized
    i2l=i2l_dataset_dict_tokenized
    all_data=DatasetDict({"train": concatenate_datasets([l2i["train"], i2l["train"]]),"val": concatenate_datasets([l2i["val"], i2l["val"]])})
    all_data_tokenized=all_data.shuffle(seed=42) #shuffle all training dataset
    # set outpur dir
    output_dir = os.path.join(data_train_args.cache_dir, f"{model_name}_{config.lang}") # where the pre-trained translation model is saved
    
    #The training arguments for the training session
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch", # the evaluation strategy to adopt during training.
        save_strategy="epoch",
        logging_steps=500,  # Log every 10 training steps
        eval_steps=1000,     # Evaluate every 10 steps
        learning_rate=config.lr,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size= int(config.batch_size/4),
        #generation_max_length=256,
        weight_decay=config.weight_decay, # Weight decay (L2 penalty) is the weight of regularization to reduce the overfitting
        save_total_limit=2, # limit the total amount of saved checkpoints to 2 directories
        num_train_epochs=config.epochs,
        predict_with_generate=True, # use model.generate()  to calculate generative metrics
        load_best_model_at_end=True, # save the best model when finished training
        greater_is_better=True, #lower score better result of the main metric
        metric_for_best_model="bleu", # the main metric in the training process
        gradient_accumulation_steps=8, # Number of update steps to accumulate the gradients for, before performing a backward/update pass
        do_train=do_train,
        generation_num_beams=5,     # Number of beams for beam search
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
    )
    # evalution metrics computation
    def compute_metrics(eval_pred):
        """
        Compute bleu metric for seq2seq model generated prediction.
        
        tip: we can run trainer.predict on our eval dataset to see what a sample
        eval_pred object would look like when implementing custom compute metrics function
        """
        predictions, labels = eval_pred
        
        # Clip predictions to ensure token IDs are within the valid range
        max_token_id = tokenizer.vocab_size - 1
        predictions = np.clip(predictions, 0, max_token_id)
        labels = np.clip(labels, 0, max_token_id)
        
        # Decode prediction samples, which is in ids into text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode tokenized labels a.k.a. reference translation into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score = bleu_score.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )
        result = {}
        result["bleu"] = score["bleu"] #The higher the value, the better the translations
        return {k: round(v, 4) for k, v in result.items()}
        
    # Data collator used for seq2seq model
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=all_data_tokenized["train"],
        eval_dataset=all_data_tokenized["val"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(1, 0.0)] #early_stopping_patience =1, early_stopping_threshold =0
    )
    
    # perform the training process
    if trainer.args.do_train:
        t1_start = perf_counter()
        trainer.train()
        t1_stop = perf_counter()
        print("Training elapsed time:", t1_stop - t1_start)
    
        # saving the pre-trained model
        trainer.save_model(fine_tuned_model_checkpoint)
        # Save tokenizer
        tokenizer.save_pretrained(fine_tuned_model_checkpoint)
    print('Training is finished !!!')
    evaluation=trainer.evaluate()
    print(evaluation)
if __name__ == "__main__":
    main()
