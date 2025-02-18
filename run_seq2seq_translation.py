#!/usr/bin/env python
# coding: utf-8
# import the libraries
import os
import numpy as np
import torch
import pandas as pd
import csv
from dataclasses import dataclass
from datasets import Dataset, DatasetDict
import argparse
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,  NllbTokenizer)
# use argparse to let the user provides values for variables at runtime
def DataTestingArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', 
        type=str, required=True, help='Load a fine-tuned model checkpoint for translation')
    parser.add_argument('--data_dir', 
        type=str, required=True, help='Directory of the dataset file')
    parser.add_argument('--file_name', 
        type=str, required=True, help='Name file to be translated')
    parser.add_argument('--lang_target', 
        type=str, required=True, help='Target language of translation')
    args = parser.parse_args()
    return args

#create the configuration class
@dataclass
class Config:
    batch_size: int = 4
    max_source_length: int = 400 # the maximum length in number of tokens for tokenizing the input sentence
    max_target_length: int = 400 # the maximum length in number of tokens for tokenizing the input sentence
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    data_args=DataTestingArguments() #call the arguments
    config = Config()

    # Load monolingual dataset
    dataset_test = pd.read_csv(os.path.join(data_args.data_dir, f'{data_args.file_name}.csv'))

    # Replace unnecessary characters
    dataset_test['english'] = dataset_test['english'].str.replace("'", "’", regex=False)

    # load the fine-tuned translation model
    model_name = data_args.model_name_or_path 
   
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    def fix_tokenizer(tokenizer, new_lang):
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
        return tokenizer
    
    # Load the tokenizer from pre-trained model to perform translation
    if 'nllb' in  model_name:
        tokenizer = NllbTokenizer.from_pretrained(model_name)
        if len(tokenizer) != tokenizer.vocab_size: #Check whether the values between len(tokenizer) and tokenizer.vocab_size are same after we added Truku language tag
        # This is only performed when we already expanded the tokenizer of the NLLB model
            eng_lang = 'eng_Latn'
            lad_lang = 'lad_Latn'
            print("fix the tokenizer configuration")
            tokenizer = fix_tokenizer(tokenizer, new_lang=lad_lang)
            print('ids of ladin is', tokenizer.convert_tokens_to_ids('lad_Latn'))
    else: 
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("number of parameters:", model.num_parameters())
    
    # perform the testing process
    # Tokenizing the batcb
    def batch_tokenize_fn(examples, source_lang, target_lang):
        """
        Generate the input_ids and labels field for dataset dict of training data.
        """
        if source_lang in ['eng_Latn']:
            sources = examples["english"]
        else:
            sources = examples["ladin"]

        # tokenizing the input sentences
        tokenizer.src_lang = source_lang
        model_inputs = tokenizer(sources, max_length=config.max_source_length, truncation=True)
    
        # tokenizing the target sentences
        tokenizer.src_lang = target_lang
        bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
        if bos_token_id is None:
            raise ValueError(f"The target language '{target_lang}' is not in the tokenizer vocabulary.")

        # set the language target tag 
        model_inputs["forced_bos_token_id"] = [tokenizer.convert_tokens_to_ids(target_lang)] * len(sources)

        # Give the dummy label of translation
        labels = tokenizer(sources, max_length=config.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    
    # Create a Wrapper Function
    def batch_tokenize_fn_wrapper(examples, source_lang, target_lang):
        return batch_tokenize_fn(examples, source_lang, target_lang)
    
    # Set translation target language
    if data_args.lang_target == 'ladin':
        source_lang=eng_lang
        target_lang=lad_lang
    else:
        source_lang=lad_lang
        target_lang=eng_lang
    print(f'+++----- Starting translation from {source_lang} to {target_lang} -----+++')
    translations_and_metrics = []

    for batch_start in range(0, len(dataset_test), config.batch_size):
        # Get the batch of test data
        test_batch = dataset_test.iloc[batch_start:batch_start + config.batch_size]

        # Convert from pandas into Dataset
        tds = Dataset.from_pandas(test_batch)
        dataset_dict = DatasetDict()
        dataset_dict['test'] = tds
        
        print('Testing at batch: ', (batch_start/config.batch_size))
        #tokenizing the input and target sentences   
        dataset_dict_tokenized = dataset_dict.map(
            lambda examples: batch_tokenize_fn_wrapper(examples, source_lang=source_lang, target_lang=target_lang),
            batched=True,
            remove_columns=dataset_dict["test"].column_names
            )
        
        # Generate predictions  
        device = config.device
        model = model.to(device)
        input_ids = dataset_dict_tokenized["test"]["input_ids"]
        max_length = config.max_source_length # Get the maximum length

        # Pad the sequences to the maximum length
        padded_input_ids = [seq + [tokenizer.pad_token_id] * (max_length - len(seq)) for seq in input_ids]

        # Convert the padded sequences to a tensor
        input_ids_tensor = torch.tensor(padded_input_ids).to(device)
        predictions = model.generate(
            input_ids = input_ids_tensor,
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang),
            )
        tokenizer.src_lang = target_lang
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                  
        # Append the translation result into Dataframe
        for i, prediction in enumerate(predictions):
            # Tokenize the prediction
            translations_and_metrics.append({
                source_lang : test_batch.english.iloc[i],
                target_lang: prediction,
            })

    # Save results to CSV
    output_file = (f'{data_args.data_dir}/{data_args.file_name}_to_{data_args.lang_target}.csv')
    df = pd.DataFrame(translations_and_metrics)
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Results saved to {output_file}")
    
if __name__ == "__main__":
    main()
