#!/usr/bin/env python
# coding: utf-8
# import the libraries
import os
import pandas as pd
import json
import argparse
from fine_tuning import FTModel

# use argparse to let the user provides values for variables at runtime
def DataTestingArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', 
        type=str, required=True, help='Load a LLM as model for few-shot learning')
    parser.add_argument('--dataset_dir', 
        type=str, required=True, help='Directory of the dataset files')
    parser.add_argument('--test_data', 
        type=str, required=True, help='Name of the testing data file')
    parser.add_argument('--target_lang', 
        type=str, required=True, help='Target of language')
    parser.add_argument('--batch_size', 
        type=int, default=5, help='Number of request sentences per batch of target translation')
    parser.add_argument('--save_dir', 
        type=str, required=True, help='Directory for saving experimental results')
    args = parser.parse_args()
    return args

def main():
    #create the configuration class
    
    args=DataTestingArguments() #call the arguments
    
    ####---Load Dataset---####
    test_data = pd.read_csv(os.path.join(args.dataset_dir, f"{args.test_data}.csv"))

    # Set the source and target languages
    if args.target_lang == 'english':
        test_data['english'] = " "
        source_lang = 'ladin'
        from_lang = "ladin"
        to_lang =  "english"
        path_f = 'ladin2english'
    else:
        test_data['ladin'] = " "
        source_lang = 'english'
        from_lang = "english"
        to_lang =  "ladin"
        path_f = 'english2ladin'  
    
    # Set the LLMs
    ft_model = FTModel(args.model_name)
    index_batch = 0
    for batch_start in range(0, len(test_data), args.batch_size):

        # Get the batch of test data
        test_batch = test_data.iloc[batch_start:batch_start + args.batch_size]
        
        # Set the source language
        requested_translation = test_batch[source_lang].tolist()
        
        # Construct the prompt for the model
        prompt_1 = f"This is a translation system that translates {from_lang} into {to_lang}. Provide accurate translations by preserving the meaning.\n"
        prompt_2 = f"Translate the following {len(test_batch)} {from_lang} texts to {to_lang}:\n"
        
        # Generate translation using LLMs with API
        generated_translation = ft_model.generating(prompt_1, prompt_2, requested_translation)
        print(generated_translation)
        # If the response is successful
        #if generated_translation.status_code == 200:
        try:
            response_json = generated_translation.json()
            print(f"Response JSON for batch {index_batch} ")

                # Save the output into json file
            if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
            output_path = os.path.join(args.save_dir, path_f, f'translation_{args.model_name}_{args.test_data}_size of_{args.batch_size}_batch_{index_batch}.json')
                
                # Save the JSON response for this batch
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(response_json, json_file, ensure_ascii=False, indent=4)
                
            print(f"Saved batch {index_batch} to {output_path}")
        except (json.JSONDecodeError, KeyError) as e:
            print("Error parsing the translation output.")

        index_batch += 1    
if __name__ == "__main__":
    main()
