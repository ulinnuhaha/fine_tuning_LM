import pandas as pd
import json
import argparse

# use argparse to let the user provides values for variables at runtime
def DataTestingArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', 
        type=str, required=True, help='Directory of the dataset files')
    parser.add_argument('--file_name', 
        type=str, required=True, help='Name of data')
    parser.add_argument('--batch_sample', 
        type=int, default=3, help='Number of request sentences per batch of target translation')
    args = parser.parse_args()
    return args

def main():
    args = DataTestingArguments()  # call the arguments

    # Load the file
    data = pd.read_csv(f"{args.dataset_dir}/{args.file_name}.csv")
    
    # Define a function to remove punctuation
    data['ladin'] = data['ladin'].str.replace('"', '', regex=False)
    data['english'] = data['english'].str.replace('"', '', regex=False)
    data['ladin'] = data['ladin'].str.replace("'", "’", regex=False)
    data['english'] = data['english'].str.replace("'", "’", regex=False)

    llama_format = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    {model_answer}<|eot_id|>
    """

    # List to store all JSON objects
    with open(f"{args.dataset_dir}/{args.file_name}_tai.jsonl", "w", encoding="utf-8") as new_file:
        for batch_start in range(0, len(data), args.batch_sample):
            # Create batch of training data
            dataset = data.iloc[batch_start:batch_start + args.batch_sample]

            # Set the translation direction
            translation_directions = [
                {
                    "from_lang": "english",
                    "to_lang": "Ladin",
                    "from_samples": dataset['english'].tolist(),
                    "to_samples": dataset['ladin'].tolist()
                },
                {
                    "from_lang": "Ladin",
                    "to_lang": "english",
                    "from_samples": dataset['ladin'].tolist(),
                    "to_samples": dataset['english'].tolist()
                }
            ]

            # Create prompts for both translation directions
            for direction in translation_directions:
                prompt_1 = f"This is a translation system that translates {direction['from_lang']} into {direction['to_lang']}. Provide accurate translations by preserving the meaning.\n"
                prompt_2 = f"Translate the following {len(dataset)} {direction['from_lang']} texts to {direction['to_lang']}:\n"

                # Create request and response
                source_samples = direction['from_samples']
                target_samples = direction['to_samples']

                request = f"{prompt_2} {source_samples}"
                assistant_response = f"Here are the {direction['to_lang']} translations: \n{target_samples}"

                # Write the data to the JSON file
                temp_data = {
                    "text": llama_format.format(
                        system_prompt=prompt_1,
                        user_question=request,
                        model_answer=assistant_response
                    )
                }
                new_file.write(json.dumps(temp_data))
                new_file.write("\n")


    print(f"Output written to {args.dataset_dir}/{args.file_name}_tai.jsonl")
if __name__ == "__main__":
    main()
