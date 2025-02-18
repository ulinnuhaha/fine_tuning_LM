# Fine-tuning Language Model (LM) 
This repository focuses on exploring fine-tuning techniques for machine translation (MT) using Large Language Models (LLMs). We leverage models like Llama, mBART, and NLLB to perform fine-tuning of MT from Unseen language (new language). Here, we set English-Ladin as a translation system where 'Ladin' is the new language. To run the code in this repository, you should already have a parallel sentence dataset in CSV format containing two columns (Source and target sentences)

## Fine-tuning LLM of Llama
If you want to perform fine-tuning LLM, specifically Llama using TogetherAI API, please go to the following file:
```
FT_llama.ipynb
```
If you want to change the API provider and the LLM version please go to `fine_tuning` repository.

## Fine-tuning Seq2Seq models
If you want to perform fine-tuning Seq2seq (Sequence to sequence) models locally (on server) such as NLLB-200's 1.3B and mBART-50 variants, run the following command:
```
python FT_seq2seq.py \
  --model_checkpoint facebook/nllb-200-1.3B \
  --cache_dir ./pretrained_model \
  --data_dir ./dataset
  --file_name ./pair_sentences
```

After you get the fine-tuned model, you can perform the translation by running the following command:
```
python run_seq2seq_translation.py \
  --model_name_or_path ./pretrained_model/nllb-200-1.3B_Eng2Lad/ \
  --data_dir ./dataset \
  --file_name ./pair_sentences
  --lang_target ./ladin
```
