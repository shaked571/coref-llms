import os
import argparse

from transformers import AutoTokenizer

from utils.io_utils import *
from data_processing import get_dataset_readers, HebrewExampleDatasetReader
from prompt import get_prompts
from llm import get_generations
from evaluate import get_evaluations


def hebrew_coref():
    """
       What I saw in 25/12/24 - which fails on the tokenizer - of NLTK that does not exist
       The files of mentions_by_model are the mentions produced by  Danit's parse tree model with my mention detection model
       The mentioned produced using the script in: HebNpChunker/make_paper_mentions_for_llm.py
       The files of conllu_gold are the gold mentions (Can use the one "with singletons" to get all the mentions or without to do an easier version).

       In order to run gpt on "real life" mentions, you need to run the following command:
       --exp_dir ../results/dev_heb
       --gold_data /Users/s0g0a87/studies/coref-llms/data_coref/hebrew/conllu_gold/no_singleton/dev
       --eval_data /Users/s0g0a87/studies/coref-llms/data_coref/hebrew/mentions_by_model/dev
        --model_id gpt-4 --prompt_template doc_template

        In order to run gpt on gold mentions, you need to run the following command:
        --exp_dir  ../results/dev_heb_on_gold
        --gold_data /Users/s0g0a87/studies/coref-llms/data_coref/hebrew/conllu_gold/no_singleton/dev
        --eval_data /Users/s0g0a87/studies/coref-llms/data_coref/hebrew/conllu_gold/no_singleton/dev
        --model_id gpt-4
        --prompt_template doc_template

        For running e2e on raw text we first produce prediction using train data:
            1. Run the following command to create some reference for teh prompt (need to be done one time):
                --exp_dir ../results/e2e_train/train_output_for_reference
                --gold_data /Users/s0g0a87/studies/coref-llms/data_coref/hebrew/conllu_gold/no_singleton/train
                --eval_data /Users/s0g0a87/studies/coref-llms/data_coref/hebrew/conllu_gold/no_singleton/train
                --model_id gpt-4
                --prompt_template doc_template
            2. Take the shortest example from the output and create the prompt for the e2e model (need to be done one time):
            3. Run the following command to create the predictions:
                --exp_dir ../results/e2e_train/raw_text/dev
                --gold_data /Users/s0g0a87/studies/coref-llms/data_coref/hebrew/conllu_gold/no_singleton/dev
                --eval_data /Users/s0g0a87/studies/coref-llms/data_coref/hebrew/raw_documents/dev
                --model_id gpt-4
                --prompt_template e2e_template

        For running e2e on documented text we first produce prediction using train data:
            1. Run the following command to create some reference for teh prompt (need to be done one time):
                --exp_dir ../results/e2e_train/train_output_for_reference
                --gold_data /Users/s0g0a87/studies/coref-llms/data_coref/hebrew/conllu_gold/no_singleton/train
                --eval_data /Users/s0g0a87/studies/coref-llms/data_coref/hebrew/conllu_gold/no_singleton/train
                --model_id gpt-4
                --prompt_template doc_template
            2. Take the shortest example from the output and create the prompt for the e2e model (need to be done one time):
            3. Run the following command to create the predictions:
                --exp_dir ../results/e2e_train/tokenized_text/dev
                --gold_data /Users/s0g0a87/studies/coref-llms/data_coref/hebrew/conllu_gold/no_singleton/dev
                --eval_data /Users/s0g0a87/studies/coref-llms/data_coref/hebrew/tokenized_documents/dev
                --model_id gpt-4
                --prompt_template e2e_template

       """
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, help="Filepath to experiment directory")
    parser.add_argument("--eval_data", type=str, help="Filepath to evaluation data")
    parser.add_argument("--gold_data", type=str, help="Filepath to gold data")
    parser.add_argument("--model_id", type=str, help="ID of LLMs, e.g. gpt-4")
    parser.add_argument(
        "--prompt_template",
        type=str,
        help="`doc_template` or `qa_template` or `e2e_template`)",
    )

    args = parser.parse_args()
    # make experiment directory
    os.makedirs(args.exp_dir, exist_ok=True)
    predicted_data = args.eval_data
    gold_data = args.gold_data
    # read in data
    if args.prompt_template == "e2e_template":
        dataset_reader = HebrewExampleDatasetReader(gold_data, predicted_data)
        data = dataset_reader.split_raw_text()
        print(f"Dataset has {len(data)} examples")
    else:
        dataset_reader = HebrewExampleDatasetReader(gold_data, predicted_data)
        data = dataset_reader.split()
        print(f"Dataset has {len(data)} examples")

    # prompt generation
    prompts = get_prompts(data, args.exp_dir, args.prompt_template)

    # response generation
    responses = get_generations(prompts, args.exp_dir, args.model_id)

    # extract answers from response
    doc_predictions = dataset_reader.aggregate(responses)
    dp_filepath = os.path.join(args.exp_dir, f"doc_predictions.jsonl")
    write_jsonl(doc_predictions, dp_filepath)

    # evaluate results
    get_evaluations(doc_predictions, args.exp_dir)

def main():
    """
    --exp_dir ./test_en --eval_data  /Users/s0g0a87/studies/coref-llms/data/example.jsonl --model_id gpt-4 --prompt_template doc_template     :return:
    """



    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, help="Filepath to experiment directory")
    parser.add_argument("--eval_data", type=str, help="Filepath to evaluation data")
    parser.add_argument("--model_id", type=str, help="ID of LLMs, e.g. gpt-4")
    parser.add_argument(
        "--prompt_template",
        type=str,
        help="`doc_template` or `qa_template`)",
    )

    args = parser.parse_args()

    # make experiment directory
    os.makedirs(args.exp_dir, exist_ok=True)

    # read in data
    dataset_reader = get_dataset_readers(args)
    data = dataset_reader.split()
    print(f"Dataset has {len(data)} examples")

    # prompt generation
    prompts = get_prompts(data, args.exp_dir, args.prompt_template)

    # response generation
    responses = get_generations(prompts, args.exp_dir, args.model_id)

    # extract answers from response
    doc_predictions = dataset_reader.aggregate(responses)
    dp_filepath = os.path.join(args.exp_dir, f"doc_predictions.jsonl")
    write_jsonl(doc_predictions, dp_filepath)

    # evaluate results
    get_evaluations(doc_predictions, args.exp_dir)


if __name__ == "__main__":
    # take args
    # main()
    hebrew_coref()
