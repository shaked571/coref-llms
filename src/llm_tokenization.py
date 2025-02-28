#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from typing import Dict


from call_llm_walmart import completion_with_backoff

def read_all_raw_docs(folder_path: str) -> Dict[str,str]:
    all_raw_docs = {}
    for file_name in os.listdir(folder_path):
        with open(os.path.join(folder_path, file_name), "r", encoding='utf-8') as f:
            all_raw_docs[file_name] = f.read()
    return all_raw_docs


def build_messages(input_text):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Annotate all entity mentions in the following text for a coreference clusters. "
                        "Use Markdown tags to indicate a mention in the output, with the following format [mention](#) e.g.\n"
                        "כש[הם](#) הלכו ל[בית של [אנחנו]](#)(#)\n"
                        "### As a first step tokenized words in order to introduce the clitics e.g.\n"
                        "ביתנו -> בית_ _של_ _אנחנו\n"
                        "ביתך -> בית_ _של_ _אתה\n"
                        "העסקתם -> העסקה_ _של_הם\n"
                        "# As a second step mark the mentions e.g.\n"
                        "Input:\n"
                        "כשהוא איבד את הכרתו.\n"
                        "Output:\n"
                        "Tokenized - \n"
                        "######\n"
                        "כש[הוא](#) איבד את [הכרה_ _של_ [_הוא]](#)(#)\n"
                        "### Allow nested mentions\n"
                        "### Mark both nested mentions and also inner nested mentions and every noun phrase which can be candidate\n"
                        "### Mark all spans as if it is Ontonotes 5.0 coreference dataset\n"
                        "### It is very important to keep the text exactly as it was except the mention Markdown\n"
                        "### In the MarkDown [mention](#), '#' is the exact symbol, NOT A Number\n"
                        "Input:\n"
                        f"{input_text}\n"
                        "Output:\n"
                                            )
                }
            ]
        }
    ]
    return messages


if __name__ == '__main__':

    input_docs = "/Users/s0g0a87/studies/coref-llms/data_coref/hebrew/raw_documents/dev"
    output_fodler = "/Users/s0g0a87/studies/coref-llms/data_coref/hebrew/mentions_by_llm_from_raw"
    all_raw_docs = read_all_raw_docs(input_docs)


    messages = {fname: build_messages(doc)for fname, doc in all_raw_docs.items()}
    for fname, messages in messages.items():
        response = completion_with_backoff(messages, 0.5, 8000, 0, 0)
        result = response.json()['choices'][0]['message']['content']
        print(result)
        print("_______________________")
        with open(os.path.join(output_fodler, fname), "a", encoding='utf-8') as f:
            f.write(result)
