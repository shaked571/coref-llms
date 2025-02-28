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
                        "Cluster all entity mentions in the following Hebrew text to coreference clusters. "
                        "Use Markdown tags to indicate the coreference group in the output, with the following format [mention](#) e.g.\n"
                        "כש[הם](#) הלכו ל[בית של [אנחנו]](#)(#)\n"
                        "### As a first step tokenized words in order to introduce the clitics\n" 
                        "### As a second step mark the mentions\n" 
                        "### Finally, cluster all the coreference clusters together\n"
                        "## The tokenization would be done in the following way, introducing the clitics\n"
                        "# Input word:\n"
                        "ביתנו\n"
                        "# Output:\n"
                        "בית_ _של_ _אנחנו\n"
                        "# Input word:\n"
                        "העסקתם\n"
                        "# Output:\n"
                        "העסקה_ _של_הם\n"
                        "## The mentions would be mark in the following way:\n"
                        "# Input tokenized:\nהוא איבד את הכרה_ _של_ _הוא\n"
                        "# Output:\n[הוא](#) איבד את [הכרה_ _של_ [_הוא]](#)(#)\n\n"
                        
                        "## The clusters would be marked in the following way:\n"
                        "# Input tokenized:\n[הוא](#) איבד את [הכרה_ _של_ [_הוא]](#)(#)\n"
                        "# Output:\n[הוא](#cluster_1) איבד את [הכרה_ _של_ [_הוא]](#cluster_1)(#cluster_2)\n\n"
                        "Examples:\n\n"
                        "# Input document:\nלידס עלתה למקום החמישי אחרי שניצחה אתמול בחוץ במשחק השלמה את מנצסטר סיטי 3 2. השערים ללידס: לי צפמאן (14), קארל שאט (42), גורדון סטראקאן (62). לסיטי: אשלי וורד (49 מ-11 מ), דייויד ווייט (65)"
                        ".\n# Output cluster document:\n[לידס](#cluster_0) עלתה ל ה_ מקום ה חמישי אחרי ש ניצחה אתמול ב ה_ חוץ ב משחק השלמה את[מנצסטר סיטי](#cluster_1) 3 2.[[ה שערים ל[לידס](#cluster_0)](#) :[לי צפמאן (14), קארל שאט (42), גורדון סטראקאן (62)](#)]. ל[סיטי](#cluster_1) : אשלי וורד (49 מ - 11 מ), דייויד ווייט (65)."
                        "\n# Input document:\nהרבה החמצות ממצבים נוחים של יבנה, בגלל משחק הגנתי של טבריה שהזמינה התקפות. בין חלוצי יבנה, שהרבו להחמיץ, ניצל אנריקה ורון הזדמנות אחת בלבד, כדי להעניק לקבוצתו פרס של 3 נקודות בעד נצחון שהיתה ראויה לו. שפט אריה וולף, 1,000 צופים, ביבנה."
                        "# Output cluster document:\n  הרבה החמצות מ מצבים נוחים של[יבנה](#cluster_0), בגלל משחק הגנתי של טבריה ש הזמינה התקפות. בין חלוצי[יבנה](#cluster_0), ש הרבו להחמיץ, ניצל[אנריקה ורון](#cluster_1) הזדמנות אחת בלבד, כדי להעניק ל[קבוצה _של_[_הוא](#cluster_1)](#cluster_0) פרס של 3 נקודות בעד[נצחון ש היתה ראויה ל[_הוא](#cluster_0)](#cluster_1). שפט אריה וולף, 1,000 צופים, ב יבנה.\n"

                        "### Allow nested mentions\n"
                        "### Mark both nested mentions and also inner nested mentions and every noun phrase which can be candidate\n"
                        "### Mark all spans as if it is Ontonotes 5.0 coreference dataset\n"
                        "### It is very important to keep the text exactly as it was except the mention Markdown\n"
                        "### No need to output sinfgelton in the final Output cluster document\n" 
                        "Input document:\n"
                        f"{input_text}\n"
                        "Output cluster document:\n"
                                            )
                }
            ]
        }
    ]
    return messages


if __name__ == '__main__':

    input_docs = "/Users/s0g0a87/studies/coref-llms/data_coref/hebrew/raw_documents/dev"
    output_fodler = "/Users/s0g0a87/studies/coref-llms/src/e2e_train/e2e_by_llm_dev"
    all_raw_docs = read_all_raw_docs(input_docs)

    messages = {fname: build_messages(doc)for fname, doc in all_raw_docs.items()}
    for fname, messages in messages.items():
        response = completion_with_backoff(messages, 0.0, 8000, 0, 0)
        try:
            result = response.json()['choices'][0]['message']['content']
            print(result)
            print("_______________________")
            with open(os.path.join(output_fodler, fname), "a", encoding='utf-8') as f:
                f.write(result)
        except KeyError as e:
            print(e)