import os

from utils.io_utils import *


def get_prompts(data: list, exp_dir: str, prompt_template: str):

    # get all the arguments
    TEMPLATES = {"doc_template": doc_template, "qa_template": qa_template, "e2e_template": e2e_template}
    template_fn = TEMPLATES[prompt_template]
    prompts_filepath = os.path.join(exp_dir, "prompts.json")

    # read in if exists
    if os.path.exists(prompts_filepath):
        prompts = read_json(prompts_filepath)

    else:
        prompts = dict()
        for example in data:

            e_key = example["example_key"]
            if e_key not in prompts:

                # generate prompt
                prompt = template_fn(example)
                prompts[e_key] = prompt
        write_json(prompts, prompts_filepath)
    return prompts


def doc_template(example: dict) -> str:
    """Example prompt instantiated from this template:
    ```
    Annotate all entity mentions in the following text with coreference clusters.
    Use Markdown tags to indicate clusters in the output, with the following format
    [mention](#cluster_name)

    Input: [Tom](#) and [Mary](#) go to [the park](#). [It](#) was full of trees.
    Output: [Tom](#cluster_0) and [Mary](#cluster_1) go to [the park](#cluster_3). [It](#cluster_3) was full of trees.
    ```
    """
    # instructions
    prompt = "Annotate all entity mentions in the following text with coreference clusters. Use Markdown tags to indicate clusters in the output, with the following format [mention](#cluster_name)\n\n"

    # add example itself
    prompt += "Input: {0}\nOutput:".format(example["input_context_str"])
    prompt += " " + example["output_priming"]

    return prompt

def mention_template_heb(example: dict) -> str:
    """Example prompt instantiated from this template:
    ```
    Annotate all entity mentions in the following text with coreference clusters.
    Use Markdown tags to indicate clusters in the output, with the following format
    [mention](#cluster_name)

    Input: [Tom](#) and [Mary](#) go to [the park](#). [It](#) was full of trees.
    Output: [Tom](#cluster_0) and [Mary](#cluster_1) go to [the park](#cluster_3). [It](#cluster_3) was full of trees.
    ```
    """
    # instructions
    prompt = ("Annotate all entity mentions in the following text for a coreference clusters. "
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
               "כש[הוא](#) איבד את [הכרה_ _של_ [_הוא]](#)(#)\n"
               "### Allow nested mentions\n"
               "### Mark both nested mentions and also inner nested mentions and every noun phrase which can be candidate\n"
               "### Do not add any extra new files to the generated text\n"
               "### Mark all spans as if it is Ontonotes 5.0 coreference dataset\n"
               "### It is very important to keep the text exactly as it was except the mention Markdown\n"
               "### In the MarkDown [mention](#), '#' is the exact symbol, NOT A Number\n"
)

    # add example itself
    prompt += "Input: {0}\nOutput:".format(example["input_context_str"])


    return prompt


def e2e_template(example: dict) ->str:
    """Example prompt instantiated from this template:
    ```
    Annotate all entity mentions in the following text with coreference clusters.
    Use Markdown tags to indicate clusters in the output, with the following format
    [mention](#cluster_name)

    Input: [Tom](#) and [Mary](#) go to [the park](#). [It](#) was full of trees.
    Output: [Tom](#cluster_0) and [Mary](#cluster_1) go to [the park](#cluster_3). [It](#cluster_3) was full of trees.
    ```
    """
    # instructions
    prompt = (
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
        "# Input:\n[הוא](#) איבד את [הכרה_ _של_ [_הוא]](#)(#)\n"
        "# Output:\n[הוא](#cluster_1) איבד את [הכרה_ _של_ [_הוא]](#cluster_1)(#cluster_2)\n\n"
      
        "Examples:\n"
        "# Input:\n"
        "לידס עלתה למקום החמישי אחרי שניצחה אתמול בחוץ במשחק השלמה את מנצסטר סיטי 3 2. השערים ללידס: לי צפמאן (14), קארל שאט (42), גורדון סטראקאן (62). לסיטי: אשלי וורד (49 מ-11 מ), דייויד ווייט (65)"
        ".\n# Output:\n"
        "[לידס](#cluster_0) עלתה ל ה_ מקום ה חמישי אחרי ש ניצחה אתמול ב ה_ חוץ ב משחק השלמה את[מנצסטר סיטי](#cluster_1) 3 2.[[ה שערים ל[לידס](#cluster_0)](#) :[לי צפמאן (14), קארל שאט (42), גורדון סטראקאן (62)](#)]. ל[סיטי](#cluster_1) : אשלי וורד (49 מ - 11 מ), דייויד ווייט (65)."
        "\n# Input:\n"
        "הרבה החמצות ממצבים נוחים של יבנה, בגלל משחק הגנתי של טבריה שהזמינה התקפות. בין חלוצי יבנה, שהרבו להחמיץ, ניצל אנריקה ורון הזדמנות אחת בלבד, כדי להעניק לקבוצתו פרס של 3 נקודות בעד נצחון שהיתה ראויה לו. שפט אריה וולף, 1,000 צופים, ביבנה."
        "\n# Output:\n"
        "הרבה החמצות מ מצבים נוחים של[יבנה](#cluster_0), בגלל משחק הגנתי של טבריה ש הזמינה התקפות. בין חלוצי[יבנה](#cluster_0), ש הרבו להחמיץ, ניצל[אנריקה ורון](#cluster_1) הזדמנות אחת בלבד, כדי להעניק ל[קבוצה _של_[_הוא](#cluster_1)](#cluster_0) פרס של 3 נקודות בעד[נצחון ש היתה ראויה ל[_הוא](#cluster_0)](#cluster_1). שפט אריה וולף, 1,000 צופים, ב יבנה."
        "\n"
        "### Allow nested mentions\n"
        "### Mark both nested mentions and also inner nested mentions and every noun phrase which can be candidate\n"
        "### Mark all spans as if it is Ontonotes 5.0 coreference dataset\n"
        "### It is very important to keep the text exactly as it was except the mention Markdown\n"
        "### No need to output singleton in the final Output cluster document\n"

    )
    # add example itself
    prompt += "#Input: {0}\n# Output:\n".format(example["input_raw_text"])
    return prompt

def qa_template(example: dict) -> str:
    """Example prompt instantiated from this template:
    ```
    Please carefully read the following passages. For each passage, you must identify
    which noun the mention marked in *bold* refers to.

    Passage: [Tom] and [Mary] go to [the park]. *It* was full of trees.
    Question: In the above passage, what does *It* refer to?
    Answer: *It* refers to [the park]
    ```
    """
    # instructions
    prompt = "Please carefully read the following passages. For each passage, you must identify which noun the mention marked in *bold* refers to.\n\n"

    # add example itself
    prompt += "Passage: {0}\nQuestion: In the above passage, what does {1} refer to?\nAnswer: {1} refers to ".format(
        example["context_str"],
        example["anaphor_str"],
    )

    return prompt
