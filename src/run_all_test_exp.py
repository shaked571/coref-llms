import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

seeds = [1, 2, 3, 4, 5]
model_ids = [
    "o3",
    "o1",
    "gpt-4o",
    "gpt-4.1",
    "dicta-il/dictalm2.0-instruct",
    "gemini-2.5-pro",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gpt-4o-mini"
]

gold_data = "../data_coref/hebrew/conllu_gold/no_singleton/test"

tasks = [
    {
        "name": "e2e_train/tokenized_text",
        "eval_data": "../data_coref/hebrew/tokenized_documents/test",
        "prompt_template": "e2e_template"
    },
    {
        "name": "e2e_train/raw_text",
        "eval_data": "../data_coref/hebrew/raw_documents/test",
        "prompt_template": "e2e_template"
    },
    {
        "name": "e2e_train/danit_tokenization",
        "eval_data": "../data_coref/hebrew/tokenized_documents_danit_tokenization/test",
        "prompt_template": "e2e_template"
    },
    {
        "name": "danit_parse_md_mentions",
        "eval_data": "../data_coref/hebrew/mentions_by_model_danit_parse/test",
        "prompt_template": "doc_template"
    },
    {
        "name": "gold_parse_md_mentions",
        "eval_data": "../data_coref/hebrew/mentions_by_model_gold_parse/test",
        "prompt_template": "doc_template"
    },
    {
        "name": "gold_mentions",
        "eval_data": "../data_coref/hebrew/conllu_gold/no_singleton/test",
        "prompt_template": "doc_template"
    },
]

model_id_to_dir = defaultdict(
    lambda: None,  # Default value if key is not found
    {
        "dicta-il/dictalm2.0-instruct": "dicta",
        "o3": "o3",
        "o1": "o1",
        "gpt-4o": "gpt4o",
        "gpt-4.1": "gpt4.1",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
        "gemini-2.0-flash": "gemini-2.0-flash",
        "gpt-35-turbo": "gpt-3.5-turbo",
    },
)

def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Finished:", " ".join(cmd))
        return result.returncode, cmd, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print("Failed:", " ".join(cmd))
        return e.returncode, cmd, e.stdout, e.stderr

all_jobs = []
for model_id in model_ids:
    model_dir = model_id_to_dir[model_id] or model_id  # Use key as value if not found
    for task in tasks:
        task_name = task["name"]
        eval_data = task["eval_data"]
        prompt_template = task["prompt_template"]
        for seed in seeds:
            exp_dir = f"../results/heb/{model_dir}/test/{task_name}/{task_name.split('/')[-1].rstrip('s')}_{seed}"
            cmd = [
                "python", "main.py",
                "--exp_dir", exp_dir,
                "--gold_data", gold_data,
                "--eval_data", eval_data,
                "--model_id", model_id,
                "--prompt_template", prompt_template
            ]
            all_jobs.append(cmd)

max_workers = 16  # Number of parallel jobs; adjust as needed!

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(run_cmd, cmd) for cmd in all_jobs]
    for future in as_completed(futures):
        rc, cmd, out, err = future.result()
        if rc != 0:
            print(f"Error with command: {' '.join(cmd)}\nError:\n{err}")
