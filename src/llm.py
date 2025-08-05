import os
from nltk import tokenize

os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,7"

import openai
import tiktoken
import requests
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import re

from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
from Crypto.Hash import SHA256
# from peft import PeftModel, PeftConfig

from utils.io_utils import *
import torch



# Check if CUDA (NVIDIA GPU) is available
print(torch.cuda.is_available())
# If True, print the number of GPUs and the name of the first GPU
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")






def get_generations(data, prompts: dict, exp_dir: str, model_id: str) -> dict:
    # run generation
    data_by_doc = {d['doc_key']:d for d in data}
    generation_filepath = os.path.join(exp_dir, "generations.json")
    model = MODEL_TYPE[model_id](model_id)
    if os.path.exists(generation_filepath):  #
        generations = read_json(generation_filepath)
    else:
        generations = dict()
    missing_prompts = {e_key: prompt for e_key, prompt in prompts.items() if e_key not in generations}
    print(f"Number of missing prompts: {len(missing_prompts)}")
    for e_key, _ in missing_prompts.items():
        print(f"Generating prompt {e_key}")
    if isinstance(model, OpenAIModel):
        model.compute_cost(missing_prompts)
    generations = model.inference(data, missing_prompts, generation_filepath)

    while True:
        invalid_generations = {}
        for g_key, gen in generations.items():
            if not is_valid_gen(
                    data_by_doc[g_key]["input_context_str"].strip(),
                    g_key,
                    generations,
                    data_by_doc[g_key]["output_priming"]
            ):
                invalid_generations[g_key] = gen

        if len(invalid_generations) == 0:
            break  # Exit loop when all generations are valid

        print(f"Retrying {len(invalid_generations)} invalid generations...")
        print("The following generations failed:")
        for g_key, gen in invalid_generations.items():
            print(f"Failed: {g_key}")
            generations.pop(g_key)
        write_json(generations, generation_filepath)
        new_generations = model.inference(data,
            {g_key: gen["prompt"] for g_key, gen in invalid_generations.items()},
            generation_filepath
        )

        generations = new_generations

    return generations



def is_valid_gen(input_text: str, g_key: str, generations: dict, output_priming: str) -> bool:
    original_text = generations[g_key]["generated_text"]
    generations[g_key]["generated_text"] = generations[g_key]["generated_text"].strip().strip("</s>").strip("<s>")
    if generations[g_key]["generated_text"].startswith(output_priming): # handle cases where the llm just generate the all text
        generations[g_key]["generated_text"] = generations[g_key]["generated_text"].removeprefix(output_priming)

    # Define a robust regex pattern to match variations of E2E_END and clusters_end
    coref_pattern = re.compile(r"-+\n+#+\s*Coreference Clusters:?\n", re.IGNORECASE)
    clusters_pattern = re.compile(r"-+\n+#+\s*Clusters:?\n", re.IGNORECASE)

    # Remove everything after and including the matched patterns
    generations[g_key]["generated_text"] = coref_pattern.split(generations[g_key]["generated_text"])[0].strip()
    generations[g_key]["generated_text"] = clusters_pattern.split(generations[g_key]["generated_text"])[0].strip()
    generated_text = "{0}{1}".format(
        output_priming,
        generations[g_key]["generated_text"],
    ).strip() if not generations[g_key]["generated_text"].startswith(output_priming) else generations[g_key]["generated_text"]
    generated_sents = tokenize.sent_tokenize(generated_text)
    input_sents = tokenize.sent_tokenize(input_text)
    if len(generated_sents) != len(input_sents):
        heuristic_fix_for_sent_sep = apply_sentence_split_heuristic(tokenize, generated_text)
        return len(heuristic_fix_for_sent_sep) == len(input_sents)
    return len(generated_sents) == len(input_sents)




from openai import OpenAI
import os, json
from tqdm import tqdm

class HFModels:  # preserves old name for backwards compatibility
    def __init__(self, model_name, temperature=0.0):
        self.model_name = model_name
        self.temperature = temperature #
        self.client = OpenAI(
            api_key="EMPTY",
            base_url="http://10.22.131.235:8000/v1",
        )

    def inference(self, data, prompts: dict, generation_filepath: str):
        generations = {}
        if os.path.exists(generation_filepath):
            generations = json.load(open(generation_filepath))

        new_keys, payloads = zip(*[(k, p) for k, p in prompts.items()
                                   if k not in generations]) if prompts else ([], [])

        for k, prompt in tqdm(zip(new_keys, payloads), total=len(payloads)):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            generations[k] = {
                "prompt": prompt,
                "generated_text": response.choices[0].message.content,
            }

        write_json(generations, generation_filepath)
        return generations


    def build_prompt_messages(self, prompt):
        messages = [
            {"role": "user", "content": prompt},
        ]
        return messages




class OpenAIModel:
    """Wrapper for OpenAI Models (eg gpt-35, gpt-4)"""

    # per-token pricing https://openai.com/api/pricing/ snapshot on 11/09/2023
    MODEL_INFO = {
        "gpt-4": {
            "max_context_len": 8100,
            "input_cost": 0.00003,  # $0.03 per 1K tokens
            "output_cost": 0.00006,  # $0.06 per 1K tokens
        },
        "gpt-4-32k": {
            "max_context_len": 32000,
            "input_cost": 0.00006,  # $0.06 per 1K tokens
            "output_cost": 0.00012,  # $0.12 per 1K tokens
        },
        "gpt-4o": {
            "max_context_len": 16000,
            "input_cost": 0.0000025,
            "output_cost": 0.00001,
        },
        "gpt-4o-mini": {
            "max_context_len": 16000,
            "input_cost": 0.0000025,
            "output_cost": 0.00001,
        },
        "gemini-2.5-pro": {
            "max_context_len": 8000,
            "input_cost": 0.0000025,
            "output_cost": 0.00001,
        },
        "gemini-2.0-flash": {
            "max_context_len": 8000,
            "input_cost": 0.0000025,
            "output_cost": 0.00001,
        },
        "gemini-2.0-flash-lite": {
            "max_context_len": 8000,
            "input_cost": 0.0000025,
            "output_cost": 0.00001,
        },"o1": {
            "max_context_len": 16000,
            "input_cost": 0.0000025,
            "output_cost": 0.00001,
        },"o3": {
            "max_context_len": 16000,
            "input_cost": 0.0000025,
            "output_cost": 0.00001,
        },
        "gpt-35-turbo": {
            "max_context_len": 4000,
            "input_cost": 0.0000015,  # $0.003 per 1K tokens
            "output_cost": 0.000002,  # $0.004 per 1K tokens
        },
        "gpt-4.1": {
            "max_context_len": 16000,
            "input_cost": 0.0000015,  # $0.003 per 1K tokens
            "output_cost": 0.000002,  # $0.004 per 1K tokens
        },
        "gpt-3.5-turbo-16k": {
            "max_context_len": 16000,
            "input_cost": 0.000003,  # $0.003 per 1K tokens
            "output_cost": 0.000004,  # $0.004 per 1K tokens
        },
        "gpt-3.5-turbo-instruct": {
            "max_context_len": 4000,
            "input_cost": 0.0000015,  # $0.003 per 1K tokens
            "output_cost": 0.000002,  # $0.004 per 1K tokens
        },
    }

    def __init__(self, model_name: str) -> None:
        self._api_version = self.determine_api_version(model_name)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name) if model_name not in {"o1","o3","gpt-4.1"} and not  model_name.startswith("gemini") else  tiktoken.encoding_for_model("gpt-4-o")
        self.model_info = OpenAIModel.MODEL_INFO[model_name]
        self.max_context_len = self.model_info["max_context_len"]

    def determine_api_version(self, model_name: str) -> str:
        """
        Determines the API version based on the model name.
        """
        if model_name in {"o1","o3"} :
            return "2024-12-01-preview"
        elif model_name.startswith("gemini") :
            return "v1"
        return "2024-10-21"

    def create_payload(self, messages, model, temperature, max_tokens):
        """
        Prepares the payload for model calls. Handles O1/O3, OpenAI, and Gemini Vertex models.
        """
        if model in {"o1", "o3"}:
            # O1/O3, likely OpenAI-compatible
            return {
                "model": model,
                "task": "chat/completions",
                "api-version": self._api_version,
                "model-params": {
                    "messages": messages,
                    # "temperature": temperature,
                    # "max_tokens": max_tokens,
                }
            }
        elif model.startswith("gemini"):
            # Convert OpenAI-style messages to Gemini-style contents
            contents = []
            for m in messages:
                # Each message: {"role": "...", "content": "..."}
                role = m.get("role", "user").replace("assistant", "user") # Gemini uses User rather than assistant
                content = m.get("content", "")
                # Gemini expects a "parts" array with one or more dicts (e.g., {"text": ...})
                if isinstance(content, list):
                    parts = [{"text": str(x)} for x in content]
                else:
                    parts = [{"text": str(content)}]
                contents.append({"role": role, "parts": parts})

            return {
                "model": model,
                "task": "generateContent",
                "api-version": self._api_version,
                "generation_config": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "topP": 0.8,  #
                    "topK": 40
                          },
                "model-params": {
                    "contents": contents,
                }
            }
        else:
            # Generic OpenAI format
            return {
                "model": model,
                "task": "chat/completions",
                "api-version": self._api_version,
                "model-params": {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            }
    def get_headers(self):
        key_version = 2
        # key_version = 1
        # consumer_id = "e6a8f4da-9070-46e1-8aa0-f1c33cd094ee"
        consumer_id = "262363a8-70c5-4fd6-ac4e-083a301db22d"
        env = "stage"
        with open('pk_llama_non_prod', 'r') as file:
        # with open('pk_stage', 'r') as file:
            pvt_key_base64 = file.read()

        rsa_pem = base64.b64decode(pvt_key_base64)
        timestamp = int(time.time()) * 1000
        data = f"{consumer_id}\n{timestamp}\n{key_version}\n"
        rsakey = RSA.importKey(rsa_pem)
        signer = PKCS1_v1_5.new(rsakey)
        digest = SHA256.new()
        digest.update(data.encode('utf-8'))
        sign = signer.sign(digest)

        s, ts = base64.b64encode(sign).decode("utf-8"), str(timestamp)
        return {
            "WM_CONSUMER.ID": consumer_id,
            "WM_SVC.NAME": "WMTLLMGATEWAY",
            "WM_SVC.ENV": env,
            "WM_SEC.KEY_VERSION": str(key_version),
            "WM_SEC.AUTH_SIGNATURE": s,
            "WM_CONSUMER.INTIMESTAMP": ts,
            "Content-Type": "application/json"
        }

    # @retry(wait=wait_random_exponential(min=0.3, max=2), stop=stop_after_attempt(2))
    def completion_with_backoff(
            self,
            messages,
            temperature,
            max_tokens,
            model="gpt-4o",

    ):
        pathway = "google-genai" if model.startswith("gemini") else "openai"
        url = f"https://wmtllmgateway.stage.walmart.com/wmtllmgateway/v1/{pathway}"
        payload = self.create_payload(messages=messages,
                                 model=model,
                                 temperature=temperature,
                                 max_tokens=max_tokens,
                                 )
        response = requests.request("POST", url, headers=self.get_headers(), json=payload, verify=False)
        return response

    def inference(self, data, missing_prompts: dict, generation_filepath: str) -> dict:
        data_by_doc = {d['doc_key']: d for d in data}

        # resume generation if exist
        if os.path.exists(generation_filepath):  #
            generations = read_json(generation_filepath)
        else:
            generations = dict()
        print("Missing prompts are:")
        for prompt_key in missing_prompts.keys():
            print(prompt_key)
        with ThreadPoolExecutor(max_workers=max(min(len(missing_prompts), (os.cpu_count() or 1)),1))  as executor:
            futures = {
                executor.submit(self.generate_text, e_key, prompt): e_key
                for e_key, prompt in missing_prompts.items()
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                e_key = futures[future]
                try:
                    result = future.result()
                    if result["generated_text"] != "":
                        generations[e_key] = result
                        write_json(generations, generation_filepath)
                except Exception as e:
                    print(e)
                    print("Cannot generate text for example={0}".format(e_key))
                write_json(generations, generation_filepath)

        while True:
            invalid_generations = {}

            for g_key, gen in generations.items():
                if not is_valid_gen(
                        data_by_doc[g_key]["input_context_str"].strip(),
                        g_key,
                        generations,
                        data_by_doc[g_key]["output_priming"]
                ):
                    invalid_generations[g_key] = gen

            if len(invalid_generations) == 0:
                break  # Exit loop when all generations are valid

            print(f"Retrying {len(invalid_generations)} invalid generations...")
            print("The following generations failed:")
            for g_key, gen in invalid_generations.items():
                print(f"Failed: {g_key}")
                generations.pop(g_key)
            write_json(generations, generation_filepath)
            new_generations = self.inference(data,
                {g_key: gen["prompt"] for g_key, gen in invalid_generations.items()},
                generation_filepath
            )

            generations = new_generations

        return generations

    def compute_cost(self, prompts: dict):
        """Estimate the amount of $ it takes to run this experiment =
        $/token x sum over all prompts of (# of tokens/prompt)
        """
        estimated_cost = 0
        for key, prompt in prompts.items():

            # tokenize and get tokens for this input
            num_input_tokens = len(self.tokenizer.encode(prompt))
            max_output_tokens = self.max_context_len - num_input_tokens
            estimated_cost += (
                self.model_info["input_cost"] * num_input_tokens
                + self.model_info["output_cost"] * max_output_tokens
            )

        # output the cost
        print(f"Estimated cost for {self.model_name}: ${estimated_cost:0.2f}")

    def extract_gemini_response_text(self, response_json):
        candidates = response_json.get("candidates", [])
        if not candidates:
            return ""
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        return "".join(part.get("text", "") for part in parts)

    def generate_text(self, e_key, prompt):
        max_generated_len = self.max_context_len - len(self.tokenizer.encode(prompt))
        temperature = 0.0
        completion = self.completion_with_backoff(
            model=self.model_name,
            messages=[
                {
                    "role": "assistant",
                    "content": prompt,
                },
            ],
            max_tokens=max_generated_len,
            temperature= temperature,
        )

        if completion.status_code != 200:
            print(f"Completion status code: {completion.status_code}")
            print(f"Failed to generate text for example={e_key}")
            return {
                "prompt": prompt,
                "generated_text": "",
            }

        response_json = json.loads(completion.text)
        if self.model_name.startswith("gemini"):
            output_text = self.extract_gemini_response_text(response_json)
        else:
            output_text = response_json['choices'][0]['message']["content"]

        print("Finished Prompt: {0}{1}".format(prompt, output_text))
        return {
            "prompt": prompt,
            "generated_text": output_text,
        }

class AzureModel:
    """Wrapper for Azure Models (eg gpt-35, gpt-4)
    TODO: refactor into OpenAI when released
    """

    # per-token pricing https://openai.com/api/pricing/ snapshot on 11/09/2023
    MODEL_INFO = {
        "gpt-4": {
            "max_context_len": 8000,
            "input_cost": 0.00003,  # $0.03 per 1K tokens
            "output_cost": 0.00006,  # $0.06 per 1K tokens
        },
        "gpt-3.5-turbo": {
            "max_context_len": 4000,
            "input_cost": 0.0000015,  # $0.003 per 1K tokens
            "output_cost": 0.000002,  # $0.004 per 1K tokens
        },
        "gpt-3.5-turbo-16k": {
            "max_context_len": 16000,
            "input_cost": 0.000003,  # $0.003 per 1K tokens
            "output_cost": 0.000004,  # $0.004 per 1K tokens
        },
    }

    def __init__(self, model_name: str, max_generated_len: int) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.model_info = OpenAIModel.MODEL_INFO[model_name]
        self.max_context_len = self.model_info["max_context_len"]
        self.max_generated_len = max_generated_len  # TODO I think this is deprecated

    def inference(self, prompts: dict, generation_filepath: str) -> dict:

        DEPLOYMENT_NAME = "coref"

        # resume generation if exist
        generations = dict()
        if os.path.exists(generation_filepath):
            generations = read_json(generation_filepath)

        for e_key, prompt in prompts.items():

            if e_key not in generations:

                try:
                    max_generated_len = self.max_context_len - len(
                        self.tokenizer.encode(prompt)
                    )

                    completion = openai.ChatCompletion.create(
                        engine=DEPLOYMENT_NAME,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ],
                        max_tokens=max_generated_len,
                        temperature=0,
                    )
                    output_text = completion.choices[0].message["content"]
                    print("Finished Prompt: {0}{1}".format(prompt, output_text))
                    generations[e_key] = {
                        "prompt": prompt,
                        "generated_text": output_text,
                    }
                    write_json(generations, generation_filepath)

                except:
                    print("Cannot generate text for example={0}".format(e_key))

        write_json(generations, generation_filepath)
        return generations

    def compute_cost(
        self,
        prompts: dict,
    ):
        """Estimate the amount of $ it takes to run this experiment =
        $/token x sum over all prompts of (# of tokens/prompt)
        """
        estimated_cost = 0
        for key, prompt in prompts.items():

            # tokenize and get tokens for this input
            num_input_tokens = len(self.tokenizer.encode(prompt))
            max_output_tokens = self.max_context_len - num_input_tokens
            estimated_cost += (
                self.model_info["input_cost"] * num_input_tokens
                + self.model_info["output_cost"] * max_output_tokens
            )

        # output the cost
        print(f"Estimated cost for {self.model_name}: ${estimated_cost:0.2f}")


MODEL_TYPE = {
    "llama": HFModels,
    "llama-2": HFModels,
    "codellama": HFModels,
    "dicta-il/dictalm2.0-instruct": HFModels,
    "gpt-35-turbo": OpenAIModel,
    "gpt-3.5-turbo-16k": OpenAIModel,
    "gpt-3.5-turbo-instruct": OpenAIModel,
    "gpt-4": OpenAIModel,
    "gpt-4.1": OpenAIModel,
    "gpt-4o": OpenAIModel,
    "gpt-4o-mini": OpenAIModel,
    "gpt-4-32k": OpenAIModel,
    "o1": OpenAIModel,
    "o3": OpenAIModel,
    "gemini-2.5-pro": OpenAIModel,
    "gemini-2.0-flash-lite": OpenAIModel,
    "gemini-2.0-flash": OpenAIModel
}

# if __name__ == '__main__':
#     model_name = "gpt-4o"
#     model = OpenAIModel(model_name)
#     with open("prompt_temp", encoding='utf-8', mode='r') as file:
#         prompt = "\n".join(file.readlines())
#     #Annotate all entity mentions in the following text with coreference clusters. Use Markdown tags to indicate clusters in the output, with the following format [mention](#cluster_name)\n\nInput: [Tom](#) and [Mary](#) go to [the park](#). [It](#) was full of trees.\nOutput:"
#     max_generated_len = 16000 - len(model.tokenizer.encode(prompt))
#     completion = model.completion_with_backoff(
#         model=model.model_name,
#         messages=[
#             {
#                 "role": "assistant",
#                 "content": prompt,
#             },
#         ],
#         max_tokens=max_generated_len,
#         temperature=1.0,
#     )
#     output_text = json.loads(completion.text)['choices'][0]['message']["content"]
