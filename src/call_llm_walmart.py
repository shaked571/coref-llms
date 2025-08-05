import requests
import base64
import time

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
from Crypto.Hash import SHA256

def get_headers():
    key_version = 1
    # consumer_id = "262363a8-70c5-4fd6-ac4e-083a301db22d"
    consumer_id = "e6a8f4da-9070-46e1-8aa0-f1c33cd094ee"
    env = "stage"
    env = "staging_p13n"
    with open('sandbox_key', 'r') as file:
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

def create_payload(messages, model, temperature, max_tokens, frequency_penalty,
                   presence_penalty):
    return {
        "model": model,
        "task": "chat/completions",
        "model-params": {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }
    }

@retry(wait=wait_random_exponential(min=0.2, max=3), stop=stop_after_attempt(2))
def completion_with_backoff(
        messages,
        temperature,
        max_tokens,
        frequency_penalty,
        presence_penalty,
        model="gpt-4o",

):
    url = "https://wmtllmgateway.stage.walmart.com/wmtllmgateway/v1/openai"
    payload = create_payload(messages=messages,
                             model=model,
                             temperature=temperature,
                             max_tokens=max_tokens,
                             frequency_penalty=frequency_penalty,
                             presence_penalty=presence_penalty)
    response = requests.request("POST", url, headers=get_headers(), json=payload, verify=False)

    return response

