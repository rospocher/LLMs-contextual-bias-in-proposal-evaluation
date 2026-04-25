import os
import json
import sys
from time import sleep
import pandas as pd
import re
import openai # Import the openai library
from pathlib import Path
import random
import torch

from llama_cpp import Llama
from huggingface_hub import hf_hub_download

AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
client = openai.AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-10-21",
    azure_endpoint = AZURE_OPENAI_ENDPOINT
    )
OPENAI_MODEL = "gpt-5-mini" # Or "gpt-4", "gpt-4o", etc. based on your needs and access
model_path = "gpt"
TEMPERATURE = 0.7

base_dir = "proposals"  # Default base directory, can be overridden by command-line argument
text_data_folder="text_data_short2"
output_data_folder="text_data_full2"

def leggiTesti(percorso_testi):
    dict_testi = {}
    for p in sorted(Path(percorso_testi).glob("*.txt")):
        text = p.read_text(encoding="utf-8")  # keeps \n exactly as in the file
        dict_testi[p.name.replace(".txt","")]=text
    return dict_testi

dict_testi = leggiTesti(os.path.join(base_dir, text_data_folder))

for k,v in dict_testi.items():
   print(f"File: {k} - Lunghezza testo: {len(v)} caratteri")
   print(v[:100])  # print first 100 characters

def generateGpt(testo, client, TEMPERATURE=0.7) -> str:
  
    prompt = f"""Transform the following ERC project objective summary into a plausible 3000 words ERC "Extended Synopsis" (Part B1).
    The "Extended Synopsis" is a high-level, persuasive overview of the scientific proposal. It covers the state-of-the-art, key objectives, and the overall research strategy. It should be ambitious yet convey the high impact, rather than detailed technical methodology.

    Here is the project objective summary to expand into the "Extended Synopsis":

    {testo}
    """
    
    print(f"Prompt prepared.")

    messages = [
        #{"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    test = False
    
    while True:
        if not test:
            try:
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    max_completion_tokens=10000, # Increased max_tokens for potentially longer responses
                    #temperature=TEMPERATURE,
                    #top_p=0.95,
                    # stop parameters are handled differently in OpenAI, usually by including them in the prompt
                    # or letting the model naturally conclude. For a specific format, ensure prompt guides it.
                )
                risposta = response.choices[0].message.content.strip()
                break
            except Exception as e:
                print(f'Eccezione {e}. Retrying...')
                sleep(10)
                continue
        else:
            risposta = "Test generation output for debugging purposes."
            break
    return risposta

rng = random.Random(42)  # pick any fixed seed

storie_items = list(dict_testi.items())
rng.shuffle(storie_items)
index = 1
output_data_folder = os.path.join(base_dir, output_data_folder)
os.makedirs(output_data_folder, exist_ok=True)
for storia, testoStoria in storie_items:
    #for lang in LANGUAGES:
        print(f"Processing story: {storia}")
        #risposta_str = traduzione_gpt(testoStoria, client, TEMPERATURE)
        #risposta_str = traduzione_qwen(testoStoria, lang, llm, TEMPERATURE)
        risposta_str = generateGpt(testoStoria, client, TEMPERATURE)
        print(f"Finito Processing numero {index}")
        print(f"Lunghezza testo: {len(risposta_str)} caratteri")
        print('Scrivo su file\n')
        with open(os.path.join(output_data_folder, f"{storia}_full.txt"), "w", encoding="utf-8") as f:
            f.write(risposta_str)
        index+=1
  