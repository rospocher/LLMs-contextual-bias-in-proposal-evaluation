#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import re
import sys
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from time import sleep

import pandas as pd

# Optional backends are imported lazily inside loader/call helpers where possible.
import openai
from llama_cpp import Llama
from huggingface_hub import hf_hub_download, login
import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend


NONE_COND_ID = "NONE"


# ----------------------------
# Model registry
# ----------------------------
MODEL_SPECS = {
    "gpt": {
        "backend": "azure_openai",
        "api_model": "gpt-4.1-mini",
    },
    "deepseek": {
        "backend": "nvidia_openai",
        "api_model": "deepseek-ai/deepseek-v3.1",
        "extra_body": {"chat_template_kwargs": {"thinking": False}},
    },
    "kimi": {
        "backend": "nvidia_openai",
        "api_model": "moonshotai/kimi-k2-instruct-0905",
        "extra_body": None,
    },  
    "mistral": {
        "backend": "nvidia_openai",
        "api_model": "mistralai/mistral-large-3-675b-instruct-2512",
        "extra_body": None,
    },      
    "glm":{
        "backend": "nvidia_openai",
        "api_model": "z-ai/glm4.7",
        "extra_body": {"chat_template_kwargs":{"enable_thinking":False,"clear_thinking":True}},
    },  
    "phi":{
        "backend": "nvidia_openai",
        "api_model": "microsoft/phi-3-medium-128k-instruct",
        "extra_body": None,
    },    
    "llama":{
        "backend": "nvidia_openai",
        "api_model": "meta/llama-3.3-70b-instruct",
        "extra_body": None,
    },             
    # "llama": {
    #     "backend": "llama_cpp",
    #     "model_id": "unsloth/Llama-3.1-8B-Instruct-GGUF",
    #     "model_basename": "Llama-3.1-8B-Instruct-BF16.gguf",
    #     "prompt_style": "llama",
    # },
    "gemma_local": {
        "backend": "llama_cpp",
        "model_id": "google/gemma-3-27b-it-qat-q4_0-gguf",
        "model_basename": "gemma-3-27b-it-q4_0.gguf",
        "prompt_style": "gemma",
    },
    "gemma": {
        #"backend": "google_openai",
        #"api_model": "gemma-3-27b-it",
        "backend": "nvidia_openai",
        "api_model": "google/gemma-3-27b-it",
        "extra_body": None,        
    },    
    "qwen": {
        "backend": "llama_cpp",
        "model_id": "unsloth/Qwen3-30B-A3B-GGUF",
        "model_basename": "Qwen3-30B-A3B-Q4_K_M.gguf",
        "prompt_style": "qwen",
    },
    "ministral": {
        "backend": "ministral_transformers",
        "model_id": "mistralai/Ministral-3-14B-Instruct-2512-BF16",
    },
    "qwen_api": {
        "backend": "cerebras_openai",
        "api_model": "qwen-3-235b-a22b-instruct-2507",
    },     
}


# ----------------------------
# CLI
# ----------------------------

RUN_MODES = {"blind", "pi-only", "ai-only", "full"}

def parse_args() -> Tuple[str, str, str]:
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python script.py <BASE_DIR> <config_json> <model_path_name> [run_mode]  "
            "(e.g., python script.py proposals qwen pi-only)"
        )
    base_dir = sys.argv[1].strip()
    config_json = sys.argv[2].strip()
    model_path_name = sys.argv[3].strip().lower()
    run_mode = sys.argv[4].strip().lower() if len(sys.argv) >= 5 else "full"

    if model_path_name not in MODEL_SPECS:
        raise SystemExit(
            f"Unknown model_path_name='{model_path_name}'. "
            f"Available: {', '.join(sorted(MODEL_SPECS))}"
        )
    if run_mode not in RUN_MODES:
        raise SystemExit(
            f"Unknown run_mode='{run_mode}'. "
            f"Available: {', '.join(sorted(RUN_MODES))}"
        )
    return base_dir, config_json, model_path_name, run_mode


base_dir, config_json, model_path_name, run_mode = parse_args()
MODEL_SPEC = MODEL_SPECS[model_path_name]

config_path = os.path.join(base_dir, config_json)
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)


# ----------------------------
# Config helpers
# ----------------------------

def ensure_list_of_strings(x: Any, label: str) -> List[str]:
    if not isinstance(x, list) or not all(isinstance(v, str) for v in x):
        raise ValueError(f"{label} must be a list of strings")
    if not x:
        raise ValueError(f"{label} cannot be empty")
    return x



def ensure_metric_candidates(x: Any, label: str) -> List[Dict[str, int]]:
    """
    Accepts either:
      "LOW": [{"h_index": 8, "citations": 320}, ...]
    or
      "LOW": {"h_index": 8, "citations": 320}

    Returns a list of metric dicts.
    """
    if isinstance(x, dict):
        x = [x]

    if not isinstance(x, list) or not x:
        raise ValueError(f"{label} must be a non-empty list of metric dicts")

    out = []
    for i, item in enumerate(x):
        if not isinstance(item, dict):
            raise ValueError(f"{label}[{i}] must be a dict")
        if "h_index" not in item or "citations" not in item:
            raise ValueError(f"{label}[{i}] must contain 'h_index' and 'citations'")
        out.append({
            "h_index": int(item["h_index"]),
            "citations": int(item["citations"]),
        })
    return out


# ----------------------------
# Required config
# ----------------------------
names_pi: Dict[str, List[str]] = {
    "F_N": ensure_list_of_strings(cfg["names_pi"]["F_N"], "names_pi.F_N"),
    "M_N": ensure_list_of_strings(cfg["names_pi"]["M_N"], "names_pi.M_N"),
}
institutions_pi: Dict[str, List[str]] = {
    "TOP_I": ensure_list_of_strings(cfg["institutions_pi"]["TOP_I"], "institutions_pi.TOP_I"),
    "LOW_I": ensure_list_of_strings(cfg["institutions_pi"]["LOW_I"], "institutions_pi.LOW_I"),
}
surname_pi: str = cfg["surname_pi"]

metrics_pi: Dict[str, List[Dict[str, int]]] = {
    "LOW_B": ensure_metric_candidates(cfg["metrics_pi"]["LOW_B"], "metrics_pi.LOW_B"),
    "HIGH_B": ensure_metric_candidates(cfg["metrics_pi"]["HIGH_B"], "metrics_pi.HIGH_B"),
}

ai_usage_cfg: Dict[str, List[str]] = {
    "AI_Y": ensure_list_of_strings(cfg["ai_usage"]["AI_Y"], "ai_usage.AI_Y"),
    "AI_N": ensure_list_of_strings(cfg["ai_usage"]["AI_N"], "ai_usage.AI_N"),
}

TEMPERATURE = float(cfg["temperature"])
TOP_P = float(cfg["top_p"])
MAX_TOKENS = int(cfg["max_tokens"])
NUMBER_OF_ITERATIONS = int(cfg["number_of_iterations"])
data_folder = cfg["data_folder"]
text_data_folder = cfg["text_data_folder"]

AUTHOR_SECTION = cfg["author_section"]
GENAI_SECTION = cfg["genai_section"]

ALLOWED_SCORES = cfg["allowed_scores"]

PROMPT_FILE = cfg["prompt_file"]


# ----------------------------
# Input / output paths
# ----------------------------
percorso_testi = os.path.join(base_dir, text_data_folder)
percorso_prompt = os.path.join(base_dir, PROMPT_FILE)

output_dir = os.path.join(base_dir, data_folder, model_path_name, run_mode)
json_out_dir = os.path.join(output_dir, "JSONoutput")
csv_out_dir = os.path.join(output_dir, "CSVoutput")
os.makedirs(json_out_dir, exist_ok=True)
os.makedirs(csv_out_dir, exist_ok=True)


# ----------------------------
# Deterministic sampling helpers
# ----------------------------

def stable_rng(*parts: str) -> random.Random:
    s = "||".join(str(p) for p in parts)
    seed = int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)



def pick_one(text_id: str, tag: str, options: List[Any]) -> Any:
    if not options:
        raise ValueError(f"Empty options list for tag={tag}")
    rng = stable_rng(text_id, tag)
    return rng.choice(options)



def build_text_level_tokens(text_id: str) -> Dict[str, Any]:
    """
    Deterministic picks per text. These stay fixed across all iterations
    for a given text.
    """
    pi_bf_first = pick_one(text_id, "PI_NAME_F_N", names_pi["F_N"])
    pi_wm_first = pick_one(text_id, "PI_NAME_M_N", names_pi["M_N"])
    pi_top_inst = pick_one(text_id, "PI_INST_TOP_I", institutions_pi["TOP_I"])
    pi_low_inst = pick_one(text_id, "PI_INST_LOW_I", institutions_pi["LOW_I"])
    pi_low_metrics = pick_one(text_id, "PI_METRIC_LOW_B", metrics_pi["LOW_B"])
    pi_high_metrics = pick_one(text_id, "PI_METRIC_HIGH_B", metrics_pi["HIGH_B"])
    ai_yes = pick_one(text_id, "AI_USAGE_Y", ai_usage_cfg["AI_Y"])
    ai_no = pick_one(text_id, "AI_USAGE_N", ai_usage_cfg["AI_N"])

    return {
        "F_N_name_pi": f"{pi_bf_first} {surname_pi}",
        "M_N_name_pi": f"{pi_wm_first} {surname_pi}",
        "TOP_I_inst_pi": pi_top_inst,
        "LOW_I_inst_pi": pi_low_inst,
        "LOW_B_metrics_pi": pi_low_metrics,
        "HIGH_B_metrics_pi": pi_high_metrics,
        "AI_Y_usage": ai_yes,
        "AI_N_usage": ai_no,
    }


# ----------------------------
# IO helpers
# ----------------------------

def leggiTesti(percorso: str) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for p in sorted(Path(percorso).glob("*.txt")):
        d[p.name.replace(".txt", "")] = p.read_text(encoding="utf-8")
    return d



def safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))
    return s.strip("_")[:200]

# ----------------------------
# Load stimuli + prompt templates
# ----------------------------
dict_testi = leggiTesti(percorso_testi)

with open(percorso_prompt, "r", encoding="utf-8") as f:
    prompt_tpl = f.read()

#remove unused parts
prompt_noinfo_tpl = prompt_tpl.replace("{author_section}\n","").replace("{genai_section}\n","")
prompt_pi_only_tpl = prompt_tpl.replace("{genai_section}\n","")
prompt_ai_only_tpl = prompt_tpl.replace("{author_section}\n","")

#replace sections
prompt_tpl = prompt_tpl.replace("{genai_section}",GENAI_SECTION).replace("{author_section}",AUTHOR_SECTION)
prompt_pi_only_tpl = prompt_pi_only_tpl.replace("{author_section}",AUTHOR_SECTION)
prompt_ai_only_tpl = prompt_ai_only_tpl.replace("{genai_section}",GENAI_SECTION)

# ----------------------------
# Conditions
# ----------------------------
FULL_CONDITIONS: List[Tuple[str, str, str, str]] = [
    ("F_N", "TOP_I", "LOW_B", "AI_Y"),
    ("F_N", "TOP_I", "LOW_B", "AI_N"),
    ("F_N", "TOP_I", "HIGH_B", "AI_Y"),
    ("F_N", "TOP_I", "HIGH_B", "AI_N"),
    ("F_N", "LOW_I", "LOW_B", "AI_Y"),
    ("F_N", "LOW_I", "LOW_B", "AI_N"),
    ("F_N", "LOW_I", "HIGH_B", "AI_Y"),
    ("F_N", "LOW_I", "HIGH_B", "AI_N"),
    ("M_N", "TOP_I", "LOW_B", "AI_Y"),
    ("M_N", "TOP_I", "LOW_B", "AI_N"),
    ("M_N", "TOP_I", "HIGH_B", "AI_Y"),
    ("M_N", "TOP_I", "HIGH_B", "AI_N"),
    ("M_N", "LOW_I", "LOW_B", "AI_Y"),
    ("M_N", "LOW_I", "LOW_B", "AI_N"),
    ("M_N", "LOW_I", "HIGH_B", "AI_Y"),
    ("M_N", "LOW_I", "HIGH_B", "AI_N"),
]

PI_ONLY_CONDITIONS: List[Tuple[str, str, str]] = [
    ("F_N", "TOP_I", "LOW_B"),
    ("F_N", "TOP_I", "HIGH_B"),
    ("F_N", "LOW_I", "LOW_B"),
    ("F_N", "LOW_I", "HIGH_B"),
    ("M_N", "TOP_I", "LOW_B"),
    ("M_N", "TOP_I", "HIGH_B"),
    ("M_N", "LOW_I", "LOW_B"),
    ("M_N", "LOW_I", "HIGH_B"),
]

AI_ONLY_CONDITIONS: List[str] = ["AI_Y", "AI_N"]



def render_prompt(
    text_body: str,
    tokens: Dict[str, Any],
    pi_name_group: str,
    pi_inst_tier: str,
    pi_metric_level: str,
    ai_flag: str,
) -> Tuple[str, dict]:
    name_pi = tokens[f"{pi_name_group}_name_pi"]
    institution_pi = tokens[f"{pi_inst_tier}_inst_pi"]
    metric_obj = tokens[f"{pi_metric_level}_metrics_pi"]
    h_index_pi = metric_obj["h_index"]
    citations_pi = metric_obj["citations"]
    ai_usage = tokens[f"{ai_flag}_usage"]

    p = prompt_tpl
    p = p.replace("{text}", text_body)
    p = p.replace("{name_pi}", str(name_pi))
    p = p.replace("{institution_pi}", str(institution_pi))
    p = p.replace("{h_index_pi}", str(h_index_pi))
    p = p.replace("{citations_pi}", str(citations_pi))
    p = p.replace("{ai_usage}", str(ai_usage))

    meta = {
        "pi_name_group": pi_name_group,
        "pi_inst_tier": pi_inst_tier,
        "pi_metric_level": pi_metric_level,
        "ai_flag": ai_flag,
        "name_pi": name_pi,
        "institution_pi": institution_pi,
        "h_index_pi": h_index_pi,
        "citations_pi": citations_pi,
        "ai_usage": ai_usage,
    }
    return p, meta



def render_prompt_none(text_body: str) -> Tuple[str, dict]:
    p = prompt_noinfo_tpl.replace("{text}", text_body)

    meta = {
        "pi_name_group": "NONE",
        "pi_inst_tier": "NONE",
        "pi_metric_level": "NONE",
        "ai_flag": "NONE",
        "name_pi": "NONE",
        "institution_pi": "NONE",
        "h_index_pi": "NONE",
        "citations_pi": "NONE",
        "ai_usage": "NONE",
    }
    return p, meta



def render_prompt_pi_only(
    text_body: str,
    tokens: Dict[str, Any],
    pi_name_group: str,
    pi_inst_tier: str,
    pi_metric_level: str,
) -> Tuple[str, dict]:
    name_pi = tokens[f"{pi_name_group}_name_pi"]
    institution_pi = tokens[f"{pi_inst_tier}_inst_pi"]
    metric_obj = tokens[f"{pi_metric_level}_metrics_pi"]
    h_index_pi = metric_obj["h_index"]
    citations_pi = metric_obj["citations"]

    p = prompt_pi_only_tpl
    p = p.replace("{text}", text_body)
    p = p.replace("{name_pi}", str(name_pi))
    p = p.replace("{institution_pi}", str(institution_pi))
    p = p.replace("{h_index_pi}", str(h_index_pi))
    p = p.replace("{citations_pi}", str(citations_pi))

    meta = {
        "pi_name_group": pi_name_group,
        "pi_inst_tier": pi_inst_tier,
        "pi_metric_level": pi_metric_level,
        "ai_flag": "NONE",
        "name_pi": name_pi,
        "institution_pi": institution_pi,
        "h_index_pi": h_index_pi,
        "citations_pi": citations_pi,
        "ai_usage": "NONE",
    }
    return p, meta



def render_prompt_ai_only(
    text_body: str,
    tokens: Dict[str, Any],
    ai_flag: str,
) -> Tuple[str, dict]:
    ai_usage = tokens[f"{ai_flag}_usage"]

    p = prompt_ai_only_tpl
    p = p.replace("{text}", text_body)
    p = p.replace("{ai_usage}", str(ai_usage))

    meta = {
        "pi_name_group": "NONE",
        "pi_inst_tier": "NONE",
        "pi_metric_level": "NONE",
        "ai_flag": ai_flag,
        "name_pi": "NONE",
        "institution_pi": "NONE",
        "h_index_pi": "NONE",
        "citations_pi": "NONE",
        "ai_usage": ai_usage,
    }
    return p, meta


# ----------------------------
# Model init helpers
# ----------------------------

def init_model(spec: dict):
    backend = spec["backend"]

    if backend == "azure_openai":
        azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        azure_api_key = os.environ["AZURE_OPENAI_API_KEY"]
        client = openai.AzureOpenAI(
            api_key=azure_api_key,
            api_version="2024-10-21",
            azure_endpoint=azure_endpoint,
        )
        print("Azure OpenAI client ready.")
        return {"backend": backend, "client": client, "api_model": spec["api_model"]}
    
    if backend == "google_openai":
        google_api_key = os.environ["GEMINI_API_KEY"]
        client = openai.OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=google_api_key,
        )
        print("Google OpenAI client ready.")
        return {"backend": backend, "client": client, "api_model": spec["api_model"]}

    if backend == "cerebras_openai":
        cerebras_api_key = os.environ["CEREBRAS_API_KEY"]
        client = openai.OpenAI(
            base_url="https://api.cerebras.ai/v1",
            api_key=cerebras_api_key,
        )
        print("Cerebras OpenAI client ready.")
        return {"backend": backend, "client": client, "api_model": spec["api_model"]}

    if backend == "nvidia_openai":
        nvidia_api_key = os.environ["NVIDIA_API_KEY"]
        client = openai.OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key,
        )
        print("NVIDIA OpenAI-compatible client ready.")
        return {
            "backend": backend,
            "client": client,
            "api_model": spec["api_model"],
            "extra_body": spec.get("extra_body"),
        }

    if backend == "llama_cpp":
        gguf_path = hf_hub_download(
            repo_id=spec["model_id"],
            filename=spec["model_basename"],
            resume_download=True,
        )
        print(f"Model downloaded to: {gguf_path}")
        llm = Llama(
            model_path=gguf_path,
            n_gpu_layers=-1,
            n_ctx=16384,
            n_batch=512,
        )
        print("Model loading complete.")
        return {
            "backend": backend,
            "llm": llm,
            "prompt_style": spec["prompt_style"],
        }

    if backend == "ministral_transformers":
        token = os.environ["HF_TOKEN"]
        login(token=token)

        tokenizer = MistralCommonBackend.from_pretrained(spec["model_id"])
        llm = Mistral3ForConditionalGeneration.from_pretrained(
            spec["model_id"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "22GiB", 1: "22GiB"},
        )
        llm.eval()
        print("Model loading complete.")
        print("hf_device_map:", getattr(llm, "hf_device_map", None))

        input_device = torch.device("cuda:0")
        return {
            "backend": backend,
            "llm": llm,
            "tokenizer": tokenizer,
            "input_device": input_device,
        }

    raise ValueError(f"Unsupported backend: {backend}")


MODEL_RUNTIME = init_model(MODEL_SPEC)


# ----------------------------
# LLM call + parsing/validation
# ----------------------------

def normalize_q_keys(s: str) -> str:
    def repl(m):
        prefix = m.group(1)
        num = int(m.group(2))
        return f"{prefix}{num:02d}="
    return re.sub(r"\b([qQ])0*([1-6])=", repl, s)



def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^\s*</?think>\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()



def clean_output(text: str) -> str:
    t = (text or "").replace("\n", " ")
    t = t.replace("\\", "")
    t = t.strip()
    t = re.sub(r"\s*\|\s*", " | ", t)
    t = t.rstrip(" |")
    return t



def call_openai_compatible(
    prompt_text: str,
    client,
    api_model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    add_system_message: bool = False,
    extra_body: Optional[dict] = None,
) -> str:
    prompt = prompt_text
    print(f"[PROMPT]  Prepared.")

    messages = []
    if add_system_message:
        messages.append({"role": "system", "content": "Provide only the final output, no thinking."})
    messages.append({"role": "user", "content": prompt})

    while True:
        try:
            kwargs = {
                "model": api_model,
                "messages": messages,
                "max_completion_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            response = client.chat.completions.create(**kwargs)
            risposta = normalize_q_keys(clean_output(response.choices[0].message.content.strip()))
            break
        except Exception as e:
            print(f"Eccezione {e}. Retrying...")
            #print(api_model)
            sleep(10)
            continue

    return risposta



def call_llama_cpp(
    prompt_text: str,
    llm: Llama,
    prompt_style: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    if prompt_style == "llama":
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
  {prompt_text}
  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
  """
        stop = ["<|eot_id|>", "<|start_header_id|>"]
    elif prompt_style == "gemma":
        prompt = f"""<start_of_turn>user
{prompt_text}<end_of_turn>
<start_of_turn>model
"""
        stop = ["<end_of_turn>", "<start_of_turn>"]
    elif prompt_style == "qwen":
        prompt = f"""<|im_start|>user
  {prompt_text}<|im_end|>
  <|im_start|>assistant
  """
        stop = ["<|im_end|>", "<|im_start|>"]
    else:
        raise ValueError(f"Unsupported llama.cpp prompt_style: {prompt_style}")

    print(f"[PROMPT]  Prepared.")
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        echo=False,
        stream=False,
    )
    raw = output["choices"][0]["text"].strip()
    if prompt_style == "qwen":
        raw = strip_think(raw)
    return normalize_q_keys(clean_output(raw))



def call_ministral_transformers(
    prompt_text: str,
    llm: Mistral3ForConditionalGeneration,
    tokenizer: MistralCommonBackend,
    input_device,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    system_prompt = (
        "Before emitting the output, enforce that it has the correct format "
        "(pipe separated answers) and that there are exactly 6 answers "
        "(q01..q06) and no other keys. Emit everything on a single line, "
        "without line breaks. ONLY the answers, do NOT repeat the question "
        "or add any commentary. Format: q01=.. | q02=.. | ... | q06=.. "
        "(with q01..q06 as keys). For the scores, enforce they are among "
        "the available values: " + str(ALLOWED_SCORES)
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
        },
    ]

    tokenized = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
    )

    for k, v in list(tokenized.items()):
        if torch.is_tensor(v):
            tokenized[k] = v.to(input_device)

    do_sample = bool(temperature is not None and float(temperature) > 0)

    gen_kwargs = {
        **tokenized,
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output = llm.generate(**gen_kwargs)[0]

    prompt_len = tokenized["input_ids"].shape[1]
    raw = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
    raw = raw.strip()
    print(raw)
    return normalize_q_keys(clean_output(raw))



def questionario_LLM(
    prompt_text: str,
    runtime: dict,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    backend = runtime["backend"]

    print(f"[PROMPT] In evaluation:")
    #print("#"*50+"\n"+prompt_text+"\n"+"#"*50)
    #print(f"   Author Check: {"H-index" in prompt_text} ## GenAI Check: {"declares that generative AI" in prompt_text}")

    if backend == "azure_openai":
        return call_openai_compatible(
            prompt_text=prompt_text,
            client=runtime["client"],
            api_model=runtime["api_model"],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            add_system_message=False,
            extra_body=None,
        )

    if backend == "google_openai":
        return call_openai_compatible(
            prompt_text=prompt_text,
            client=runtime["client"],
            api_model=runtime["api_model"],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            add_system_message=False,
            extra_body=None,
        )

    if backend == "cerebras_openai":
        return call_openai_compatible(
            prompt_text=prompt_text,
            client=runtime["client"],
            api_model=runtime["api_model"],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            add_system_message=False,
            extra_body=None,
        )


    if backend == "nvidia_openai":
        return call_openai_compatible(
            prompt_text=prompt_text,
            client=runtime["client"],
            api_model=runtime["api_model"],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            add_system_message=True,
            extra_body=runtime.get("extra_body"),
        )

    if backend == "llama_cpp":
        return call_llama_cpp(
            prompt_text=prompt_text,
            llm=runtime["llm"],
            prompt_style=runtime["prompt_style"],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    if backend == "ministral_transformers":
        return call_ministral_transformers(
            prompt_text=prompt_text,
            llm=runtime["llm"],
            tokenizer=runtime["tokenizer"],
            input_device=runtime["input_device"],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unsupported runtime backend: {backend}")



def parse_llm_answers(risposta_str: str) -> Optional[dict]:
    d: Dict[str, str] = {}
    try:
        for item in risposta_str.split("|"):
            item = item.strip()
            qn, val = item.split("=", maxsplit=1)
            d[qn.replace("\n", "").replace(" ", "").strip()] = val.strip()
        return d
    except Exception:
        return None


EXPECTED_KEYS = [f"q{i:02d}" for i in range(1, 7)]
EXPECTED_LEN = len(EXPECTED_KEYS)
EXPECTED_SET = set(EXPECTED_KEYS)

score_re = re.compile(r"^\d+\.\d+$")



def allowed_value_checker(risposta_dict: dict) -> bool:
    if set(risposta_dict.keys()) != EXPECTED_SET:
        print(f"[DEBUG] Keys mismatch. Expected: {EXPECTED_SET}, Got: {set(risposta_dict.keys())}")
        return False

    for v in risposta_dict.values():
        s = str(v).strip()
        if not score_re.match(s):
            print(f"[DEBUG] Value '{v}' is not in decimal format like 2.5")
            return False
        if s not in ALLOWED_SCORES:
            print(f"[DEBUG] Value '{v}' is not one of allowed scores: {sorted(ALLOWED_SCORES)}")
            return False
    return True



def correct_structure_checker(answer_str: str) -> bool:
    parts = [p for p in answer_str.split("|") if p.strip()]
    return len(parts) == EXPECTED_LEN


# ----------------------------
# Main loop
# ----------------------------
rng = random.Random(42)
storie_items = list(dict_testi.items())
rng.shuffle(storie_items)

errori = 0
index = 0

#save config and prompt template in output dir for reference
with open(os.path.join(output_dir, "config_used.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

with open(os.path.join(output_dir, "prompts_used.txt"), "w", encoding="utf-8") as f:
    f.write("PROMPT_NOINFO:\n\n" + prompt_noinfo_tpl + "\n\n" + "=" * 80 + "\n\n")
    f.write("PROMPT_PI_ONLY:\n\n" + prompt_pi_only_tpl + "\n\n" + "=" * 80 + "\n\n")
    f.write("PROMPT_AI_ONLY:\n\n" + prompt_ai_only_tpl + "\n\n" + "=" * 80 + "\n\n")
    f.write("PROMPT_FULL:\n\n" + prompt_tpl + "\n\n" + "=" * 80 + "\n\n")

for storia, testoStoria in storie_items:
    tokens = build_text_level_tokens(storia)

    prompt_by_condition: Dict[str, str] = {}
    condition_meta: Dict[str, dict] = {}

    if run_mode == "full":
        for pi_name_group, pi_inst_tier, pi_metric_level, ai_flag in FULL_CONDITIONS:
            cond_id = f"PI+{pi_name_group}+{pi_inst_tier}+{pi_metric_level}+{ai_flag}"

            prompt_filled, meta = render_prompt(
                text_body=testoStoria,
                tokens=tokens,
                pi_name_group=pi_name_group,
                pi_inst_tier=pi_inst_tier,
                pi_metric_level=pi_metric_level,
                ai_flag=ai_flag,
            )

            prompt_by_condition[cond_id] = prompt_filled
            condition_meta[cond_id] = meta

    elif run_mode == "blind":
        prompt_none, meta_none = render_prompt_none(text_body=testoStoria)
        prompt_by_condition[NONE_COND_ID] = prompt_none
        condition_meta[NONE_COND_ID] = meta_none

    elif run_mode == "pi-only":
        for pi_name_group, pi_inst_tier, pi_metric_level in PI_ONLY_CONDITIONS:
            cond_id = f"PI+{pi_name_group}+{pi_inst_tier}+{pi_metric_level}"

            prompt_filled, meta = render_prompt_pi_only(
                text_body=testoStoria,
                tokens=tokens,
                pi_name_group=pi_name_group,
                pi_inst_tier=pi_inst_tier,
                pi_metric_level=pi_metric_level,
            )

            prompt_by_condition[cond_id] = prompt_filled
            condition_meta[cond_id] = meta

    elif run_mode == "ai-only":
        for ai_flag in AI_ONLY_CONDITIONS:
            cond_id = f"AI+{ai_flag}"

            prompt_filled, meta = render_prompt_ai_only(
                text_body=testoStoria,
                tokens=tokens,
                ai_flag=ai_flag,
            )

            prompt_by_condition[cond_id] = prompt_filled
            condition_meta[cond_id] = meta

    else:
        raise ValueError(f"Unsupported run_mode: {run_mode}")

    cond_items = list(prompt_by_condition.items())
    rng.shuffle(cond_items)

    for cond_id, full_prompt in cond_items:
        risposte_strutturate: List[dict] = []
        n_ok = 0
        n_try = 0

        json_path = os.path.join(json_out_dir, f"{safe_filename(storia)}__{safe_filename(cond_id)}.json")
        csv_path = os.path.join(csv_out_dir, f"{safe_filename(storia)}__{safe_filename(cond_id)}.csv")

        while n_ok < NUMBER_OF_ITERATIONS:
            n_try += 1
            meta = condition_meta[cond_id]

            print("\n" + "=" * 80)
            print(
                f"[RUN] {model_path_name} | "
                f"text={storia} | len={len(testoStoria)} | mode={run_mode} | condition={cond_id} | "
                f"ok={n_ok}/{NUMBER_OF_ITERATIONS} | try={n_try} | index={index}"
            )

            if cond_id == NONE_COND_ID:
                print("[COND] NONE (no PI profile / no GenAI disclosure)")
            else:
                print(
                    "[PI] "
                    f"name={meta['name_pi']} | institution={meta['institution_pi']} | "
                    f"h_index={meta['h_index_pi']} | citations={meta['citations_pi']} | "
                    f"ai_usage={meta['ai_usage']}"
                )

            text_preview = testoStoria[0:120].replace("\n", " ")
            print(f"[TEXT] {text_preview} ...")

            risposta_str = questionario_LLM(
                full_prompt,
                MODEL_RUNTIME,
                TEMPERATURE,
                TOP_P,
                MAX_TOKENS,
            )
            print(f"[LLM]  {risposta_str}")

            if not correct_structure_checker(risposta_str):
                errori += 1
                print("[ERROR01] Wrong pipe/field count.")
                continue

            risposta_dict = parse_llm_answers(risposta_str)
            if risposta_dict is None:
                errori += 1
                print("[ERROR02] Parsing failed.")
                continue

            if not allowed_value_checker(risposta_dict):
                errori += 1
                print("[ERROR03] Values out of range or missing/extra keys.")
                continue

            print("[VALID] Answer accepted.")
            rec = {
                **risposta_dict,
                "text_id": storia,
                "condition_id": cond_id,
                "iteration": n_ok,
                "model": model_path_name,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
                "run_mode": run_mode,
                **meta,
            }
            risposte_strutturate.append(rec)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(risposte_strutturate, f, ensure_ascii=False, indent=2)

            df = pd.DataFrame(risposte_strutturate)
            df.to_csv(csv_path, index=False, encoding="utf-8")

            n_ok += 1
        index += 1

print("\n" + "=" * 80)
print(f"Finished. Total accepted outputs: {index} | total errors: {errori}")
