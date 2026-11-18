import json, os, re, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROVIDERS = [
    {"name": "groq", "base_url": "https://api.groq.com/openai/v1",
     "key_env": "GROQ_API_KEY", "model": "openai/gpt-oss-120b"},
    {"name": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
     "key_env": "GEMINI_API_KEY", "model": "gemini-2.5-flash"},
]

def chat(messages, json_mode=False, temperature=0.3, max_tokens=1500) -> str:
    last_err = None
    for p in PROVIDERS:
        key = os.getenv(p["key_env"])
        if not key:
            continue
        client = OpenAI(api_key=key, base_url=p["base_url"])
        for attempt in range(2):
            try:
                kw = dict(model=p["model"], messages=messages,
                          temperature=temperature, max_tokens=max_tokens)
                if json_mode:
                    kw["response_format"] = {"type": "json_object"}
                r = client.chat.completions.create(**kw)
                return r.choices[0].message.content or ""
            except Exception as e:  # rate limit, network, etc.
                last_err = e
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"No LLM provider worked (is a key set in .env?): {last_err}")

def chat_json(messages, **kw):
    txt = chat(messages, json_mode=True, **kw).strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    return json.loads(txt)