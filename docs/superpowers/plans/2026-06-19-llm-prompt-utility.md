# LLM Prompt Utility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a general-purpose `call_llm()` function using langchain that sends prompts and returns results with thinking process.

**Architecture:** Single-file Python module with langchain's ChatOpenAI for OpenAI-compatible APIs. .env config loaded via python-dotenv.

**Tech Stack:** Python, langchain-openai, python-dotenv

---

### Task 1: Create requirements.txt

**Files:**
- Create: `llm_prompt/requirements.txt`

- [ ] **Create requirements.txt**

```text
langchain-openai>=0.3.0
python-dotenv>=1.0.0
```

- [ ] **Commit**

```bash
git add llm_prompt/requirements.txt
git commit -m "add: requirements.txt for llm-prompt utility"
```

### Task 2: Implement call_llm and main

**Files:**
- Create: `llm_prompt/llm_prompt.py`

- [ ] **Create llm_prompt.py**

```python
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def call_llm(prompt: str, enable_thinking: bool = True, **kwargs) -> dict:
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    model = os.getenv("MODEL")

    if not api_key:
        raise ValueError("API_KEY not found in .env")
    if not model:
        raise ValueError("MODEL not found in .env")

    llm_kwargs = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url if base_url else None,
        **kwargs,
    }

    if enable_thinking:
        llm_kwargs["extra_body"] = {"thinking": {}}

    llm = ChatOpenAI(**llm_kwargs)
    response = llm.invoke(prompt)

    content = response.content
    thinking = None

    reasoning = response.additional_kwargs.get("reasoning_content")
    if reasoning:
        thinking = reasoning

    if not thinking:
        reasoning = response.response_metadata.get("reasoning_content")
        if reasoning:
            thinking = reasoning

    return {"content": content, "thinking": thinking}


def main():
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    enable_thinking = True
    prompt_args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if "--no-thinking" in sys.argv:
        enable_thinking = False

    prompt = " ".join(prompt_args) if prompt_args else "请简要介绍你自己"

    print(f"Prompt: {prompt}")
    print(f"Thinking mode: {'ON' if enable_thinking else 'OFF'}")
    print("-" * 40)

    result = call_llm(prompt, enable_thinking=enable_thinking)

    if result["thinking"]:
        print("\n=== THINKING ===")
        print(result["thinking"])

    print("\n=== CONTENT ===")
    print(result["content"])


if __name__ == "__main__":
    main()
```

- [ ] **Commit**

```bash
git add llm_prompt/llm_prompt.py
git commit -m "add: call_llm function with thinking mode support"
```

### Task 3: Install dependencies and verify

- [ ] **Install dependencies**

```bash
pip install -r llm_prompt/requirements.txt
```

- [ ] **Run smoke test**

```bash
cd /root/projects/llm_prompt && python -c "from llm_prompt import call_llm; print('import OK')"
```

Expected: `import OK`

- [ ] **Run with actual LLM call**

```bash
cd /root/projects/llm_prompt && python -m llm_prompt "hello"
```

Expected: Prints thinking and content from the model.
