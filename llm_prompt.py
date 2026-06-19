import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
)


def call_llm(prompt: str, enable_thinking: bool = True) -> dict:
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    model = os.getenv("MODEL")

    if not api_key:
        raise ValueError("API_KEY not found in .env")
    if not model:
        raise ValueError("MODEL not found in .env")

    extra_body = {}
    if enable_thinking:
        extra_body = {"thinking": {"type": "enabled"}}
    else:
        extra_body = {"thinking": {"type": "disabled"}}

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        raw = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            extra_body=extra_body,
            reasoning_effort="high" if enable_thinking else None,
        )
    except AuthenticationError:
        raise RuntimeError("API_KEY invalid or expired, please check .env")
    except BadRequestError as e:
        raise RuntimeError(f"Request error: {e.message}")
    except RateLimitError:
        raise RuntimeError("Rate limit exceeded, please try later")
    except APIConnectionError:
        raise RuntimeError(
            f"Connection failed, check BASE_URL ({base_url}) and network"
        )
    except APIError as e:
        raise RuntimeError(f"API error: {e.message}")

    try:
        choice = raw.choices[0]
        content = choice.message.content
        thinking = getattr(choice.message, "reasoning_content", None)
    except (AttributeError, IndexError, TypeError):
        raise RuntimeError("Unexpected API response format")

    return {"content": content, "thinking": thinking}


def main():
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
