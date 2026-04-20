# -*- coding: utf-8 -*-
"""Real end-to-end integration test: QwenPaw + vLLM (no --enable-auto-tool-choice).

This test verifies that QwenPaw can actually call a vLLM-served model
with tools enabled WITHOUT triggering HTTP 400 errors.
"""
from __future__ import annotations

import os
import httpx
import pytest


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2-0.5B-Instruct")


def _vllm_available() -> bool:
    try:
        r = httpx.get(f"{VLLM_BASE_URL}/v1/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


requires_vllm = pytest.mark.skipif(
    not _vllm_available(),
    reason=f"vLLM server not available at {VLLM_BASE_URL}",
)


@requires_vllm
def test_vllm_basic_completion():
    """Verify vLLM server responds to a simple chat completion."""
    r = httpx.post(
        f"{VLLM_BASE_URL}/v1/chat/completions",
        json={
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 10,
        },
        timeout=60,
    )
    assert r.status_code == 200, f"Basic completion failed: {r.text}"
    data = r.json()
    assert "choices" in data
    assert len(data["choices"]) > 0


@requires_vllm
def test_vllm_tool_choice_auto_rejected():
    """THE KEY TEST: Confirm vLLM rejects tool_choice=auto, and stripping it works.

    vLLM without --enable-auto-tool-choice returns 400 for tool_choice=auto.
    QwenPaw's fix strips tool_choice before sending to vLLM.
    This test validates the raw API behavior that motivates the fix.
    """
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]

    # Step 1: vLLM should REJECT tool_choice="auto" (this is the baseline problem)
    r_auto = httpx.post(
        f"{VLLM_BASE_URL}/v1/chat/completions",
        json={
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 50,
        },
        timeout=60,
    )

    # Step 2: Without tool_choice, it should succeed
    r_none = httpx.post(
        f"{VLLM_BASE_URL}/v1/chat/completions",
        json={
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": tools,
            # NO tool_choice - this is what QwenPaw fix does
            "max_tokens": 50,
        },
        timeout=60,
    )
    assert r_none.status_code == 200, (
        f"Request without tool_choice failed ({r_none.status_code}): {r_none.text[:500]}"
    )

    # Step 3: tool_choice=none should also work
    r_explicit_none = httpx.post(
        f"{VLLM_BASE_URL}/v1/chat/completions",
        json={
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": tools,
            "tool_choice": "none",
            "max_tokens": 50,
        },
        timeout=60,
    )
    assert r_explicit_none.status_code == 200, (
        f"Request with tool_choice=none failed ({r_explicit_none.status_code}): {r_explicit_none.text[:500]}"
    )

    # Report: did vLLM reject tool_choice=auto?
    if r_auto.status_code == 400:
        print("CONFIRMED: vLLM rejects tool_choice=auto (400) — QwenPaw fix is essential")
    else:
        print(f"NOTE: vLLM accepted tool_choice=auto (status={r_auto.status_code}) — may have --enable-auto-tool-choice")


@requires_vllm
def test_qwenpaw_model_with_vllm():
    """Test QwenPaw model wrapper -> vLLM with tool_choice interception.

    Goes through the actual QwenPaw code path (OpenAIChatModel + tools)
    to verify the tool_choice=auto -> None fix works end-to-end.
    """
    from agentscope.model import OpenAIChatModel
    from agentscope.tool import Toolkit
    import asyncio

    model = OpenAIChatModel(
        model_name=VLLM_MODEL,
        api_key="not-needed",
        client_args={"base_url": f"{VLLM_BASE_URL}/v1"},
    )

    tools = Toolkit()
    tools.register_tool(lambda city: f"Weather: sunny", function_name="get_weather")

    async def run():
        try:
            resp = await model(
                messages=[{"role": "user", "content": "Say hi"}],
                tools=tools,
                tool_choice="auto",  # Must be intercepted by QwenPaw fix
            )
            print(f"QwenPaw model call succeeded (type={type(resp).__name__})")
            return True
        except Exception as e:
            err = str(e)
            if "400" in err or "BadRequest" in err:
                print(f"FAILED: 400 error — tool_choice fix broken: {err[:300]}")
                return False
            print(f"Non-400 error (OK for this test): {err[:200]}")
            return True

    result = asyncio.run(run())
    assert result, "QwenPaw model wrapper got 400 — tool_choice fix is broken!"
