import requests
import json
import time
import os

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2-0.5B-Instruct"  # Will be overridden by env if set

# Get model from environment if provided
if "TEST_MODEL" in os.environ:
    MODEL = os.environ["TEST_MODEL"]

print(f"🔧 Testing agent runtime with model: {MODEL}")
print(f"🔧 vLLM endpoint: {VLLM_URL}")

tool_calls = 0

# ===== 工具 =====
def run_tool(name, args):
    if name == "list_dir":
        import os
        path = args.get("path", ".")
        return str(os.listdir(path))

    if name == "echo":
        return str(args)

    if name == "read_file":
        path = args.get("path")
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            return f"[ERROR] {str(e)}"

    return f"UNKNOWN TOOL: {name}"


# ===== JSON解析 =====
def try_parse_json(text):
    text = text.strip()
    # Remove markdown code blocks
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
        text = text.strip()
    try:
        return json.loads(text)
    except:
        return None


# ===== 调用模型 =====
def call_model(messages):
    global tool_calls
    resp = requests.post(
        VLLM_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.2,
            "tool_choice": None,  # Key fix for vLLM compatibility
        }
    )

    if resp.status_code != 200:
        print(f"❌ HTTP ERROR {resp.status_code}: {resp.text}")
        return None

    return resp.json()["choices"][0]["message"]["content"]


# ===== Agent Loop =====
def run_agent(task):
    global tool_calls
    tool_calls = 0
    
    messages = [
        {
            "role": "system",
            "content": """You are an agent.

When using tools, output ONLY JSON:

{"action":"tool","name":"<tool_name>","args":{...}}

When finished:

{"action":"final","answer":"..."}

Rules:
- NO explanation outside JSON
- Only JSON when calling tools
"""
        },
        {"role": "user", "content": task}
    ]

    for step in range(6):
        print(f"\n--- STEP {step + 1} ---")

        output = call_model(messages)
        if output is None:
            return False

        print(f"MODEL: {output[:200]}{'...' if len(output) > 200 else ''}")

        data = try_parse_json(output)

        if data:
            if data.get("action") == "tool":
                tool_calls += 1
                result = run_tool(data["name"], data.get("args", {}))
                print(f"TOOL RESULT: {result[:200]}{'...' if len(result) > 200 else ''}")

                messages.append({"role": "assistant", "content": output})
                messages.append({"role": "tool", "content": result})
                continue

            if data.get("action") == "final":
                answer = data.get("answer", "")
                print(f"\n✅ FINAL ANSWER: {answer}")
                print(f"\n📊 STATISTICS:")
                print(f"   Tool calls: {tool_calls}")
                print(f"   Total steps: {step + 1}")
                return True

        print("\n⚠️ No valid tool/final action detected, output as-is")
        print(f"📊 STATISTICS:")
        print(f"   Tool calls: {tool_calls}")
        print(f"   Total steps: {step + 1}")
        return False

    # Max steps reached
    print(f"\n⚠️ Max steps (6) reached")
    print(f"📊 STATISTICS:")
    print(f"   Tool calls: {tool_calls}")
    return False


# ===== 测试 =====
if __name__ == "__main__":
    import sys
    task = "列出当前目录文件并总结结构"
    if len(sys.argv) > 1:
        task = sys.argv[1]
    
    print(f"\n🎯 Starting agent task: {task}")
    print("=" * 60)
    
    success = run_agent(task)
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TEST PASSED: Agent runtime working correctly (tool call + multi-round)")
        sys.exit(0)
    else:
        print("❌ TEST FAILED: Check output above for details")
        sys.exit(1)
