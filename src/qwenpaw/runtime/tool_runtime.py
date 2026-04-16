import json
import logging

MAX_STEPS = 6

logger = logging.getLogger(__name__)


# ===== 工具注册 =====
def run_tool(name: str, args: dict):
    try:
        from qwenpaw.app.server.tools_registry import get_tool_manager
        tm = get_tool_manager()
        
        tool = tm.get_tool(name)
        if tool is None:
            return f"[ERROR] Unknown tool: {name}"
        
        result = tool.run(**args)
        return str(result)
    
    except Exception as e:
        return f"[ERROR] {str(e)}"


# ===== JSON解析 =====
def try_parse_json(text: str):
    text = text.strip()

    # 去掉 markdown code block
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
        text = text.strip()

    try:
        return json.loads(text)
    except:
        return None


# ===== 工具处理 =====
def handle_tool_call(resp_content, result, messages):
    # messages are already in dict format for the model call
    messages.append({
        "role": "assistant",
        "content": resp_content
    })
    
    messages.append({
        "role": "tool",
        "content": str(result)
    })
    
    return messages


# ===== Agent Loop =====
async def run_agent_loop(model, messages):
    for step in range(MAX_STEPS):
        logger.debug(f"[vLLM agent loop] step {step+1}/{MAX_STEPS}")
        
        resp = await model(messages=messages)
        
        # 兼容 dict / str / ChatResponse / Msg
        if isinstance(resp, dict):
            content = resp.get("content", "")
        elif hasattr(resp, 'text'):
            content = resp.text
        elif hasattr(resp, 'content'):
            content = str(resp.content)
        else:
            content = str(resp)
        
        logger.debug(f"[vLLM agent loop] model output: {repr(content[:500])}")
        print(f"\n[MODEL OUTPUT]: {content[:500]}...")
        
        data = try_parse_json(content)
        
        # Tool call
        if data and data.get("action") == "tool":
            tool_name = data.get("name")
            args = data.get("args", {})
            
            logger.info(f"[vLLM agent loop] calling tool: {tool_name}, args: {args}")
            result = run_tool(tool_name, args)
            
            updated = handle_tool_call(content, result, messages)
            if updated:
                messages = updated
                continue
        
        # Final answer
        if data and data.get("action") == "final":
            answer = data.get("answer", "")
            logger.info(f"[vLLM agent loop] final answer: {repr(answer[:200])}")
            return answer
        
        # Fallback: return as plain text
        logger.debug(f"[vLLM agent loop] fallback to plain text")
        return content
    
    # Max steps reached
    logger.warning(f"[vLLM agent loop] max steps ({MAX_STEPS}) reached")
    return f"[ERROR] Agent stopped: maximum {MAX_STEPS} steps reached"
