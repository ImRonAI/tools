"""Unified tool catalog: discover, load, execute, and unload tools."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from strands import tool, ToolContext

from strands_tools.tool_catalog_manager import get_tool_catalog_manager

logger = logging.getLogger(__name__)


def _load_tool(agent: Any, name: str, path: str) -> None:
    """Load a tool into the agent using process_tools (non-deprecated SDK pattern)."""
    agent.tool_registry.process_tools([path])


def _unload_tool(agent: Any, name: str) -> None:
    """Remove a tool from the agent's registry."""
    registry = agent.tool_registry
    registry.registry.pop(name, None)
    if hasattr(registry, "dynamic_tools") and isinstance(registry.dynamic_tools, dict):
        registry.dynamic_tools.pop(name, None)
    # Invalidate cached tool config so next call rebuilds it
    if hasattr(registry, "tool_config"):
        registry.tool_config = None


def _execute_one(agent: Any, name: str, arguments: Dict[str, Any], path: str) -> Dict[str, Any]:
    """Fire-and-forget: load tool, invoke it, unload it, return result."""
    try:
        _load_tool(agent, name, path)
        caller = getattr(agent.tool, name)
        result = caller(record_direct_tool_call=False, **arguments)
        return {"name": name, "result": result}
    except Exception as exc:
        return {"name": name, "error": str(exc)}
    finally:
        _unload_tool(agent, name)


@tool(context=True)
def tool_catalog(
    tool_context: ToolContext,
    action: str,
    name: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Unified tool lifecycle interface: discover, inspect, load, execute, and unload tools.

    Actions:
      - list_categories: returns all categories with tool names, descriptions, and parameter summaries.
      - get_tool: returns full details for a tool by name (schema, path, description).
      - execute: fire-and-forget execution. Loads the tool, runs it, unloads it.
        For a single tool: provide name and arguments.
        For parallel execution: provide tools (list of {name, arguments}).
      - load: load a tool into the agent for repeated use. Provide name.
      - unload: remove a previously loaded tool from the agent. Provide name.
    """
    catalog = get_tool_catalog_manager()
    agent = tool_context.agent

    # --- discover ---
    if action == "list_categories":
        return {"status": "success", "content": [{"json": catalog.build_catalog_overview()}]}

    if action == "get_tool":
        if not name:
            return {"status": "error", "content": [{"text": "name is required for get_tool"}]}
        details = catalog.get_tool_details(name)
        if not details:
            return {"status": "error", "content": [{"text": f"Tool not found: {name}"}]}
        return {"status": "success", "content": [{"json": details}]}

    # --- execute (fire-and-forget, single or parallel) ---
    if action == "execute":
        # Build list of tool invocations
        invocations: List[Dict[str, Any]] = []
        if tools and isinstance(tools, list):
            invocations = tools
        elif name:
            invocations = [{"name": name, "arguments": arguments or {}}]
        else:
            return {"status": "error", "content": [{"text": "execute requires name or tools list"}]}

        # Resolve paths from catalog
        resolved: List[Dict[str, Any]] = []
        for inv in invocations:
            tool_name = inv.get("name", "")
            details = catalog.get_tool_details(tool_name)
            if not details or not details.get("path"):
                return {"status": "error", "content": [{"text": f"No path found for tool: {tool_name}"}]}
            resolved.append({
                "name": tool_name,
                "arguments": inv.get("arguments") or {},
                "path": details["path"],
            })

        # Single tool: run directly
        if len(resolved) == 1:
            r = resolved[0]
            result = _execute_one(agent, r["name"], r["arguments"], r["path"])
            if "error" in result:
                return {"status": "error", "content": [{"text": f"{r['name']}: {result['error']}"}]}
            return {"status": "success", "content": [{"json": result["result"]}]}

        # Parallel execution
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(len(resolved), 4)) as pool:
            futures = {
                pool.submit(_execute_one, agent, r["name"], r["arguments"], r["path"]): r["name"]
                for r in resolved
            }
            for future in as_completed(futures):
                results.append(future.result())

        return {"status": "success", "content": [{"json": results}]}

    # --- load (for repeated use) ---
    if action == "load":
        if not name:
            return {"status": "error", "content": [{"text": "name is required for load"}]}
        details = catalog.get_tool_details(name)
        if not details or not details.get("path"):
            return {"status": "error", "content": [{"text": f"No path found for tool: {name}"}]}
        try:
            _load_tool(agent, name, details["path"])
            return {"status": "success", "content": [{"text": f"Loaded tool: {name}"}]}
        except Exception as exc:
            return {"status": "error", "content": [{"text": f"Failed to load {name}: {exc}"}]}

    # --- unload ---
    if action == "unload":
        if not name:
            return {"status": "error", "content": [{"text": "name is required for unload"}]}
        try:
            _unload_tool(agent, name)
            return {"status": "success", "content": [{"text": f"Unloaded tool: {name}"}]}
        except Exception as exc:
            return {"status": "error", "content": [{"text": f"Failed to unload {name}: {exc}"}]}

    return {"status": "error", "content": [{"text": f"Unsupported action: {action}"}]}
