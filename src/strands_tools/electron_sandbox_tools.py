
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from strands import tool
from .browser.browser import Browser
from .browser.models import EvaluateAction

logger = logging.getLogger(__name__)

class ElectronSandboxTools:
    """
    Wraps Electron's sandboxed IPC methods as Strands tools.
    Requires a connected Browser instance.
    """
    
    def __init__(self, browser: Browser):
        self.browser = browser

    async def _call_electron(self, method: str, args: List[Any]) -> Dict[str, Any]:
        """
        Calls window.electron.sandbox.<method>(...args) via browser.evaluate.
        """
        # Serialize args to JSON
        json_args = json.dumps(args)
        session_name = self.browser.get_default_session_name() or "default"
        try:
            # Use JSON.stringify in the browser context to keep parsing deterministic.
            script_json = f"""
                (async () => {{
                    const args = {json_args};
                    const res = await window.electron.sandbox.{method}(...args);
                    return JSON.stringify(res);
                }})()
            """
            
            result_json = await self.browser.evaluate(EvaluateAction(
                type="evaluate",
                script=script_json,
                session_name=session_name
            ))
            
            if result_json.get("status") != "success":
                return {
                    "success": False,
                    "error": result_json.get("content", [{}])[0].get("text", "Evaluation failed")
                }

            text = result_json["content"][0]["text"]
            if text.startswith("Evaluation result: "):
                raw_json = text[len("Evaluation result: "):]
                # If the result is a string (which is JSON), allow parsing it
                # It might be double-quoted if Playwright returned a string.
                # e.g. '"{\\"success\\":true}"'
                try:
                    # First try direct parse
                    data = json.loads(raw_json)
                    # If data is string, parse again
                    if isinstance(data, str):
                        data = json.loads(data)
                    return data
                except json.JSONDecodeError:
                    return {"success": False, "error": f"Failed to parse result: {raw_json}"}
            
            return {"success": False, "error": f"Unexpected response format: {text}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    @tool
    async def shell(
        self,
        command: Union[str, List[Union[str, Dict[str, Any]]]],
        parallel: bool = False,
        ignore_errors: bool = False,
        timeout: int = 30000,
        work_dir: str = None, # API signature match, but ignored/enforced by sandbox
        non_interactive: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute shell commands in the secure agent sandbox.
        """
        # Normalize command
        cmds = command if isinstance(command, list) else [command]
        
        results = []
        for cmd_item in cmds:
            cmd_str = cmd_item if isinstance(cmd_item, str) else cmd_item.get("command")
            if not cmd_str: continue
            
            # Call Electron
            # signature: shell(command, args, options)
            # We treat the whole string as command + args for simplicity in this wrapper?
            # Or we let Electron spawn handle it with shell=True.
            
            resp = await self._call_electron("shell", [cmd_str, [], {"timeout": timeout}])
            
            results.append({
                "command": cmd_str,
                "status": "success" if resp.get("success") else "error",
                "output": resp.get("stdout", "") + resp.get("stderr", ""),
                "exit_code": resp.get("exitCode", -1),
                "error": resp.get("error")
            })
            
            if not ignore_errors and not resp.get("success"):
                break

        # Format output similar to original shell tool
        return {
            "status": "success" if all(r["status"] == "success" for r in results) else "error",
            "content": [{"json": results}]
        }

    @tool
    async def file_read(
        self,
        path: str,
        mode: str = "read", # Only "read" supported for now in sandbox
        start_line: int = 0,
        end_line: int = None
    ) -> Dict[str, Any]:
        """
        Read files from the secure agent sandbox.
        """
        resp = await self._call_electron("readFile", [path])
        
        if not resp.get("success"):
            return {
                "status": "error", 
                "content": [{"text": f"Error reading {path}: {resp.get('error')}"}]
            }
        
        content = resp.get("content", "")
        
        # Handle lines mode
        if start_line > 0 or end_line is not None:
             lines = content.split('\n')
             end = end_line if end_line is not None else len(lines)
             content = '\n'.join(lines[start_line:end])
             
        return {
            "status": "success",
            "content": [{"text": content}]
        }

    @tool
    async def file_write(
        self,
        path: str,
        content: str,
        mode: str = "write" # write vs append? Electron API implies overwrite currently
    ) -> Dict[str, Any]:
        """
        Write files to the secure agent sandbox.
        """
        # TODO: Implement append in Electron if needed. For now assume overwrite.
        
        resp = await self._call_electron("writeFile", [path, content])
        
        if not resp.get("success"):
             return {
                "status": "error", 
                "content": [{"text": f"Error writing {path}: {resp.get('error')}"}]
            }
            
        return {
            "status": "success",
            "content": [{"text": f"Successfully wrote to {path}"}]
        }
        
    @tool
    async def list_files(
        self,
        path: str = "."
    ) -> Dict[str, Any]:
        """
        List files in the secure agent sandbox.
        """
        resp = await self._call_electron("listFiles", [path])
        
        if not resp.get("success"):
             return {
                "status": "error", 
                "content": [{"text": f"Error listing {path}: {resp.get('error')}"}]
            }
            
        return {
            "status": "success",
            "content": [{"json": resp.get("files", [])}]
        }

    @tool
    async def editor(
        self,
        command: str,
        path: str,
        file_text: str = None,
        view_range: List[int] = None,
        old_str: str = None,
        new_str: str = None,
        insert_line: int = None
    ) -> Dict[str, Any]:
        """
        Edit files in the secure agent sandbox.
        Commands:
        - view: Read file content
        - create: Create new file with content
        - str_replace: Replace string in file
        - insert: Insert line (not fully impl)
        """
        if command == "view":
            start = view_range[0] if view_range else 0
            end = view_range[1] if view_range and len(view_range) > 1 else None
            return await self.file_read(path, start_line=start, end_line=end)
            
        elif command == "create":
            return await self.file_write(path, file_text)
            
        elif command == "str_replace":
            # Read, replace, write
            read_res = await self.file_read(path)
            if read_res["status"] == "error": return read_res
            
            content = read_res["content"][0]["text"]
            if old_str not in content:
                return {"status": "error", "content": [{"text": f"String '{old_str}' not found in {path}"}]}
                
            new_content = content.replace(old_str, new_str)
            return await self.file_write(path, new_content)
            
        return {"status": "error", "content": [{"text": f"Command {command} not supported in sandbox"}]}
