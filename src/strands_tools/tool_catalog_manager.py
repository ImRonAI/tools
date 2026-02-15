"""
Tool catalog management for Strands tools.

Maintains a JSON catalog (and optional Markdown summary) of available tools,
including built-in, dynamically loaded, and MCP-provided tools.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

CATALOG_SCHEMA_VERSION = 2
DEFAULT_CATALOG_FILENAME = "tool_catalog.json"
DEFAULT_MARKDOWN_SUFFIX = ".md"

ENV_CATALOG_PATH = "STRANDS_TOOL_CATALOG_PATH"
ENV_CATALOG_DIR = "STRANDS_TOOL_CATALOG_DIR"
ENV_WRITE_MARKDOWN = "STRANDS_TOOL_CATALOG_MARKDOWN"
ENV_DEFAULT_SANDBOX_STATUS = "STRANDS_TOOL_SANDBOX_STATUS"
ENV_DISCOVERY_MANIFEST = "STRANDS_TOOL_DISCOVERY_MANIFEST"
ENV_CATEGORY_ORDER = "STRANDS_TOOL_CATEGORY_ORDER"
ENV_SANDBOX_ROOT = "RON_AGENT_SANDBOX_ROOT"
ENV_OVERVIEW_CACHE_TTL = "STRANDS_TOOL_OVERVIEW_CACHE_TTL"

DEFAULT_OVERVIEW_CACHE_TTL_SECONDS = 30.0

DEFAULT_CATEGORY_ORDER = [
    "built_in",
    "dynamically_loaded",
    "mcp_tools",
    "strands_tools",
    "strands_fun_tools",
    "strands_google",
    "fda",
    "pubmed",
    "perplexity",
    "custom",
    "mcp_servers",
    "openapi_specs",
]

CATEGORY_LABELS = {
    "built_in": "Built-in Tools",
    "dynamically_loaded": "Dynamically Loaded Tools",
    "mcp_tools": "MCP Tools",
    "strands_tools": "Strands Tools",
    "strands_fun_tools": "Strands Fun Tools",
    "strands_google": "Strands Google Tools",
    "fda": "FDA Tools",
    "pubmed": "PubMed Tools",
    "perplexity": "Perplexity Tools",
    "custom": "Custom Tools",
    "mcp_servers": "MCP Servers",
    "openapi_specs": "OpenAPI Specs",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_sandbox_root() -> Path:
    if platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    elif platform.system() == "Windows":
        base = Path(os.getenv("APPDATA", Path.home()))
    else:
        base = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "RonBrowser" / "agent-sandbox"


def _resolve_sandbox_root() -> Path:
    env_root = os.getenv(ENV_SANDBOX_ROOT)
    if env_root:
        return Path(env_root).expanduser()
    return _default_sandbox_root()


def _resolve_catalog_path() -> Path:
    path_env = os.getenv(ENV_CATALOG_PATH)
    if path_env:
        return Path(path_env).expanduser()
    dir_env = os.getenv(ENV_CATALOG_DIR)
    if dir_env:
        base_dir = Path(dir_env).expanduser()
    else:
        sandbox_root = _resolve_sandbox_root()
        try:
            sandbox_root.mkdir(parents=True, exist_ok=True)
            base_dir = sandbox_root
        except Exception:
            base_dir = sandbox_root if sandbox_root.exists() else Path.cwd()
    return base_dir / DEFAULT_CATALOG_FILENAME


def _resolve_markdown_path(catalog_path: Path) -> Path:
    return catalog_path.with_suffix(DEFAULT_MARKDOWN_SUFFIX)


def _default_sandbox_status() -> str:
    return os.getenv(ENV_DEFAULT_SANDBOX_STATUS, "Electron UtilityProcess")


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _compute_schema_hash(schema: Dict[str, Any]) -> Optional[str]:
    """Compute a stable hash of the schema for lightweight comparison."""
    if not schema:
        return None
    try:
        return hashlib.md5(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:12]
    except (TypeError, ValueError):
        return None


def _extract_tool_name(tool_obj: Any) -> str:
    if tool_obj is None:
        return ""

    tool_spec_attr = getattr(tool_obj, "TOOL_SPEC", None)
    if isinstance(tool_spec_attr, dict):
        spec_name = tool_spec_attr.get("name")
        if isinstance(spec_name, str) and spec_name:
            return spec_name

    name = getattr(tool_obj, "tool_name", None)
    if isinstance(name, str) and name:
        return name

    spec = getattr(tool_obj, "tool_spec", None)
    if spec:
        if isinstance(spec, dict):
            spec_name = spec.get("name")
            if isinstance(spec_name, str) and spec_name:
                return spec_name
        else:
            spec_name = getattr(spec, "name", None)
            if isinstance(spec_name, str) and spec_name:
                return spec_name

    fallback = getattr(tool_obj, "__name__", None)
    if isinstance(fallback, str) and fallback:
        return fallback

    return tool_obj.__class__.__name__


def _extract_tool_description(tool_obj: Any) -> str:
    tool_spec_attr = getattr(tool_obj, "TOOL_SPEC", None)
    if isinstance(tool_spec_attr, dict):
        desc = tool_spec_attr.get("description")
        if isinstance(desc, str) and desc.strip():
            return _first_line(desc)

    spec = getattr(tool_obj, "tool_spec", None)
    if spec:
        if isinstance(spec, dict):
            desc = spec.get("description")
            if isinstance(desc, str) and desc.strip():
                return _first_line(desc)
        else:
            desc = getattr(spec, "description", None)
            if isinstance(desc, str) and desc.strip():
                return _first_line(desc)

    doc = inspect.getdoc(tool_obj)
    if not doc and hasattr(tool_obj, "__call__"):
        doc = inspect.getdoc(tool_obj.__call__)
    return _first_line(doc or "")


def _extract_tool_path(tool_obj: Any) -> Optional[str]:
    if tool_obj is None:
        return None

    candidates = []
    try:
        candidates.append(inspect.getsourcefile(tool_obj))
    except Exception:
        pass

    if hasattr(tool_obj, "__call__"):
        try:
            candidates.append(inspect.getsourcefile(tool_obj.__call__))
        except Exception:
            pass

    try:
        module = inspect.getmodule(tool_obj)
        if module is not None:
            candidates.append(getattr(module, "__file__", None))
    except Exception:
        pass

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return str(Path(candidate).resolve())
        except Exception:
            return str(candidate)

    return None


def _default_execute_pathway(name: str, load_path: Optional[str] = None) -> Optional[str]:
    if not name:
        return None
    if load_path:
        return (
            f"tool_execute(name='{name}', arguments={{...}}, load_path='{load_path}', "
            "load_if_missing=True)"
        )
    return f"tool_execute(name='{name}', arguments={{...}})"


def _extract_input_schema_summary(tool_obj: Any) -> Dict[str, str]:
    tool_spec_attr = getattr(tool_obj, "TOOL_SPEC", None)
    if isinstance(tool_spec_attr, dict):
        schema = tool_spec_attr.get("inputSchema") or tool_spec_attr.get("input_schema")
    else:
        schema = None

    spec = getattr(tool_obj, "tool_spec", None)

    if schema is None and spec:
        if isinstance(spec, dict):
            schema = spec.get("inputSchema") or spec.get("input_schema")
        else:
            schema = getattr(spec, "input_schema", None) or getattr(spec, "inputSchema", None)

    if isinstance(schema, dict):
        if "json" in schema and isinstance(schema["json"], dict):
            schema = schema["json"]

        if isinstance(schema, dict):
            props = schema.get("properties") if isinstance(schema.get("properties"), dict) else None
            if props:
                summary: Dict[str, str] = {}
                for key, value in props.items():
                    if isinstance(value, dict):
                        type_value = value.get("type")
                        if not type_value and "anyOf" in value and isinstance(value["anyOf"], list):
                            type_value = " | ".join(
                                _safe_str(item.get("type", "any"))
                                for item in value["anyOf"]
                                if isinstance(item, dict)
                            )
                        summary[key] = _safe_str(type_value) or "any"
                    else:
                        summary[key] = "any"
                if summary:
                    return summary

    # Fallback to signature inspection
    try:
        signature = inspect.signature(tool_obj)
    except (TypeError, ValueError):
        try:
            signature = inspect.signature(tool_obj.__call__)
        except (TypeError, ValueError):
            return {}

    summary = {}
    for name, param in signature.parameters.items():
        if name in {"self", "tool", "tool_use", "toolUse", "toolUseId", "invocation_state"}:
            continue
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            summary[name] = "any"
        else:
            if isinstance(annotation, type):
                summary[name] = annotation.__name__
            else:
                summary[name] = _safe_str(annotation)

    return summary


def _extract_full_input_schema(tool_obj: Any) -> Optional[Dict[str, Any]]:
    tool_spec_attr = getattr(tool_obj, "TOOL_SPEC", None)
    if isinstance(tool_spec_attr, dict):
        schema = tool_spec_attr.get("inputSchema") or tool_spec_attr.get("input_schema")
    else:
        schema = None

    if schema is None:
        spec = getattr(tool_obj, "tool_spec", None)
        if spec is not None:
            if isinstance(spec, dict):
                schema = spec.get("inputSchema") or spec.get("input_schema")
            elif hasattr(spec, "model_dump"):
                spec_dict = spec.model_dump(mode="python")
                schema = spec_dict.get("inputSchema") or spec_dict.get("input_schema")
            else:
                schema = getattr(spec, "input_schema", None) or getattr(spec, "inputSchema", None)

    # Unwrap {"json": {actual_schema}} envelope from DecoratedFunctionTool
    if isinstance(schema, dict) and "json" in schema and isinstance(schema["json"], dict):
        schema = schema["json"]

    return schema


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|")


class ToolCatalogManager:
    """Manages a tool catalog stored on disk."""

    def __init__(self, catalog_path: Optional[Path] = None, write_markdown: Optional[bool] = None) -> None:
        self.catalog_path = Path(catalog_path) if catalog_path else _resolve_catalog_path()
        self.markdown_path = _resolve_markdown_path(self.catalog_path)
        if write_markdown is None:
            write_markdown = os.getenv(ENV_WRITE_MARKDOWN, "true").lower() in {"1", "true", "yes"}
        self.write_markdown = write_markdown
        # Cache infrastructure for build_catalog_overview()
        self._overview_cache: Optional[Dict[str, Any]] = None
        self._overview_cache_time: Optional[float] = None
        cache_ttl_env = os.getenv(ENV_OVERVIEW_CACHE_TTL)
        self._overview_cache_ttl = float(cache_ttl_env) if cache_ttl_env else DEFAULT_OVERVIEW_CACHE_TTL_SECONDS

    def _load_catalog(self) -> Dict[str, Any]:
        if self.catalog_path.exists():
            try:
                with self.catalog_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict) and "tools" in data:
                    return data
            except Exception as exc:
                logger.warning("Failed to read tool catalog %s: %s", self.catalog_path, exc)
        return {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "generated_at": _now_iso(),
            "tools": [],
        }

    def _write_catalog(self, data: Dict[str, Any]) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        data["schema_version"] = CATALOG_SCHEMA_VERSION
        data["generated_at"] = _now_iso()
        with self.catalog_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=True)

        if self.write_markdown:
            self._write_markdown(data)
        # Invalidate cache after any write
        self.invalidate_cache()

    def _write_markdown(self, data: Dict[str, Any]) -> None:
        tools = data.get("tools", [])
        lines = [
            "# Tool Catalog",
            "",
            f"Generated: {data.get('generated_at', '')}",
            "",
            "| Tool Name | Description | Input Parameters | Origin | Sandbox Status | Last Updated |",
            "| --- | --- | --- | --- | --- | --- |",
        ]

        for entry in tools:
            params = entry.get("input_summary") or entry.get("input_schema") or {}
            if isinstance(params, dict):
                params_text = ", ".join(
                    f"{_escape_md(str(key))}:{_escape_md(str(value))}" if value else _escape_md(str(key))
                    for key, value in params.items()
                )
            else:
                params_text = _escape_md(_safe_str(params))

            lines.append(
                "| {name} | {description} | {params} | {origin} | {sandbox} | {updated} |".format(
                    name=_escape_md(_safe_str(entry.get("name"))),
                    description=_escape_md(_safe_str(entry.get("description"))),
                    params=_escape_md(params_text or "-"),
                    origin=_escape_md(_safe_str(entry.get("origin"))),
                    sandbox=_escape_md(_safe_str(entry.get("sandbox_status"))),
                    updated=_escape_md(_safe_str(entry.get("last_updated"))),
                )
            )

        self.markdown_path.parent.mkdir(parents=True, exist_ok=True)
        with self.markdown_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))

    def _upsert_entry(self, entry: Dict[str, Any]) -> None:
        data = self._load_catalog()
        tools = data.get("tools", [])
        if not isinstance(tools, list):
            tools = []

        name = entry.get("name")
        if not name:
            return

        existing = next((item for item in tools if item.get("name") == name), None)
        if existing:
            existing.update(entry)
        else:
            tools.append(entry)

        data["tools"] = tools
        self._write_catalog(data)

    def _remove_entries(self, tool_names: Iterable[str]) -> None:
        data = self._load_catalog()
        tools = data.get("tools", [])
        if not isinstance(tools, list):
            tools = []

        names = {name for name in tool_names if name}
        if not names:
            return

        tools = [entry for entry in tools if entry.get("name") not in names]
        data["tools"] = tools
        self._write_catalog(data)

    def _build_entry(
        self,
        tool_obj: Any,
        origin: str,
        sandbox_status: Optional[str],
        category: Optional[str],
        load_pathway: Optional[str],
        execute_pathway: Optional[str],
        unload_pathway: Optional[str],
    ) -> Dict[str, Any]:
        name = _extract_tool_name(tool_obj)
        if not name:
            return {}
        full_schema = _extract_full_input_schema(tool_obj)
        summary = _extract_input_schema_summary(tool_obj)
        tool_path = _extract_tool_path(tool_obj)
        schema_hash = _compute_schema_hash(full_schema)
        return {
            "name": name,
            "description": _extract_tool_description(tool_obj),
            "input_schema": full_schema,
            "input_summary": summary,
            "input_schema_hash": schema_hash,
            "path": tool_path,
            "origin": origin,
            "category": category or origin,
            "sandbox_status": sandbox_status or _default_sandbox_status(),
            "last_updated": _now_iso(),
            "load_pathway": load_pathway,
            "execute_pathway": execute_pathway or _default_execute_pathway(name, tool_path),
            "unload_pathway": unload_pathway or (f"unload_tool(name='{name}')" if name else None),
        }

    def register_tool(
        self,
        tool_obj: Any,
        origin: str,
        sandbox_status: Optional[str] = None,
        category: Optional[str] = None,
        load_pathway: Optional[str] = None,
        execute_pathway: Optional[str] = None,
        unload_pathway: Optional[str] = None,
    ) -> None:
        entry = self._build_entry(
            tool_obj,
            origin=origin,
            sandbox_status=sandbox_status,
            category=category,
            load_pathway=load_pathway,
            execute_pathway=execute_pathway,
            unload_pathway=unload_pathway,
        )
        if entry:
            self._upsert_entry(entry)

    def register_tools(
        self,
        tools: Iterable[Any],
        origin: str,
        sandbox_status: Optional[str] = None,
        category: Optional[str] = None,
        load_pathway: Optional[str] = None,
        execute_pathway: Optional[str] = None,
        unload_pathway: Optional[str] = None,
    ) -> None:
        data = self._load_catalog()
        tools_list = data.get("tools", [])
        if not isinstance(tools_list, list):
            tools_list = []
        tool_index = {entry.get("name"): entry for entry in tools_list if isinstance(entry, dict)}

        for tool_obj in tools:
            try:
                entry = self._build_entry(
                    tool_obj,
                    origin=origin,
                    sandbox_status=sandbox_status,
                    category=category,
                    load_pathway=load_pathway,
                    execute_pathway=execute_pathway,
                    unload_pathway=unload_pathway,
                )
                if not entry:
                    continue
                tool_index[entry["name"]] = {**tool_index.get(entry["name"], {}), **entry}
            except Exception as exc:
                logger.debug("Failed to register tool in catalog: %s", exc)

        data["tools"] = list(tool_index.values())
        self._write_catalog(data)

    def register_entry(
        self,
        name: str,
        description: str,
        input_schema: Optional[Dict[str, Any]],
        origin: str,
        sandbox_status: Optional[str] = None,
        category: Optional[str] = None,
        path: Optional[str] = None,
        load_pathway: Optional[str] = None,
        execute_pathway: Optional[str] = None,
        unload_pathway: Optional[str] = None,
    ) -> None:
        entry = {
            "name": name,
            "description": description,
            "input_schema": input_schema or {},
            "path": path,
            "origin": origin,
            "category": category or origin,
            "sandbox_status": sandbox_status or _default_sandbox_status(),
            "last_updated": _now_iso(),
            "load_pathway": load_pathway,
            "execute_pathway": execute_pathway or _default_execute_pathway(name, path),
            "unload_pathway": unload_pathway,
        }
        self._upsert_entry(entry)

    def remove_tools(self, tool_names: Iterable[str]) -> None:
        try:
            self._remove_entries(tool_names)
        except Exception as exc:
            logger.debug("Failed to remove tools from catalog: %s", exc)

    def _resolve_manifest_path(self) -> Optional[Path]:
        env_path = os.getenv(ENV_DISCOVERY_MANIFEST)
        if env_path:
            candidate = Path(env_path).expanduser()
            if candidate.exists():
                return candidate
        sandbox_root = _resolve_sandbox_root()
        candidate = sandbox_root / "tool_manifests" / "tools_discovery_manifest.json"
        if candidate.exists():
            return candidate
        fallback = Path.cwd() / "tool_manifests" / "tools_discovery_manifest.json"
        if fallback.exists():
            return fallback
        return None

    def _load_discovery_manifest(self) -> Optional[Dict[str, Any]]:
        manifest_path = self._resolve_manifest_path()
        if not manifest_path:
            return None
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            logger.debug("Failed to load discovery manifest: %s", exc)
            return None

    def _category_order(self) -> List[str]:
        env = os.getenv(ENV_CATEGORY_ORDER)
        if env:
            return [item.strip() for item in env.split(",") if item.strip()]
        return DEFAULT_CATEGORY_ORDER

    def build_catalog_overview(self) -> Dict[str, Any]:
        # Check cache first
        now = time.time()
        if (
            self._overview_cache is not None
            and self._overview_cache_time is not None
            and (now - self._overview_cache_time) < self._overview_cache_ttl
        ):
            return self._overview_cache

        catalog = self._load_catalog()
        discovery = self._load_discovery_manifest()

        tools_by_name: Dict[str, Dict[str, Any]] = {}
        for entry in catalog.get("tools", []) if isinstance(catalog, dict) else []:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not name:
                continue
            entry.setdefault("execute_pathway", _default_execute_pathway(name))
            entry.setdefault("unload_pathway", f"unload_tool(name='{name}')")
            entry.setdefault("load_pathway", "already_loaded")
            tools_by_name[name] = entry

        if discovery and isinstance(discovery.get("loadable_tools"), list):
            for tool in discovery["loadable_tools"]:
                if not isinstance(tool, dict):
                    continue
                name = tool.get("name")
                if not name:
                    continue
                if name in tools_by_name:
                    continue
                tools_by_name[name] = {
                    "name": name,
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("input_schema"),
                    "path": tool.get("path"),
                    "origin": tool.get("category", "loadable"),
                    "category": tool.get("category", "loadable"),
                    "sandbox_status": _default_sandbox_status(),
                    "last_updated": tool.get("last_updated") or catalog.get("generated_at") or _now_iso(),
                    "load_pathway": tool.get("load_command"),
                    "execute_pathway": _default_execute_pathway(name, tool.get("path")),
                    "unload_pathway": tool.get("unload_command") or f"unload_tool(name='{name}')",
                    "status": "available",
                }

        categories: Dict[str, Dict[str, Any]] = {}

        def _add_entry_to_category(entry: Dict[str, Any]) -> None:
            category_id = entry.get("category") or entry.get("origin") or "other"
            category = categories.setdefault(
                category_id,
                {
                    "id": category_id,
                    "label": CATEGORY_LABELS.get(category_id, category_id.replace("_", " ").title()),
                    "tools": [],
                },
            )
            tool_info: Dict[str, Any] = {"name": entry.get("name")}
            if entry.get("description"):
                tool_info["description"] = entry["description"]
            if entry.get("input_summary"):
                tool_info["input_summary"] = entry["input_summary"]
            category["tools"].append(tool_info)

        for entry in tools_by_name.values():
            _add_entry_to_category(entry)

        if discovery and isinstance(discovery.get("mcp_servers"), list):
            category_id = "mcp_servers"
            category = categories.setdefault(
                category_id,
                {
                    "id": category_id,
                    "label": CATEGORY_LABELS.get(category_id, "MCP Servers"),
                    "tools": [],
                },
            )
            for server in discovery["mcp_servers"]:
                if not isinstance(server, dict):
                    continue
                server_id = server.get("id")
                if not server_id:
                    continue
                category["tools"].append(server_id)

        if discovery and isinstance(discovery.get("openapi_specs"), list):
            category_id = "openapi_specs"
            category = categories.setdefault(
                category_id,
                {
                    "id": category_id,
                    "label": CATEGORY_LABELS.get(category_id, "OpenAPI Specs"),
                    "tools": [],
                },
            )
            for spec in discovery["openapi_specs"]:
                if not isinstance(spec, dict):
                    continue
                name = spec.get("name")
                if not name:
                    continue
                category["tools"].append(name)

        ordered_categories: List[Dict[str, Any]] = []
        seen = set()
        for category_id in self._category_order():
            if category_id in categories:
                ordered_categories.append(categories[category_id])
                seen.add(category_id)
        for category_id, category in categories.items():
            if category_id not in seen:
                ordered_categories.append(category)

        for category in ordered_categories:
            tools = [name for name in category.get("tools", []) if name]
            category["tools"] = sorted(set(tools))
            category["count"] = len(category["tools"])

        result = {
            "generated_at": _now_iso(),
            "categories": ordered_categories,
        }
        # Update cache
        self._overview_cache = result
        self._overview_cache_time = time.time()
        return result

    def invalidate_cache(self) -> None:
        """Invalidate the overview cache. Called automatically after catalog writes."""
        self._overview_cache = None
        self._overview_cache_time = None

    def get_tool_details(self, tool_name: str) -> Optional[Dict[str, Any]]:
        if not tool_name:
            return None

        catalog = self._load_catalog()
        for entry in catalog.get("tools", []) if isinstance(catalog, dict) else []:
            if isinstance(entry, dict) and entry.get("name") == tool_name:
                entry.setdefault("execute_pathway", _default_execute_pathway(tool_name))
                entry.setdefault("unload_pathway", f"unload_tool(name='{tool_name}')")
                entry.setdefault("load_pathway", "already_loaded")
                entry.setdefault("path", None)
                entry.setdefault("input_schema", None)
                entry.setdefault("description", "")
                return entry

        discovery = self._load_discovery_manifest()
        if discovery:
            for tool in discovery.get("loadable_tools", []) if isinstance(discovery.get("loadable_tools"), list) else []:
                if not isinstance(tool, dict):
                    continue
                if tool.get("name") == tool_name:
                    return {
                        "name": tool_name,
                        "description": tool.get("description", ""),
                        "input_schema": tool.get("input_schema"),
                        "path": tool.get("path"),
                        "origin": tool.get("category", "loadable"),
                        "category": tool.get("category", "loadable"),
                        "sandbox_status": _default_sandbox_status(),
                        "last_updated": tool.get("last_updated") or _now_iso(),
                        "load_pathway": tool.get("load_command"),
                        "execute_pathway": _default_execute_pathway(tool_name, tool.get("path")),
                        "unload_pathway": tool.get("unload_command") or f"unload_tool(name='{tool_name}')",
                        "status": "available",
                    }
            for server in discovery.get("mcp_servers", []) if isinstance(discovery.get("mcp_servers"), list) else []:
                if not isinstance(server, dict):
                    continue
                if server.get("id") == tool_name:
                    return {
                        "name": tool_name,
                        "description": server.get("description", ""),
                        "input_schema": None,
                        "path": server.get("path"),
                        "origin": "mcp_server",
                        "category": "mcp_servers",
                        "sandbox_status": _default_sandbox_status(),
                        "last_updated": _now_iso(),
                        "load_pathway": server.get("connect_command") or server.get("connection_command"),
                        "execute_pathway": None,
                        "unload_pathway": f"mcp_client(action='disconnect', connection_id='{tool_name}')",
                        "status": "available",
                    }
            for spec in discovery.get("openapi_specs", []) if isinstance(discovery.get("openapi_specs"), list) else []:
                if not isinstance(spec, dict):
                    continue
                if spec.get("name") == tool_name:
                    return {
                        "name": tool_name,
                        "description": spec.get("file", ""),
                        "input_schema": None,
                        "path": spec.get("path"),
                        "origin": "openapi_spec",
                        "category": "openapi_specs",
                        "sandbox_status": _default_sandbox_status(),
                        "last_updated": _now_iso(),
                        "load_pathway": spec.get("mcp_command"),
                        "execute_pathway": None,
                        "unload_pathway": None,
                        "status": "available",
                    }

        return None


_catalog_manager: Optional[ToolCatalogManager] = None


def get_tool_catalog_manager() -> ToolCatalogManager:
    global _catalog_manager
    if _catalog_manager is None:
        _catalog_manager = ToolCatalogManager()
    return _catalog_manager
