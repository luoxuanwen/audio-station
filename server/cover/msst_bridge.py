from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


class MSSTBridgeError(RuntimeError):
    """Raised when the bundled MSST installation cannot be used."""


_MODULE_LOCK = threading.Lock()
_MODULE_CACHE: dict[Path, ModuleType] = {}


def _resolve_root(base_dir: Path, configured_root: Any) -> Path:
    raw = str(configured_root or ".").strip()
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = base_dir / root
    return root.resolve()


def _load_module(root: Path) -> ModuleType:
    module_path = root / "msst" / "msst_separate.py"
    if not module_path.is_file():
        raise MSSTBridgeError(f"找不到 MSST 分离模块: {module_path}")

    with _MODULE_LOCK:
        cached = _MODULE_CACHE.get(module_path)
        if cached is not None:
            return cached
        module_name = f"svcvc_msst_{hashlib.sha256(str(module_path).encode('utf-8')).hexdigest()[:12]}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise MSSTBridgeError(f"无法载入 MSST 分离模块: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE_CACHE[module_path] = module
        return module


def describe_msst(base_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the selected model and return a cheap cache fingerprint."""
    root = _resolve_root(base_dir, config.get("root"))
    module = _load_module(root)
    model_name = str(config.get("model") or getattr(module, "DEFAULT_MODEL_ID", "")).strip()
    try:
        model_id, model_path_raw, config_path_raw = module.resolve_msst_model(model_name)
    except Exception as exc:
        raise MSSTBridgeError(f"MSST 模型解析失败: {exc}") from exc

    module_path = root / "msst" / "msst_separate.py"
    runtime_path = root / "runtime-rocm" / "Scripts" / "python.exe"
    model_path = Path(model_path_raw).resolve()
    model_config_path = Path(config_path_raw).resolve()
    required = {
        "分离模块": module_path,
        "原生 ROCm Python": runtime_path,
        "模型权重": model_path,
        "模型配置": model_config_path,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise MSSTBridgeError("MSST 依赖不完整：" + "；".join(missing))

    manifest: list[tuple[str, int, int]] = []
    for label, path in required.items():
        stat = path.stat()
        manifest.append((label, stat.st_size, stat.st_mtime_ns))
    settings = {
        "batch_size": max(1, int(config.get("batch_size", 1))),
        "num_overlap": max(1, int(config.get("num_overlap", 4))),
        "normalize": bool(config.get("normalize", False)),
        "use_tta": bool(config.get("use_tta", False)),
    }
    serialized = json.dumps(
        {"manifest": manifest, "model_id": model_id, "settings": settings},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "root": str(root),
        "model_id": model_id,
        "model_path": str(model_path),
        "config_path": str(model_config_path),
        "runtime_path": str(runtime_path),
        "settings": settings,
        "digest": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def list_msst_models(base_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the installed model allow-list with the persisted selection marked."""
    root = _resolve_root(base_dir, config.get("root"))
    module = _load_module(root)
    getter = getattr(module, "get_msst_models", None)
    if not callable(getter):
        raise MSSTBridgeError("MSST 分离模块未提供模型列表接口")

    current = str(config.get("model") or getattr(module, "DEFAULT_MODEL_ID", "")).strip()
    models: list[dict[str, Any]] = []
    try:
        raw_models = getter()
    except Exception as exc:
        raise MSSTBridgeError(f"读取 MSST 模型列表失败: {exc}") from exc
    for item in raw_models or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        model_id = str(item["id"]).strip()
        if not model_id:
            continue
        try:
            resolved_id, model_path, model_config_path = module.resolve_msst_model(model_id)
        except Exception:
            continue
        models.append(
            {
                "id": str(resolved_id),
                "name": str(item.get("name") or resolved_id),
                "current": str(resolved_id) == current,
                "model_path": str(Path(model_path).resolve()),
                "config_path": str(Path(model_config_path).resolve()),
            }
        )
    if not models:
        raise MSSTBridgeError("没有发现权重和配置均完整的 MSST 模型")
    if not any(item["current"] for item in models):
        models[0]["current"] = True
    return models


def separate_target(
    input_audio: Path,
    output_dir: Path,
    base_dir: Path,
    config: dict[str, Any],
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Run bundled MSST in its native ROCm subprocess and return vocal/other stems."""
    info = describe_msst(base_dir, config)
    module = _load_module(Path(info["root"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        vocal_raw, instrumental_raw = module.separate_vocal(
            str(input_audio.resolve()),
            output_dir=str(output_dir.resolve()),
            output_format="wav",
            mode="subprocess",
            inference_params=dict(info["settings"]),
            model_name=info["model_id"],
            progress_callback=progress_callback,
        )
    except Exception as exc:
        raise MSSTBridgeError(f"MSST 人声分离失败: {exc}") from exc

    vocal_path = Path(vocal_raw).resolve() if vocal_raw else None
    instrumental_path = Path(instrumental_raw).resolve() if instrumental_raw else None
    if vocal_path is None or not vocal_path.is_file() or vocal_path.stat().st_size <= 0:
        raise MSSTBridgeError("MSST 未生成有效的人声文件")
    if instrumental_path is None or not instrumental_path.is_file() or instrumental_path.stat().st_size <= 0:
        raise MSSTBridgeError("MSST 未生成有效的伴奏文件")
    return vocal_path, instrumental_path, info
