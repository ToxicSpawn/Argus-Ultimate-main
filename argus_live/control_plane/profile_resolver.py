from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class YamlRootTypeError(ValueError):
    pass


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: UniqueKeyLoader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key detected: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=UniqueKeyLoader) or {}
    if not isinstance(data, dict):
        raise YamlRootTypeError(f"YAML root must be a mapping: {p}")
    return data


def load_profile_bundle(
    *,
    constitution_path: str,
    runtime_path: str,
    exchange_path: str,
    strategy_path: str,
) -> dict[str, dict[str, Any]]:
    return {
        "constitution": _load_yaml(constitution_path),
        "runtime": _load_yaml(runtime_path),
        "exchange": _load_yaml(exchange_path),
        "strategy": _load_yaml(strategy_path),
    }
