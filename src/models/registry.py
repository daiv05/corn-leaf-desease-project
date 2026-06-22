from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch.nn as nn


@dataclass
class ModelEntry:
    name: str
    factory: Callable[..., nn.Module]
    default_kwargs: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, ModelEntry] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., nn.Module],
        default_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._entries[name] = ModelEntry(
            name=name,
            factory=factory,
            default_kwargs=default_kwargs or {},
        )

    def build(self, name: str, num_classes: int, **override_kwargs: Any) -> nn.Module:
        if name not in self._entries:
            available = ", ".join(sorted(self._entries))
            raise KeyError(f"Modelo '{name}' no registrado. Disponibles: {available}")
        entry = self._entries[name]
        kwargs = {**entry.default_kwargs, **override_kwargs, "num_classes": num_classes}
        return entry.factory(**kwargs)

    def list_names(self) -> list[str]:
        return sorted(self._entries)

    def get_entry(self, name: str) -> ModelEntry:
        if name not in self._entries:
            available = ", ".join(sorted(self._entries))
            raise KeyError(f"Modelo '{name}' no registrado. Disponibles: {available}")
        return self._entries[name]

    def __contains__(self, name: str) -> bool:
        return name in self._entries


MODEL_REGISTRY = ModelRegistry()
