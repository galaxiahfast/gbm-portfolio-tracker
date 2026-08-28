"""Interfaces futuras: separan señales analiticas de decisiones de inversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class AnalysisContext:
    symbol: str
    as_of: datetime
    prices: Sequence[Decimal]
    volumes: Sequence[Decimal]
    fundamentals: Mapping[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    module: str
    score: Decimal
    observations: tuple[str, ...]
    risk_flags: tuple[str, ...] = ()


class AnalyticsModule(ABC):
    """Contrato para valoracion, indicadores, riesgo u otra metodologia."""

    @abstractmethod
    def evaluate(self, context: AnalysisContext) -> AnalysisResult:
        raise NotImplementedError


class PredictionModel(Protocol):
    """Contrato ML sin prometer certeza ni acoplarse a una libreria."""

    name: str
    version: str

    def predict_probability_up(
        self, context: AnalysisContext, horizon_days: int
    ) -> Decimal:
        """Devuelve una probabilidad calibrada entre 0 y 1."""
        ...

