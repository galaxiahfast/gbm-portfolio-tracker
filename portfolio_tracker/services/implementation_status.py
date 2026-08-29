"""Inventario auditable del avance y complejidad del proyecto local."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


LAST_VERIFIED_TESTS = 77
LAST_VERIFIED_DATE = "2026-08-29"


@dataclass(frozen=True, slots=True)
class ImplementationMilestone:
    name: str
    detail: str
    status: str
    weight: float


@dataclass(frozen=True, slots=True)
class ImplementationSnapshot:
    completion_percent: int
    active_modules: int
    python_lines: int
    discovered_tests: int
    verified_tests: int
    verified_date: str
    milestones: tuple[ImplementationMilestone, ...]


MILESTONES = (
    ImplementationMilestone("Persistencia SQLite y migraciones", "Inicialización incremental sin sobrescribir datos privados.", "complete", 1.0),
    ImplementationMilestone("Contabilidad, efectivo y auditoría", "Libro USD/MXN, posiciones, integridad y SHA-256.", "complete", 1.0),
    ImplementationMilestone("OCR con confirmación humana", "Extracción editable y validación matemática antes de guardar.", "complete", 1.0),
    ImplementationMilestone("Motor multi-temporal Fase 4", "Señales 5m, contexto diario/semanal y veto de riesgo.", "complete", 1.5),
    ImplementationMilestone("Proyección bootstrap de 15 sesiones", "Volatilidad histórica, ATR y reacción a estructura real.", "complete", 1.0),
    ImplementationMilestone("PDF técnico con todos los paneles", "Intradiario, contexto y estructura mensual sin omisiones.", "complete", 1.5),
    ImplementationMilestone("Cuatro reportes independientes", "Vista ejecutiva, técnica, reporte combinado y maestro con calibración.", "complete", 0.5),
    ImplementationMilestone("Detector de patrones chartistas y ondas", "Pivotes Zig-Zag, dobles/triples extremos, rupturas y estructuras impulso/ABC con veto objetivo.", "complete", 1.0),
    ImplementationMilestone("Análisis fundamental y noticias", "Estados financieros, eventos, flujo informativo versionado, ponderación direccional y veto SHA-256.", "complete", 1.0),
    ImplementationMilestone("Calibración estadística y backtesting", "Búsqueda anidada de parámetros, OOS intacto, ATR, costes, error Brier, veto de capital y huella auditable.", "complete", 1.5),
    ImplementationMilestone("Realimentación progresiva intradía", "Observaciones únicas cada cinco minutos, resolución a una sesión, umbral adaptativo y SHA-256.", "complete", 1.0),
)


def _count_test_functions(test_files: list[Path]) -> int:
    count = 0
    for path in test_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        count += sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            for node in ast.walk(tree)
        )
    return count


def inspect_implementation_status(project_root: Path) -> ImplementationSnapshot:
    """Escanea código y pruebas sin ejecutar procesos ni acceder a la red."""

    package_files = sorted(
        path
        for path in (project_root / "portfolio_tracker").rglob("*.py")
        if "__pycache__" not in path.parts
    )
    app_file = project_root / "app.py"
    code_files = package_files + ([app_file] if app_file.exists() else [])
    python_lines = 0
    for path in code_files:
        try:
            python_lines += sum(
                bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
            )
        except (OSError, UnicodeDecodeError):
            continue

    test_files = sorted((project_root / "tests").glob("test_*.py"))
    discovered_tests = _count_test_functions(test_files)
    total_weight = sum(item.weight for item in MILESTONES)
    completed_weight = sum(item.weight for item in MILESTONES if item.status == "complete")
    completion_percent = round(completed_weight / total_weight * 100) if total_weight else 0
    return ImplementationSnapshot(
        completion_percent=completion_percent,
        active_modules=len(package_files),
        python_lines=python_lines,
        discovered_tests=discovered_tests,
        verified_tests=min(LAST_VERIFIED_TESTS, discovered_tests),
        verified_date=LAST_VERIFIED_DATE,
        milestones=MILESTONES,
    )
