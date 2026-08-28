from portfolio_tracker.config import PROJECT_ROOT
from portfolio_tracker.services.implementation_status import inspect_implementation_status


def test_implementation_snapshot_uses_real_project_metrics() -> None:
    snapshot = inspect_implementation_status(PROJECT_ROOT)

    assert 0 < snapshot.completion_percent < 100
    assert snapshot.active_modules >= 20
    assert snapshot.python_lines > 1_000
    assert snapshot.discovered_tests >= 46
    assert snapshot.verified_tests <= snapshot.discovered_tests
    assert any(item.status == "complete" for item in snapshot.milestones)
    assert any(item.status == "planned" for item in snapshot.milestones)
    chart_patterns = next(
        item
        for item in snapshot.milestones
        if item.name == "Detector de patrones chartistas y ondas"
    )
    assert chart_patterns.status == "complete"
