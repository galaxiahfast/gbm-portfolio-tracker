"""Regression: a persisted exit alert outranks a recovered candle close."""
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from portfolio_tracker.models import TradeDraft, TradeSide
from portfolio_tracker.services.decision_engine import generate_decision
from portfolio_tracker.services.operational_state import synchronize_position
from tests.test_decision_engine import repository
from tests.test_pdf_report import _analysis


def test_intrabar_stop_exit_pending_never_becomes_maintain(tmp_path):
    repo = repository(tmp_path)
    base = _analysis()
    repo.add_trade(TradeDraft(
        symbol="SMCI", side=TradeSide.BUY, quantity=Decimal("5"),
        price_usd=Decimal("40"), commission_usd=Decimal("0"),
        executed_at=base.as_of,
    ))
    adopted = synchronize_position(repo.database, base)
    stop = float(adopted.execution_levels.stop_loss)
    target = float(adopted.execution_levels.take_profit_1)
    recovered_close = (stop + target) / 2
    newer = base.intraday_indicators.copy()
    stamp = base.as_of + timedelta(minutes=5)
    newer.loc[stamp] = newer.iloc[-1]
    newer.loc[stamp, "Low"] = stop - 0.10
    newer.loc[stamp, "Close"] = recovered_close
    alert = synchronize_position(repo.database, replace(
        base, as_of=stamp, intraday_indicators=newer, last_price=recovered_close,
    ))
    assert alert.position_state == "EXIT_PENDING"
    assert stop < alert.last_price < target

    # The passed analysis may be stale: only the SHA-verified persisted state
    # may decide whether a previously touched stop still requires attention.
    stale_view = replace(alert, position_state="LONG_ACTIVE", position_management="")
    decision = generate_decision("SMCI", analysis=stale_view, repository=repo)
    assert decision.action == "CONFIRMAR_SALIDA"
    assert decision.position_size == 0
    assert decision.current_shares == 5
    assert not decision.trigger_met and decision.risk_veto
    assert any(check.key == "persistent_exit" and not check.passed for check in decision.activation_checks)
    assert "Stop alcanzado" in decision.explanation
    assert len(repo.list_trades()) == 1  # No fictitious sale or fill.

    # A recorded sale needs reconciliation, but the stale alert must not
    # authorize a new purchase before the persistent state is cleared.
    repo.add_trade(TradeDraft(
        symbol="SMCI", side=TradeSide.SELL, quantity=Decimal("5"),
        price_usd=Decimal("40"), commission_usd=Decimal("0"),
        executed_at=stamp + timedelta(minutes=1),
    ))
    unresolved = generate_decision("SMCI", analysis=stale_view, repository=repo)
    assert unresolved.action == "ESPERAR"
    assert unresolved.position_size == 0


def test_decision_fails_closed_if_persisted_state_hash_is_invalid(tmp_path):
    repo = repository(tmp_path)
    analysis = _analysis()
    synchronize_position(repo.database, analysis)
    with repo.database.transaction() as connection:
        connection.execute("UPDATE operational_events SET payload_json='{}'")
    with pytest.raises(ValueError, match="SHA-256"):
        generate_decision("SMCI", analysis=analysis, repository=repo)
