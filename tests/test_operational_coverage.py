from portfolio_tracker.services.operational_coverage import summarize_operational_events


def test_five_horizon_tp_first_rows_are_one_successful_event():
    target = {"entry_at": "2026-09-03T15:00:00+00:00", "side": "LONG",
              "entry_price": 100.0, "stop_loss": 95.0, "take_profit": 105.0}
    predictions = [
        {"operational_target": target,
         "operational_result": {"resolution_status": "RESOLVED", "outcome": "TP_FIRST",
                                "exit_at": "2026-09-03T15:10:00+00:00"}}
        for _ in range(5)
    ]
    predictions.append({"operational_target": target,
                        "operational_result": {"resolution_status": "PENDING", "outcome": None}})
    summary = summarize_operational_events([{"symbol": "SMCI", "predictions": predictions}])
    assert summary["horizon_resolutions"] == 5
    assert summary["independent_events"] == 1
    assert summary["tp_first_events"] == 1
    assert summary["resolved_events"] == 1
    assert summary["tp_first_rate"] == 1.0
