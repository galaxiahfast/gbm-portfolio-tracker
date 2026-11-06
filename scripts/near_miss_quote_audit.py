"""Closed-data, one-failed-gate counterfactual. No production imports/writes.

Quotes are observed SIP data. Marketable limits are POSSIBLE executions, not
broker fills. No assumed spread, commission or slippage enters this analysis.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_trigger_replay import OUTPUT as AUDIT, digest_file
from scripts.download_massive_history import read_signed, write_signed, sha
from scripts.seal_trigger_experiment_holdout import OUTPUT as EXPERIMENT
from scripts.run_trigger_experiment import unclock, validate_environment
from scripts.trigger_experiment_rules import nonoverlapping_events
from portfolio_tracker.analytics.closed_bars import _calendar

OUTPUT = ROOT / "output/historical_replay/near_miss_quotes_v1"


def exactly_one_failure(gate):
    """Literal AND operands, not descriptive inside-zone/bullish diagnostics."""
    if gate.get("eligible") or gate.get("side") not in ("LONG", "SHORT"):
        return False
    flags = {**gate.get("non_temporal", {}), **gate.get("micro", {})}
    failed = [k for k, v in flags.items() if not v]
    return len(failed) == 1 and failed == gate.get("failed_conditions")


def prepare():
    destination = OUTPUT / "method.json"
    if destination.exists():
        method = read_signed(destination)
        verify(method)
        return method
    original = read_signed(EXPERIMENT / "preregistration.json")
    validate_environment(original)
    source_files, events = {}, []
    for stage in ("development", "holdout"):
        completed = read_signed(EXPERIMENT / f"{stage}_complete.json")
        if completed["errors"]:
            raise ValueError("Source experiment not closed.")
        source_files[str((EXPERIMENT / f"{stage}_complete.json").relative_to(ROOT))] = digest_file(EXPERIMENT / f"{stage}_complete.json")
        for symbol, obj in completed["symbols"].items():
            origins = {}
            if stage == "development":
                for f in read_signed(AUDIT / f"{symbol}_completion.json")["records"]:
                    r = read_signed(AUDIT / f["file"])
                    origins[r["cut_at"]] = (r, f)
            for f in obj["checkpoint_files"]:
                file = EXPERIMENT / f["path"]
                if digest_file(file) != f["sha256"]:
                    raise ValueError("Altered source checkpoint.")
                source_files[str(file.relative_to(ROOT))] = f["sha256"]
                for cut in read_signed(file)["cuts"]:
                    gate = cut["variants"]["CURRENT"]
                    if not exactly_one_failure(gate):
                        continue
                    if stage == "development":
                        origin, ref = origins[cut["cut_at"]]
                        origin_file = AUDIT / ref["file"]
                        if digest_file(origin_file) != ref["file_sha256"]:
                            raise ValueError("Altered original cut.")
                        source_files[str(origin_file.relative_to(ROOT))] = ref["file_sha256"]
                    else:
                        # Read the already opened/closed snapshot only. Never
                        # decrypt/reopen or call the holdout evaluation again.
                        origin = cut["native_holdout_snapshot"]
                    if sha(origin["feature_snapshot"]) != cut["original_features_sha256"]:
                        raise ValueError("Features and contract have different provenance.")
                    contracts = [unclock(h["operational_contract"]) for h in origin["horizons"]
                                 if h["horizon_minutes"] in (60, 1440)]
                    if len(contracts) != 2 or any(c["side"] != gate["side"] for c in contracts):
                        raise ValueError("Near miss must retain both original plans.")
                    event = {"stage": stage, "symbol": symbol, "cut_at": cut["cut_at"],
                             "side": gate["side"], "missing_condition": gate["failed_conditions"][0],
                             "exposure": gate["exposure"], "gate_evidence": gate,
                             "feature_snapshot_sha256": cut["original_features_sha256"],
                             "contracts": contracts,
                             "horizons": [{"contract": c} for c in contracts]}
                    event["event_id"] = sha({k: event[k] for k in ("stage", "symbol", "cut_at", "side", "missing_condition", "feature_snapshot_sha256")})
                    events.append(event)
    independent = set()
    for stage in ("development", "holdout"):
        for symbol in original["symbols"]:
            rows = [e for e in events if e["stage"] == stage and e["symbol"] == symbol]
            independent.update(e["event_id"] for e in nonoverlapping_events(rows))
    for event in events:
        event["nonoverlapping"] = event["event_id"] in independent
        cut = pd.Timestamp(event["cut_at"])
        end = max(pd.Timestamp(c["timeout_at"]) for c in event["contracts"])
        schedule = _calendar(cut.year-1, end.year+1).schedule.loc[
            str(cut.tz_convert("America/New_York").date()):str(end.tz_convert("America/New_York").date())]
        event["quote_windows"] = [{"start": max(cut-pd.Timedelta(seconds=10), r.open).isoformat(),
                                   "end": min(end, r.close).isoformat()}
                                  for r in schedule.itertuples() if max(cut, r.open) < min(end, r.close)]
    events.sort(key=lambda e: (e["stage"], e["symbol"], e["cut_at"]))
    method = {
        "version": "ONE_LITERAL_FAILED_GATE_OBSERVED_SIP_V1", "registered_at": datetime.now(timezone.utc).isoformat(),
        "scope": "POST_HOC_DIAGNOSIS_OF_ALREADY_CONSUMED_BLOCKS_NOT_NEW_HOLDOUT_OR_APPROVAL",
        "original_protocol_sha256": sha(original), "original_ledger_sha256": original["ledger_sha256"],
        "source_files_sha256": source_files,
        "implementation_sha256": {n: digest_file(ROOT/n) for n in ("scripts/near_miss_quote_audit.py", "scripts/collect_near_miss_quotes.py", "scripts/report_near_miss_quotes.py", "tests/test_near_miss_quotes.py")},
        "selection": "CURRENT in both CLOSED blocks, exactly one false literal AND operand. Duplicate veto/rejection operands remain distinct. Price-inside-zone is descriptive, not an operand. No selection by outcome.",
        "quotes": "GET historical Alpaca SIP, raw bid/ask, USD; every paginated quote in regular XNYS sessions from cut-10s through longest signed timeout. No snapshots from other dates as spread proxies.",
        "quote_quality": "Only condition R (regular), finite positive uncrossed bid<=ask, both sizes positive. Ordering checked. No forward/backward interpolation, no price-outlier capping. Condition exclusions and missing quotes reported.",
        "counterfactual": "One share, ignore ONLY missing trigger operand for research. Same frozen side/limit/TP/SL, next-full-5m evaluation start, session entry deadline and original maturity. LONG buy at first marketable ask<=limit, liquidate at observed bid; SHORT sell at bid>=limit, cover at observed ask. Cancel if SL/TP was passed before entry. No broker fill, queue, liquidity capacity, borrow or latency claim.",
        "timeouts": "Last valid quote <= exact timeout and age<=10s, else N/D. No after-close substitute.",
        "uncertainty": "No quotes since session start for >60s before entry/exit invalidates first-passage inference. Regular sessions only. Exact-timestamp event ambiguity reported when conflicting states cannot be ordered.",
        "excursions": "MFE/MAE use tradable liquidation bid LONG / ask SHORT minus actual compatible entry quote. Both until first exit and entire signed horizon are reported separately. Full-horizon excursion is ex-post, NOT achieved trade PNL.",
        "costs": "Spread observed through entry ask/exit bid (reverse SHORT); gross quote PNL only. Commission, slippage, taxes, borrow costs and net PNL N/D; NO assumed spread/slippage subtracted.",
        "classification": "Positive gross quote PNL: blocked hypothetical gross gain; negative: avoided hypothetical gross loss. Not actual gains/losses, not accuracy, not causal proof filter quality.",
        "events": events, "no_production_or_accounting_changes": True,
    }
    write_signed(destination, method)
    print("Population fixed:", len(events), dict(Counter(e["side"] for e in events)), flush=True)
    return method


def verify(method):
    validate_environment(read_signed(EXPERIMENT / "preregistration.json"))
    for name, expected in {**method["source_files_sha256"], **method["implementation_sha256"]}.items():
        if digest_file(ROOT/name) != expected:
            raise ValueError(f"Diagnostic source changed: {name}")


def write_compressed(path, payload):
    if path.exists():
        if read_compressed(path) != payload:
            raise ValueError("Immutable quote page already exists.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps({"payload": payload, "sha256": sha(payload)}, separators=(",", ":"), allow_nan=False).encode()
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as z:
            z.write(encoded)
    tmp.replace(path)


def read_compressed(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        envelope = json.load(f)
    if sha(envelope["payload"]) != envelope["sha256"]:
        raise ValueError("Altered quote page.")
    return envelope["payload"]


def quote_valid(q):
    try:
        bid, ask, bs, az = map(float, (q["bp"], q["ap"], q["bs"], q["as"]))
        return (all(math.isfinite(v) for v in (bid, ask, bs, az)) and
                0 < bid <= ask and bs > 0 and az > 0 and q.get("c") == ["R"])
    except (KeyError, TypeError, ValueError):
        return False


class QuotePath:
    """Incremental quote counterfactual; never changes original eligibility."""
    def __init__(self, contract):
        self.c = contract
        self.start = pd.Timestamp(contract["evaluation_starts_at"])
        self.end = pd.Timestamp(contract["timeout_at"])
        self.deadline = min(self.end, pd.Timestamp(contract["execution_terms"]["entry_deadline_at"]))
        self.last_at = None
        self.last_q = None
        self.session = None
        self.coverage_gap = False
        self.state = "PENDING_ENTRY"
        self.fill = None
        self.exit = None
        self.reference = None
        self.mfe = self.mae = self.trade_mfe = self.trade_mae = None
        self.n = self.invalid = 0
        self.conditions = Counter()
        self.duplicate_decision_timestamp = False
        self.last_raw_time = None

    def feed(self, q):
        t = pd.Timestamp(q["t"])
        if self.last_raw_time is not None and t < self.last_raw_time:
            raise ValueError("SIP quotes out of order.")
        self.last_raw_time = t
        self.conditions.update([','.join(q.get('c', [])) or 'MISSING'])
        if t < pd.Timestamp(self.c["observed_at"]) - pd.Timedelta(seconds=10) or t > self.end:
            return
        if not quote_valid(q):
            self.invalid += 1
            return
        if t <= pd.Timestamp(self.c["observed_at"]):
            self.reference = {"quote": q, "age_seconds": (pd.Timestamp(self.c["observed_at"])-t).total_seconds()}
        if t < self.start:
            return
        day = t.tz_convert("America/New_York").date()
        if day != self.session:
            schedule = _calendar(t.year-1, t.year+1).schedule.loc[str(day)]
            expected = max(self.start, schedule.open)
            if t - expected > pd.Timedelta(seconds=60):
                self.coverage_gap = True
        elif self.last_at is not None and t-self.last_at > pd.Timedelta(seconds=60):
            self.coverage_gap = True
        same_time = self.last_at == t
        self.session, self.last_at, self.last_q = day, t, q
        self.n += 1
        side = self.c["side"]
        liquidation = float(q["bp"] if side == "LONG" else q["ap"])
        entry_quote = float(q["ap"] if side == "LONG" else q["bp"])
        stop, target, limit = (float(self.c[k]) for k in ("stop_loss", "take_profit", "entry_price"))
        stop_hit = liquidation <= stop if side == "LONG" else liquidation >= stop
        tp_hit = liquidation >= target if side == "LONG" else liquidation <= target
        if self.state == "PENDING_ENTRY":
            if t >= self.deadline:
                self.state = "NO_ENTRY_DEADLINE"
            elif stop_hit or tp_hit:
                self.state = "NO_ENTRY_STOP_PASSED" if stop_hit else "NO_ENTRY_TARGET_PASSED"
            elif (entry_quote <= limit if side == "LONG" else entry_quote >= limit):
                self.fill = {"at": t.isoformat(), "price": entry_quote, "quote": q,
                             "spread_observed": float(q["ap"])-float(q["bp"])}
                self.state = "FILLED_POSSIBLE"
                if same_time:
                    self.duplicate_decision_timestamp = True
        if self.fill:
            mark = liquidation-self.fill["price"] if side == "LONG" else self.fill["price"]-liquidation
            self.mfe = mark if self.mfe is None else max(self.mfe, mark)
            self.mae = mark if self.mae is None else min(self.mae, mark)
            if self.exit is None:
                self.trade_mfe = mark if self.trade_mfe is None else max(self.trade_mfe, mark)
                self.trade_mae = mark if self.trade_mae is None else min(self.trade_mae, mark)
                if stop_hit or tp_hit:
                    self.exit = {"at": t.isoformat(), "price": liquidation, "quote": q,
                                 "outcome": "SL_FIRST" if stop_hit else "TP_FIRST",
                                 "gross_pnl_per_share": mark,
                                 "spread_observed": float(q["ap"])-float(q["bp"])}
                    self.state = "RESOLVED_QUOTE_COUNTERFACTUAL"
                    if same_time:
                        self.duplicate_decision_timestamp = True

    def finish(self):
        if self.state == "PENDING_ENTRY":
            self.state = "NO_ENTRY_DEADLINE"
        if self.fill and self.exit is None:
            if self.last_at is not None and 0 <= (self.end-self.last_at).total_seconds() <= 10:
                q = self.last_q
                price = float(q["bp"] if self.c["side"] == "LONG" else q["ap"])
                pnl = price-self.fill["price"] if self.c["side"] == "LONG" else self.fill["price"]-price
                self.exit = {"at": self.end.isoformat(), "observed_quote_at": self.last_at.isoformat(),
                             "quote": q, "price": price, "outcome": "TIMEOUT", "gross_pnl_per_share": pnl,
                             "spread_observed": float(q["ap"])-float(q["bp"])}
                self.state = "RESOLVED_QUOTE_COUNTERFACTUAL"
            else:
                self.state = "N_D_TIMEOUT_QUOTE"
        quality = []
        if self.coverage_gap:
            quality.append("QUOTE_GAP_OVER_60_SECONDS")
        if self.duplicate_decision_timestamp:
            quality.append("DECISION_AT_TIED_TIMESTAMP_ORDER_UNKNOWN")
        if not self.reference:
            quality.append("N_D_QUOTE_AT_CUT_WITHIN_10_SECONDS")
        elif abs((float(self.reference["quote"]["bp"])+float(self.reference["quote"]["ap"]))/2 / float(self.c["entry_price"])-1) > .05:
            quality.append("PRICE_BASIS_DISCREPANCY_OVER_5_PERCENT_AT_CUT")
        if not self.n:
            quality.append("NO_VALID_REGULAR_QUOTES")
        status = "N_D_QUOTES_INCOMPLETE" if quality else self.state
        pnl = self.exit["gross_pnl_per_share"] if self.exit else None
        interpretation = ("N_D" if quality or pnl is None else
                          "HYPOTHETICAL_GROSS_GAIN_BLOCKED" if pnl > 0 else
                          "HYPOTHETICAL_GROSS_LOSS_AVOIDED" if pnl < 0 else "GROSS_NEUTRAL")
        return {"horizon_minutes": self.c["model"]["horizon_minutes"], "status": status,
                "raw_counterfactual_status": self.state, "quality_flags": quality,
                "valid_quotes": self.n, "excluded_quotes": self.invalid, "conditions": dict(self.conditions),
                "cut_reference_quote": self.reference, "possible_fill": self.fill, "exit": self.exit,
                "gross_quote_pnl_per_share": pnl if not quality else None,
                "net_pnl": None, "broker_fill": False, "classification": interpretation,
                "mfe_until_exit_per_share": max(0., self.trade_mfe) if self.trade_mfe is not None else None,
                "mae_until_exit_per_share": min(0., self.trade_mae) if self.trade_mae is not None else None,
                "mfe_full_horizon_per_share": max(0., self.mfe) if self.mfe is not None else None,
                "mae_full_horizon_per_share": min(0., self.mae) if self.mae is not None else None,
                "last_valid_quote_at": self.last_at.isoformat() if self.last_at is not None else None}


if __name__ == "__main__":
    prepare()
