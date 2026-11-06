"""Acquire/audit an unused historical block and seal it, NEVER evaluate labels."""
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_trigger_replay import OUTPUT as AUDIT, INPUT, TIMES, digest_file
from scripts.download_massive_history import read_signed, write_signed, sha
from scripts.trigger_experiment_rules import rules_specification
from portfolio_tracker.analytics.closed_bars import _calendar

OUTPUT = ROOT / "output/historical_replay/trigger_experiment_v2"
AAD = b"TRIGGER_EXPERIMENT_HOLDOUT_V1_NOT_FOR_PRODUCTION"


def protect_key(value, decrypt=False):
    """Windows current-user DPAPI; no plaintext key is persisted."""
    class Blob(ctypes.Structure):
        _fields_ = [("cbData", ctypes.c_ulong), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]
    buffer = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
    source, destination = Blob(len(value), buffer), Blob()
    function = getattr(ctypes.windll.crypt32, "CryptUnprotectData" if decrypt else "CryptProtectData")
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(Blob)]
    function.restype = ctypes.c_int
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(destination)):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(destination.pbData, destination.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        ctypes.windll.kernel32.LocalFree(destination.pbData)


def prior_inventory():
    """Known previous 5m acquisition/replay footprint; never read new holdout."""
    protocol = read_signed(AUDIT / "preregistration.json")
    files = {"previous_preregistration": digest_file(AUDIT / "preregistration.json"),
             "previous_input_manifest": digest_file(INPUT / "manifest.json")}
    spans = []
    candidates = list((ROOT / "data/autopilot").rglob("5m.json")) + list((ROOT / "output/historical_replay/input").rglob("5m.json"))
    for p in candidates:
        d = read_signed(p)
        items = json.loads(d["frame"])["data"]
        index_name = json.loads(d["frame"])["schema"]["primaryKey"][0]
        dates = pd.to_datetime([r[index_name] for r in items], utc=True)
        first, last = dates.min().isoformat(), dates.max().isoformat()
        if first[:10] <= "2024-08-30" and last[:10] >= "2024-07-01":
            raise ValueError(f"Proposed holdout overlaps previous 5m acquisition: {p}")
        spans.append({"file": str(p.relative_to(ROOT)), "first": first, "last": last, "sha256": digest_file(p)})
    return {"known_5m_inputs": spans, "known_prior_trigger_period": {"start": protocol["start"], "end": protocol["end"]},
            "evidence_files": files, "scope": "NO_PRIOR_LOCAL_5M_REPLAY_FOR_THIS_BLOCK; not proof nobody has ever seen its daily price chart."}


def preregister():
    path = OUTPUT / "preregistration.json"
    if path.exists():
        protocol = read_signed(path)
        for name, h in protocol["experiment_source_sha256"].items():
            if digest_file(ROOT / name) != h:
                raise ValueError("Experimental implementation changed after locking.")
        return protocol
    old = read_signed(AUDIT / "preregistration.json")
    source_names = ["scripts/trigger_experiment_rules.py", "scripts/seal_trigger_experiment_holdout.py",
                    "scripts/run_trigger_experiment.py", "scripts/report_trigger_experiment.py",
                    "tests/test_trigger_experiment.py"]
    protocol = {"version": "TRIGGER_ISOLATED_ABLATION_V2", "registered_at": datetime.now(timezone.utc).isoformat(),
                "supersedes_acquisition_only": "v1 rejected extra daily 2024-09-05 from inclusive endpoint; no holdout opening or arm evaluation; dates/rules unchanged",
                "start": "2024-07-01", "end": "2026-09-25", "symbols": ["SMCI", "NVDA"],
                "development": {"start": old["start"], "end": old["end"], "status": "PREVIOUSLY_INSPECTED_NOT_VALIDATION"},
                "holdout": {"start": "2024-07-01", "end": "2024-08-30", "status": "NEW_5M_BLOCK_ENCRYPTED_OPEN_ONCE",
                            "why": "Outside all inventoried 5m replays; calendar chosen before acquiring outcomes. Retrospective, earlier than development, not nested walk-forward validation."},
                "acquisition": {"provider": "Alpaca SIP", "5m_start": "2024-05-01", "daily_start": "2023-05-01", "end": "2024-09-04",
                                "adjustment_requested": "raw", "cost": "NO_PURCHASE_USD0_INCREMENTAL",
                                "split_policy": "NVDA 10:1 on 2024-06-10 applied ONLY to prior warmup. All experimental cuts after split. SMCI raw, no future September split adjustment."},
                "cohorts": [{"id": f"intraday_{t.replace(':','')}", "time_ny": t, "horizons": [60, 1440]} for t in TIMES],
                "variants": rules_specification(), "parameters": old["parameters"],
                "engine_code_sha256": old["engine_code_sha256"],
                "experiment_source_sha256": {n: digest_file(ROOT / n) for n in source_names},
                "original_input_manifest_sha256": sha(read_signed(INPUT / "manifest.json")),
                "prior_inventory": prior_inventory(), "ledger_sha256": digest_file(ROOT / "data/portfolio.db"),
                "primary_report": "chronological nonoverlapping event book by symbol; 1D primary, 1h secondary; LONG economic results separate from nonexecutable SHORT",
                "deduplication": "same cut+side+entry+TP+SL across horizons ONE event; full max timeout embargo independent of realized outcome",
                "costs": "Frozen commission 25bps/side, slippage 5bps/side, spread 4bps full; ALL assumed, OHLC possible fills NOT real",
                "performance": "Exposure-scaled net bps per hypothetical unit allocated capital, not realized portfolio return; raw coverage and independent books separate. No rates of accuracy, training or model promotion.",
                "opening_gate": "Only after signed complete development results and frozen source tests. Durable registry receipt BEFORE decrypting. Evaluate ALL arms as one batch; no tuning between development and holdout.",
                "portfolio_or_production_changes": False}
    write_signed(path, protocol)
    print("Protocol locked:", sha(protocol), "Holdout:", protocol["holdout"]["start"], protocol["holdout"]["end"], flush=True)
    return protocol


def frame_from(bars):
    f = pd.DataFrame(bars).rename(columns={"o":"Open", "h":"High", "l":"Low", "c":"Close", "v":"Volume"})
    f.index = pd.to_datetime(f.pop("t"), utc=True)
    f = f[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()
    if f.index.duplicated().any() or not np.isfinite(f.to_numpy()).all():
        raise ValueError("Duplicated / nonfinite provider bars.")
    if (f.iloc[:, :4] <= 0).any().any() or (f.Volume <= 0).any() or (f.High < f[["Open","Close","Low"]].max(axis=1)).any() or (f.Low > f[["Open","Close","High"]].min(axis=1)).any():
        raise ValueError("Invalid provider OHLCV.")
    return f


def acquire_and_seal():
    protocol = preregister()
    if (OUTPUT / "holdout_seal.json").exists():
        seal = read_signed(OUTPUT / "holdout_seal.json")
        if seal["protocol_sha256"] != sha(protocol):
            raise ValueError("Existing holdout sealed under different protocol.")
        print("Existing unopened seal verified.")
        return
    key = os.environ.get("APCA_API_KEY_ID") or getpass.getpass("Alpaca key (hidden): ")
    secret = os.environ.get("APCA_API_SECRET_KEY") or getpass.getpass("Alpaca secret (hidden): ")
    session = requests.Session()
    session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    data, audits = {}, {}
    a = protocol["acquisition"]
    schedule = _calendar(2023, 2025).schedule.loc[a["5m_start"]:a["end"]]
    expected = pd.DatetimeIndex([], tz="UTC")
    for s in schedule.itertuples():
        expected = expected.union(pd.date_range(s.open, s.close, freq="5min", inclusive="left"))
    try:
        for symbol in protocol["symbols"]:
            frames, responses = {}, {}
            for interval, start in (("5Min", a["5m_start"]), ("1Day", a["daily_start"])):
                params = {"feed":"sip", "adjustment":"raw", "timeframe":interval, "start":f"{start}T00:00:00-04:00",
                          "end":"2024-09-04T16:00:00-04:00", "limit":10000, "sort":"asc"}
                pages, rows, seen = [], [], set()
                while True:
                    response = session.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/bars", params=params, timeout=45)
                    if response.status_code != 200:
                        raise RuntimeError(f"Market data HTTP {response.status_code}; authentication details redacted.")
                    payload = response.json()
                    if payload.get("symbol") != symbol:
                        raise ValueError("Provider symbol mismatch.")
                    pages.append(payload)
                    rows.extend(payload.get("bars",[]) or [])
                    token = payload.get("next_page_token")
                    if not token:
                        break
                    if token in seen:
                        raise ValueError("Repeated pagination token.")
                    seen.add(token)
                    params["page_token"] = token
                f = frame_from(rows)
                if interval == "5Min":
                    f = f.loc[f.index.isin(expected)]
                    if not f.index.equals(expected):
                        raise ValueError(f"{symbol}: regular 5m coverage fails; missing {len(expected.difference(f.index))}.")
                else:
                    f.index = pd.DatetimeIndex(f.index.tz_convert("America/New_York").date)
                    daily_expected = _calendar(2023,2025).schedule.loc[a["daily_start"]:a["end"]].index
                    if not f.index.equals(daily_expected):
                        raise ValueError(f"{symbol}: daily calendar coverage fails.")
                if symbol == "NVDA":
                    pre = f.index.date < pd.Timestamp("2024-06-10").date()
                    f.loc[pre, ["Open","High","Low","Close"]] /= 10
                    f.loc[pre, "Volume"] *= 10
                frames[interval], responses[interval] = f, pages
                print("Acquired/audited", symbol, interval, len(f), "bars; no outcomes inspected.", flush=True)
            intraday, daily = frames["5Min"], frames["1Day"]
            grouped = intraday.groupby(pd.Index(intraday.index.date)).agg({"Open":"first", "High":"max", "Low":"min", "Close":"last", "Volume":"sum"})
            grouped.index = pd.DatetimeIndex(grouped.index)
            daily.loc[grouped.index] = grouped
            data[symbol] = {"5m": intraday.to_json(orient="table", date_format="iso"), "daily": daily.to_json(orient="table", date_format="iso"),
                            "as_of": schedule.iloc[-1].close.isoformat(), "raw_provider_pages": responses}
            audits[symbol] = {"regular_bars": len(intraday), "sessions": len(schedule), "daily_sessions": len(daily),
                              "missing_regular_bars": 0, "duplicates": 0, "invalid_ohlcv": 0, "calendar":"XNYS", "timezone":"America/New_York",
                              "raw_payload_sha256": sha(responses), "sealed_data_sha256": sha(data[symbol]),
                              "basis": "NVDA post-Jun10-2024 basis, prior bars adjusted; SMCI actual pre-Oct2024 raw prices."}
    finally:
        session.close()
        del key, secret
    plaintext = json.dumps(data, sort_keys=True, separators=(",",":"), allow_nan=False).encode()
    cipher_key, nonce = AESGCM.generate_key(bit_length=256), os.urandom(12)
    ciphertext = nonce + AESGCM(cipher_key).encrypt(nonce, plaintext, AAD)
    protected = protect_key(cipher_key)
    (OUTPUT / "holdout.aesgcm").write_bytes(ciphertext)
    (OUTPUT / "holdout.key.dpapi").write_bytes(protected)
    seal = {"sealed_at": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha(protocol),
            "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(), "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
            "protected_key_sha256": hashlib.sha256(protected).hexdigest(), "algorithm":"AES256GCM + Windows current-user DPAPI", "audits":audits,
            "holdout":protocol["holdout"], "ledger_sha256":digest_file(ROOT / "data/portfolio.db"), "outcomes_inspected":False}
    if seal["ledger_sha256"] != protocol["ledger_sha256"]:
        raise ValueError("Ledger changed during acquisition.")
    write_signed(OUTPUT / "holdout_seal.json", seal)
    print("HOLDOUT SEALED:", sha(seal), "No plaintext holdout files saved.", flush=True)


def decrypt_after_opening(protocol, opening_receipt):
    """Require durable registered receipt; no exploration decrypt entrypoint."""
    from portfolio_tracker.analytics.holdout_registry import verify_holdout_opening
    seal = read_signed(OUTPUT / "holdout_seal.json")
    if seal["protocol_sha256"] != sha(protocol):
        raise ValueError("Different holdout protocol.")
    for symbol in protocol["symbols"]:
        if not verify_holdout_opening(opening_receipt[symbol], symbol=symbol, horizon="TRIGGER_EXPERIMENT_1H_1D_V1",
                                     protocol_sha256=sha(protocol), commitment_sha256=sha(seal)):
            raise ValueError("Missing durable holdout opening receipt.")
    ciphertext = (OUTPUT / "holdout.aesgcm").read_bytes()
    protected = (OUTPUT / "holdout.key.dpapi").read_bytes()
    if digest_file(OUTPUT / "holdout.aesgcm") != seal["ciphertext_sha256"] or digest_file(OUTPUT / "holdout.key.dpapi") != seal["protected_key_sha256"]:
        raise ValueError("Encrypted holdout altered.")
    key = protect_key(protected, decrypt=True)
    plaintext = AESGCM(key).decrypt(ciphertext[:12], ciphertext[12:], AAD)
    if hashlib.sha256(plaintext).hexdigest() != seal["plaintext_sha256"]:
        raise ValueError("Holdout plaintext commitment failed.")
    return json.loads(plaintext)


if __name__ == "__main__":
    acquire_and_seal()
