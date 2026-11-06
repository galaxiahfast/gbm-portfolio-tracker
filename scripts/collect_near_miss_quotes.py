"""Acquire only historical SIP quotes for the locked near-miss population."""
from datetime import datetime, timezone
import getpass
import os
from pathlib import Path
import sys
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.near_miss_quote_audit import OUTPUT, prepare, verify, write_compressed, read_compressed, QuotePath
from scripts.download_massive_history import read_signed, write_signed, sha
from scripts.audit_trigger_replay import digest_file


class QuoteClient:
    def __init__(self, key, secret):
        self.http = requests.Session()
        self.http.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        self.last = 0.

    def get(self, symbol, params):
        for attempt in range(5):
            time.sleep(max(0., .4 - (time.monotonic()-self.last)))
            self.last = time.monotonic()
            try:
                r = self.http.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes", params=params, timeout=40)
            except requests.RequestException:
                if attempt == 4:
                    raise RuntimeError("NETWORK_FAILURE_REDACTED") from None
                time.sleep(2**attempt)
                continue
            if r.status_code == 200:
                data = r.json()
                if data.get("symbol") != symbol or not isinstance(data.get("quotes"), (list, type(None))):
                    raise RuntimeError("PROVIDER_SYMBOL_OR_SCHEMA_MISMATCH")
                return data
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(min(20, 2**attempt+1))
                continue
            raise RuntimeError(f"QUOTE_HTTP_{r.status_code}_DETAILS_REDACTED")

    def close(self):
        self.http.close()
        self.http.headers.clear()


def acquire_window(method, event, index, window, client):
    folder = OUTPUT / "quotes" / event["event_id"] / str(index)
    complete_path = folder / "complete.json"
    if complete_path.exists():
        complete = read_signed(complete_path)
        if complete["method_sha256"] != sha(method) or complete["window"] != window:
            raise ValueError("Quote window belongs to another method.")
        return complete
    params = {"feed": "sip", "start": window["start"], "end": window["end"], "sort": "asc", "limit": 10000}
    files, tokens, count, page = [], set(), 0, 0
    first = last = None
    while True:
        page += 1
        path = folder / f"page_{page:05d}.json.gz"
        if path.exists():
            raw = read_compressed(path)
            if raw["request"] != params or raw["method_sha256"] != sha(method):
                raise ValueError("Immutable page query mismatch.")
        else:
            data = client.get(event["symbol"], params)
            raw = {"method_sha256": sha(method), "event_id": event["event_id"], "window_index": index,
                   "provider": "ALPACA", "feed": "sip", "request": dict(params),
                   "retrieved_at": datetime.now(timezone.utc).isoformat(), "response": data}
            write_compressed(path, raw)
        q = raw["response"].get("quotes") or []
        if q:
            begin, finish = pd.Timestamp(q[0]["t"]), pd.Timestamp(q[-1]["t"])
            if begin < pd.Timestamp(window["start"]) or finish > pd.Timestamp(window["end"]) or (last and begin < pd.Timestamp(last)):
                raise ValueError("Provider quotes outside fixed window/order.")
            first = first or q[0]["t"]
            last = q[-1]["t"]
        count += len(q)
        files.append({"path": str(path.relative_to(OUTPUT)), "sha256": digest_file(path), "quotes": len(q)})
        token = raw["response"].get("next_page_token")
        if not token:
            break
        if token in tokens:
            raise ValueError("Repeated pagination token.")
        tokens.add(token)
        params = {**params, "page_token": token}
        if page % 50 == 0:
            print(event["symbol"],event["cut_at"],"window",index,"quotes",count,flush=True)
    complete = {"method_sha256": sha(method), "event_id": event["event_id"], "window": window,
                "pages": files, "quotes": count, "first": first, "last": last}
    write_signed(complete_path, complete)
    return complete


def assess_event(method, event):
    paths = {int(c["model"]["horizon_minutes"]): QuotePath(c) for c in event["contracts"]}
    evidence = []
    for index, window in enumerate(event["quote_windows"]):
        complete_path = OUTPUT / "quotes" / event["event_id"] / str(index) / "complete.json"
        complete = read_signed(complete_path)
        if complete["method_sha256"] != sha(method):
            raise ValueError("Mixed quote method.")
        evidence.append({"window": window, "complete_sha256": sha(complete), "quotes": complete["quotes"]})
        for page in complete["pages"]:
            path = OUTPUT/page["path"]
            if digest_file(path) != page["sha256"]:
                raise ValueError("Compressed quote hash mismatch.")
            raw = read_compressed(path)
            for q in raw["response"].get("quotes") or []:
                t = pd.Timestamp(q["t"])
                if not pd.Timestamp(window["start"]) <= t <= pd.Timestamp(window["end"]):
                    raise ValueError("Quote outside regular session query.")
                for p in paths.values():
                    p.feed(q)
    return {"method_sha256": sha(method), "event_id": event["event_id"],
            "evidence": evidence, "horizons": {str(h): p.finish() for h,p in paths.items()}}


def run():
    method = prepare()
    missing = [e for e in method["events"] if not (OUTPUT/"results"/f"{e['event_id']}.json").exists()]
    if not missing:
        print("All quote results already closed; no API access.")
        return
    key = os.environ.get("APCA_API_KEY_ID") or getpass.getpass("Alpaca key (hidden): ")
    secret = os.environ.get("APCA_API_SECRET_KEY") or getpass.getpass("Alpaca secret (hidden): ")
    client = QuoteClient(key,secret)
    del key,secret
    failures=[]
    try:
        for i,event in enumerate(method["events"],1):
            result = OUTPUT/"results"/f"{event['event_id']}.json"
            if result.exists():
                continue
            try:
                for j,window in enumerate(event["quote_windows"]):
                    acquire_window(method,event,j,window,client)
                assessment = assess_event(method,event)
                write_signed(result,assessment)
                print(i,"/",len(method["events"]),event["symbol"],event["cut_at"],event["side"],event["missing_condition"],
                      {k:(v["status"],v["classification"]) for k,v in assessment["horizons"].items()},flush=True)
            except Exception as exc:
                error = {"event_id":event["event_id"],"type":type(exc).__name__,
                         "reason":str(exc) if isinstance(exc,(RuntimeError,ValueError)) else "EXCEPTION_DETAILS_REDACTED"}
                failures.append(error)
                print("N/D",event["event_id"][:12],error["type"],error["reason"],flush=True)
                if "HTTP_401" in error["reason"] or "HTTP_403" in error["reason"]:
                    break
    finally:
        client.close()
    verify(method)
    run_record={"method_sha256":sha(method),"closed_at":datetime.now(timezone.utc).isoformat(),
                "errors":failures,"ledger_sha256":digest_file(ROOT/"data/portfolio.db")}
    write_signed(OUTPUT/"collection_runs"/(datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")+".json"),run_record)
    print("QUOTE_COLLECTION_FINISHED",len(failures),"errors",flush=True)


if __name__ == "__main__":
    run()
