"""Same population/quote rules; acquire only through predefined trade terminal.

V1 raw quotes and protocol are preserved. This acquisition-only amendment
never changes a gate, barrier, timeout, price, cohort or production module.
"""
from datetime import datetime, timezone
import getpass
import os
from pathlib import Path
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.near_miss_quote_audit import OUTPUT as PREVIOUS, verify as verify_original, QuotePath, read_compressed, write_compressed
from scripts.collect_near_miss_quotes import QuoteClient
from scripts.download_massive_history import read_signed,write_signed,sha
from scripts.audit_trigger_replay import digest_file

OUTPUT=ROOT/'output/historical_replay/near_miss_quotes_v2'
TERMINAL={'RESOLVED_QUOTE_COUNTERFACTUAL','NO_ENTRY_DEADLINE','NO_ENTRY_STOP_PASSED','NO_ENTRY_TARGET_PASSED','N_D_TIMEOUT_QUOTE'}


def prepare():
    original=read_signed(PREVIOUS/'method.json')
    verify_original(original)
    if (OUTPUT/'method.json').exists():
        method=read_signed(OUTPUT/'method.json');verify(method);return method
    if list((PREVIOUS/'results').glob('*.json')):
        raise ValueError('Amendment must precede any V1 event closure.')
    method={**original,'version':'ONE_FAILED_GATE_SIP_FIRST_EXIT_V2',
            'registered_at':datetime.now(timezone.utc).isoformat(),
            'original_method_sha256':sha(original),
            'scope':'POST_HOC_DESCRIPTIVE_NO_APPROVAL_NO_REOPENED_HOLDOUT',
            'acquisition_amendment':'V1 whole-horizon acquisition stopped before any event result. Same 31 cuts, quotes, price rules and first-passage labels. Fetch through predetermined TP/SL/timeout or cancellation for BOTH horizons, then stop. No trajectory after actual hypothetical exit; whole-horizon excursions N/D. Cached raw V1 pages reused with original SHA, never replaced. This is not an unseen-data validation.',
            'implementation_sha256':{**original['implementation_sha256'],**{n:digest_file(ROOT/n) for n in ('scripts/near_miss_quote_fast.py','scripts/report_near_miss_quote_fast.py','tests/test_near_miss_quote_fast.py')}},
            'quotes':'All paginated Alpaca SIP quotes in fixed regular windows, stop scanning/fetching only when both original horizons reach a terminal state or mature. No outcome-favourable selection.',
            'excursions':'MFE/MAE from marketable quote entry until first exit. Post-exit full-horizon excursions NOT measured, N/D. No reversal after SL counted as achieved gain.'}
    write_signed(OUTPUT/'method.json',method)
    write_signed(PREVIOUS/'acquisition_superseded_before_results.json',{
        'reason':'Millions of unnecessary post-exit quotes; acquisition-only terminal stopping, same cohort and trading rules',
        'superseded_by_method_sha256':sha(method),'event_results_before_amendment':0,
        'preserved_raw_pages':len(list((PREVIOUS/'quotes').rglob('*.json.gz'))),
        'no_filters_or_contracts_changed':True})
    return method


def verify(method):
    verify_original(read_signed(PREVIOUS/'method.json'))
    for n,h in method['implementation_sha256'].items():
        if digest_file(ROOT/n)!=h:raise ValueError('Frozen near-miss source changed: '+n)


def conclude(path):
    result=path.finish()
    result['mfe_full_horizon_per_share']=None
    result['mae_full_horizon_per_share']=None
    result['excursion_scope']='UNTIL_FIRST_EXIT_ONLY; full-horizon NOT measured'
    return result


def consume(paths,done,quote):
    """Finalized paths are never changed by a later reversal or quote gap."""
    t=pd.Timestamp(quote['t'])
    for h,p in paths.items():
        if h in done:continue
        p.feed(quote)
        if p.state in TERMINAL or t>=p.end:
            done[h]=conclude(p)
    return len(done)==len(paths)


def acquire_event(method,event,client):
    paths={str(c['model']['horizon_minutes']):QuotePath(c) for c in event['contracts']}
    done={};evidence=[];used_quotes=0
    for window_index,window in enumerate(event['quote_windows']):
        params={'feed':'sip','start':window['start'],'end':window['end'],'sort':'asc','limit':10000}
        page=0;tokens=set();page_files=[];acquired=0;last=None
        while True:
            page+=1
            old=PREVIOUS/'quotes'/event['event_id']/str(window_index)/f'page_{page:05d}.json.gz'
            path=OUTPUT/'quotes'/event['event_id']/str(window_index)/f'page_{page:05d}.json.gz'
            if old.exists():
                raw=read_compressed(old)
                if raw['method_sha256']!=method['original_method_sha256']:raise ValueError('Old quote page belongs to different protocol.')
                actual=old
            elif path.exists():
                raw=read_compressed(path);actual=path
                if raw['method_sha256']!=sha(method):raise ValueError('Quote method mismatch.')
            else:
                data=client.get(event['symbol'],params)
                raw={'method_sha256':sha(method),'event_id':event['event_id'],'window_index':window_index,
                     'request':dict(params),'provider':'ALPACA','feed':'sip',
                     'retrieved_at':datetime.now(timezone.utc).isoformat(),'response':data}
                write_compressed(path,raw);actual=path
            if raw['request']!=params:raise ValueError('Different immutable query.')
            qs=raw['response'].get('quotes') or []
            acquired+=len(qs)
            page_files.append({'path_from_root':str(actual.relative_to(ROOT)),'sha256':digest_file(actual),'quotes':len(qs)})
            for q in qs:
                t=pd.Timestamp(q['t'])
                if not pd.Timestamp(window['start'])<=t<=pd.Timestamp(window['end']) or (last is not None and t<last):
                    raise ValueError('Quote outside fixed regular window/ordering.')
                last=t;used_quotes+=1
                if consume(paths,done,q):break
            if len(done)==len(paths):break
            token=raw['response'].get('next_page_token')
            if not token:break
            if token in tokens:raise ValueError('Repeated pagination token.')
            tokens.add(token);params={**params,'page_token':token}
            if page%25==0:print(event['symbol'],event['cut_at'],'window',window_index,'pages',page,'active',list(set(paths)-set(done)),flush=True)
        evidence.append({'window':window,'pages':page_files,'quotes_acquired':acquired,'quotes':acquired,
                         'terminated_by_original_trade_states':len(done)==len(paths)})
        if len(done)==len(paths):break
    for h,p in paths.items():
        if h not in done:done[h]=conclude(p)
    return {'method_sha256':sha(method),'event_id':event['event_id'],'evidence':evidence,
            'quotes_evaluated':used_quotes,'horizons':done,'production_authorized':False,
            'only_counterfactual_ignore_missing_gate':event['missing_condition']}


def run():
    method=prepare()
    pending=[e for e in method['events'] if not (OUTPUT/'results'/f"{e['event_id']}.json").exists()]
    if not pending:print('All results already closed.');return
    key=os.environ.get('APCA_API_KEY_ID') or getpass.getpass('Alpaca key (hidden): ')
    secret=os.environ.get('APCA_API_SECRET_KEY') or getpass.getpass('Alpaca secret (hidden): ')
    client=QuoteClient(key,secret);del key,secret
    errors=[]
    try:
        for i,event in enumerate(method['events'],1):
            dest=OUTPUT/'results'/f"{event['event_id']}.json"
            if dest.exists():continue
            try:
                result=acquire_event(method,event,client)
                write_signed(dest,result)
                print(i,'/',len(method['events']),event['symbol'],event['cut_at'],event['side'],event['missing_condition'],
                      {h:(r['status'],r['classification']) for h,r in result['horizons'].items()},flush=True)
            except Exception as exc:
                reason=str(exc) if isinstance(exc,(RuntimeError,ValueError)) else 'EXCEPTION_REDACTED'
                errors.append({'event_id':event['event_id'],'type':type(exc).__name__,'reason':reason})
                print('N/D',event['event_id'][:12],reason,flush=True)
                if 'HTTP_401' in reason or 'HTTP_403' in reason:break
    finally:client.close()
    verify(method)
    write_signed(OUTPUT/'collection_runs'/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'.json'),
                 {'method_sha256':sha(method),'errors':errors,'ledger_sha256':digest_file(ROOT/'data/portfolio.db')})
    print('COLLECTION_CLOSED','errors',len(errors),flush=True)


if __name__=='__main__':
    run()
