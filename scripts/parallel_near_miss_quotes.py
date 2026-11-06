"""Resumable multi-process orchestration of the FROZEN V2 quote evaluator.

Only scheduling and global read-only API rate limiting differ. Each event is
evaluated by near_miss_quote_fast.acquire_event without changing its rules.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import getpass
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import near_miss_quote_fast as fast
from scripts import resume_near_miss_quotes as resume
from scripts.download_massive_history import read_signed,write_signed,sha
from scripts.audit_trigger_replay import digest_file

_CLIENT=None
_GATE=None
_NEXT=None


class LimitedClient(fast.QuoteClient):
    def get(self,symbol,params):
        with _GATE:
            now=time.monotonic()
            due=max(now,_NEXT.value+.36)
            _NEXT.value=due
        time.sleep(max(0.,due-time.monotonic()))
        return super().get(symbol,params)


def init_worker(key,secret,gate,next_time):
    global _CLIENT,_GATE,_NEXT
    _GATE,_NEXT=gate,next_time
    _CLIENT=LimitedClient(key,secret)


def evaluate(event,method):
    result=fast.acquire_event(method,event,_CLIENT)
    dest=fast.OUTPUT/'results'/f"{event['event_id']}.json"
    write_signed(dest,result)
    return {'event_id':event['event_id'],'symbol':event['symbol'],'cut_at':event['cut_at'],
            'side':event['side'],'condition':event['missing_condition'],
            'horizons':{h:(r['status'],r['classification']) for h,r in result['horizons'].items()},
            'evaluated_quotes':result['quotes_evaluated']}


def main():
    method=read_signed(fast.OUTPUT/'method.json')
    original=read_signed(resume.PREVIOUS/'method.json')
    prior=resume.sources_and_results(method,original)
    ledger=digest_file(resume.LEDGER)
    transition=read_signed(resume.AMENDMENT)
    originals=transition['existing_results_before_resume_sha256']
    # The running portfolio application may change its ledger independently.
    # This read-only acquisition never opens it; retain both checksums instead
    # of claiming that the database stayed unchanged across wall-clock days.
    if any(prior.get(name)!=h for name,h in originals.items()):
        raise ValueError('Original three quote results changed.')
    if (fast.OUTPUT/'resumption_status.json').exists():
        raise ValueError('Already fully closed.')
    pending=[e for e in method['events'] if not (fast.OUTPUT/'results'/f"{e['event_id']}.json").exists()]
    plan_file=fast.OUTPUT/'parallel_acquisition_plan.json'
    if not plan_file.exists():
        write_signed(plan_file,{'registered_at':datetime.now(timezone.utc).isoformat(),
            'method_sha256':sha(method),'transition_sha256':sha(transition),
            'orchestrator_sha256':digest_file(Path(__file__)),
            'frozen_evaluator_sha256':method['implementation_sha256']['scripts/near_miss_quote_fast.py'],
            'initial_results_sha256':prior,'max_workers':4,'global_request_spacing_seconds':.36,
            'ledger_sha256_at_start':ledger,
            'prior_ledger_sha256':transition['resumption_ledger_sha256'],
            'scope':'historical Alpaca SIP GET only; same locked quote evaluator, inputs, gates, outcomes and per-event artifact',
            'no_training_or_production_change':True})
    plan=read_signed(plan_file)
    if plan['method_sha256']!=sha(method) or plan['orchestrator_sha256']!=digest_file(Path(__file__)):
        raise ValueError('Changed parallel acquisition plan.')
    print('PENDING',len(pending),'CLOSED',len(method['events'])-len(pending),flush=True)
    if not pending:return
    key=os.environ.get('APCA_API_KEY_ID') or getpass.getpass('Alpaca key (hidden): ')
    secret=os.environ.get('APCA_API_SECRET_KEY') or getpass.getpass('Alpaca secret (hidden): ')
    failures=[]
    with mp.Manager() as manager:
        lock=manager.Lock()
        next_time=manager.Value('d',0.)
        with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=(key,secret,lock,next_time)) as pool:
            del key,secret
            futures={pool.submit(evaluate,e,method):e for e in pending}
            for future in as_completed(futures):
                e=futures[future]
                try:
                    item=future.result()
                    print('CLOSED',item['symbol'],item['cut_at'],item['side'],item['condition'],
                          item['horizons'],'quotes',item['evaluated_quotes'],flush=True)
                except Exception as exc:
                    reason=str(exc) if isinstance(exc,(RuntimeError,ValueError)) else 'EXCEPTION_DETAILS_REDACTED'
                    failures.append({'event_id':e['event_id'],'type':type(exc).__name__,'reason':reason})
                    print('N/D',e['event_id'][:12],reason,flush=True)
    ledger_after=digest_file(resume.LEDGER)
    closed=sum((fast.OUTPUT/'results'/f"{e['event_id']}.json").exists() for e in method['events'])
    write_signed(fast.OUTPUT/'parallel_acquisition_result.json',{
        'plan_sha256':sha(plan),'closed':closed,'total':len(method['events']),
        'failures':failures,'ledger_at_start_sha256':ledger,
        'ledger_at_finish_sha256':ledger_after,
        'ledger_was_accessed_by_acquisition':False,
        'completed_at':datetime.now(timezone.utc).isoformat()})
    print('DONE',closed,'/',len(method['events']),'failures',len(failures),flush=True)


if __name__=='__main__':
    main()
