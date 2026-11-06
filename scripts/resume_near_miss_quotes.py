"""Resume read-only historical quotes after external ledger drift.

The original accounting SHA guard correctly detects that portfolio.db changed
while this desktop task was not running. This wrapper does NOT rewrite or read
accounting records. It documents the transition, verifies all frozen replay
inputs/source/result hashes and requires the *new* ledger checksum to remain
stable throughout this acquisition. Original trading/quote code is unchanged.
"""
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import near_miss_quote_fast as fast
from scripts.near_miss_quote_audit import OUTPUT as PREVIOUS
from scripts.download_massive_history import read_signed,write_signed,sha
from scripts.audit_trigger_replay import digest_file

LEDGER=ROOT/'data/portfolio.db'
AMENDMENT=fast.OUTPUT/'ledger_transition_read_only.json'


def sources_and_results(method,original):
    if method['original_method_sha256']!=sha(original):
        raise ValueError('V2 population no longer matches original lock.')
    mismatches=[]
    for name,expected in {**original['source_files_sha256'],**method['implementation_sha256']}.items():
        file=ROOT/name
        if not file.exists() or digest_file(file)!=expected:
            mismatches.append(name)
    if mismatches:
        raise ValueError('Historical source changed: '+', '.join(mismatches))
    result_files={}
    for e in method['events']:
        path=fast.OUTPUT/'results'/f"{e['event_id']}.json"
        if path.exists():
            item=read_signed(path)
            if item['method_sha256']!=sha(method) or item['event_id']!=e['event_id']:
                raise ValueError('Existing quote result belongs to another experiment.')
            result_files[str(path.relative_to(ROOT))]=digest_file(path)
    return result_files


def run():
    method=read_signed(fast.OUTPUT/'method.json')
    original=read_signed(PREVIOUS/'method.json')
    prior=sources_and_results(method,original)
    current=digest_file(LEDGER)
    if original['original_ledger_sha256']==current:
        raise ValueError('Ledger did not drift; use the original guarded collector.')
    if AMENDMENT.exists():
        amendment=read_signed(AMENDMENT)
        originals=amendment['existing_results_before_resume_sha256']
        if (amendment['resumption_ledger_sha256']!=current
                or any(prior.get(name)!=expected for name,expected in originals.items())):
            raise ValueError('Ledger or closed results changed again; require a new documented transition.')
    else:
        amendment={'recorded_at':datetime.now(timezone.utc).isoformat(),
                   'reason':'portfolio.db SHA changed while the computer was off/outside this acquisition; cause not inferred',
                   'initial_ledger_sha256':original['original_ledger_sha256'],
                   'resumption_ledger_sha256':current,
                   'original_method_sha256':sha(original),'resumed_method_sha256':sha(method),
                   'existing_results_before_resume_sha256':prior,
                   'wrapper_source_sha256':digest_file(Path(__file__)),
                   'guard':'verify all frozen replay sources and old results; checksum-only accounting DB before/after; no SQLite access or write',
                   'replay_or_filter_change':False,'new_holdout_opening':False}
        write_signed(AMENDMENT,amendment)
    # Only this offline collector's module-level verifier is replaced. No
    # production globals/functions, source file or database is monkeypatched.
    def verify_historical(value):
        if sha(value)!=sha(original):
            raise ValueError('Different original historical method.')
        sources_and_results(method,original)
        if digest_file(LEDGER)!=current:
            raise ValueError('Accounting checksum changed during read-only acquisition.')
    fast.verify_original=verify_historical
    fast.run()
    if digest_file(LEDGER)!=current:
        raise ValueError('Accounting checksum changed during this run; do not certify ledger unchanged.')
    closed=sum((fast.OUTPUT/'results'/f"{e['event_id']}.json").exists() for e in method['events'])
    status={
        'amendment_sha256':sha(amendment),'method_sha256':sha(method),
        'closed':closed,'total':len(method['events']),
        'resumption_ledger_sha256':current,'ledger_unchanged_during_resumed_run':True}
    name='resumption_status.json' if closed==len(method['events']) else 'resumption_progress_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'.json'
    write_signed(fast.OUTPUT/name,status)
    print('RESUMED_CLOSED',closed,'/',len(method['events']),flush=True)


if __name__=='__main__':
    run()
