"""Close the historical SIP near-miss audit without hiding external ledger drift.

The frozen population, quote evaluator, and report calculations stay unchanged.
This wrapper only verifies complete inputs and corrects the report's accounting
checksum disclosure after the desktop portfolio database changed independently.
"""
from datetime import datetime, timezone
from pathlib import Path
from types import FunctionType
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import near_miss_quote_fast as fast
from scripts import report_near_miss_quotes as original
from scripts import report_near_miss_quote_fast as report_fast
from scripts import resume_near_miss_quotes as resume
from scripts.download_massive_history import read_signed, write_signed, sha
from scripts.audit_trigger_replay import digest_file


def main():
    method = read_signed(fast.OUTPUT / 'method.json')
    initial = read_signed(resume.PREVIOUS / 'method.json')
    prior = resume.sources_and_results(method, initial)
    transition = read_signed(resume.AMENDMENT)
    for name, expected in transition['existing_results_before_resume_sha256'].items():
        if prior.get(name) != expected:
            raise ValueError('An original pre-resumption result changed.')
    missing = [e['event_id'] for e in method['events']
               if not (fast.OUTPUT / 'results' / f"{e['event_id']}.json").exists()]
    if missing:
        raise ValueError(f'Quote acquisition incomplete: {len(missing)} events remain.')
    if (fast.OUTPUT / 'report.json').exists():
        raise ValueError('Closed report already exists; do not overwrite signed report.')

    ledger_before = digest_file(resume.LEDGER)
    original_line = (f"portfolio.db SHA256 antes/después: "
                     f"`{method['original_ledger_sha256']}`. Solo se calcula checksum; "
                     'sin abrir tabla, orden ni registro contable.')
    truthful_line = (
        'portfolio.db: SHA256 al preregistro '
        f"`{method['original_ledger_sha256']}`; al reanudar "
        f"`{transition['resumption_ledger_sha256']}`; al cerrar el informe "
        f'`{ledger_before}`. Cambió fuera de esta adquisición read-only; '
        'no se infiere la causa. Esta auditoría solo calcula el checksum, '
        'sin abrir ni modificar tablas, órdenes o registros contables.'
    )

    def verify_frozen(value):
        if sha(value) != sha(method):
            raise ValueError('Report method changed.')
        resume.sources_and_results(method, initial)

    def write_with_provenance(path, value):
        if Path(path).name == 'report.json':
            value = dict(value)
            value['production_unchanged'] = False  # Ledger changed externally.
            value['production_code_unchanged_by_this_audit'] = True
            value['accounting_database_accessed_by_this_audit'] = False
            value['ledger_sha256_at_original_preregistration'] = method['original_ledger_sha256']
            value['ledger_sha256_at_resumption'] = transition['resumption_ledger_sha256']
            value['ledger_sha256_at_report_start'] = ledger_before
        write_signed(path, value)

    def publish_with_provenance(path, body):
        if original_line not in body:
            raise ValueError('Expected original ledger claim not present.')
        report_fast.publish(path, body.replace(original_line, truthful_line))

    namespace = dict(original.run.__globals__)
    namespace.update(OUTPUT=fast.OUTPUT, verify=verify_frozen,
                     write_signed=write_with_provenance,
                     artifact=publish_with_provenance)
    runner = FunctionType(original.run.__code__, namespace,
                          original.run.__name__, original.run.__defaults__,
                          original.run.__closure__)
    runner()

    report = read_signed(fast.OUTPUT / 'report.json')
    pages = {}
    for event in report['events']:
        result = event['quote_result']
        if result is None:
            raise ValueError('Report contains a missing quote result.')
        for window in result['evidence']:
            for page in window['pages']:
                name = page['path_from_root']
                if digest_file(ROOT / name) != page['sha256']:
                    raise ValueError('Raw SIP quote page hash mismatch: ' + name)
                pages[name] = page['sha256']
    write_signed(fast.OUTPUT / 'quote_inventory.json', {
        'report_sha256': sha(report), 'used_raw_pages_sha256': pages,
        'method_sha256': sha(method),
        'ledger_sha256_at_quote_inventory': digest_file(resume.LEDGER),
    })
    write_signed(fast.OUTPUT / 'closing_receipt.json', {
        'closed_at': datetime.now(timezone.utc).isoformat(),
        'method_sha256': sha(method), 'total_events': len(method['events']),
        'signed_result_files': len(prior), 'raw_quote_pages_verified': len(pages),
        'original_report_source_sha256': digest_file(Path(original.__file__)),
        'v2_report_source_sha256': digest_file(Path(report_fast.__file__)),
        'closing_wrapper_sha256': digest_file(Path(__file__)),
        'report_sha256': digest_file(fast.OUTPUT / 'report.json'),
        'markdown_sha256': digest_file(fast.OUTPUT / 'reporte.md'),
        'quote_inventory_sha256': digest_file(fast.OUTPUT / 'quote_inventory.json'),
        'ledger_sha256_at_start': ledger_before,
        'ledger_sha256_at_end': digest_file(resume.LEDGER),
        'accounting_database_accessed_by_this_audit': False,
    })
    print('CLOSED_REPORT', len(method['events']), 'events', len(pages), 'raw pages')


if __name__ == '__main__':
    main()
