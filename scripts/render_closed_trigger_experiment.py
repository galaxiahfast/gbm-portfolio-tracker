"""Repair reporting syntax ONLY after both experimental stages are immutable.

The frozen reporter is kept byte-for-byte. This separately archived renderer
fixes one bracket in memory, never evaluates or alters an experimental rule.
"""
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_massive_history import read_signed, write_signed, sha
from scripts.audit_trigger_replay import digest_file
from scripts.seal_trigger_experiment_holdout import OUTPUT
from scripts.run_trigger_experiment import validate_environment


def run():
    protocol = read_signed(OUTPUT / "preregistration.json")
    validate_environment(protocol)
    stages = {s: read_signed(OUTPUT / f"{s}_complete.json")
              for s in ("development", "holdout")}
    for value in stages.values():
        if value["errors"] or value["protocol_sha256"] != sha(protocol):
            raise ValueError("Reporting repair requires both closed stages.")
    reporter = ROOT / "scripts/report_trigger_experiment.py"
    broken = 'r["ambiguous"],r["no_fill"],r["pending"])'
    fixed = 'r["ambiguous"],r["no_fill"],r["pending"]])'
    source = reporter.read_text(encoding="utf-8")
    if source.count(broken) != 1:
        raise ValueError("Unexpected frozen reporter; no broad rewrite allowed.")
    compiled = compile(source.replace(broken, fixed, 1), str(reporter), "exec")
    amendment_path = OUTPUT / "report_render_amendment.json"
    if not amendment_path.exists():
        write_signed(amendment_path, {
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "scope": "REPORTING_SYNTAX_ONLY_AFTER_ALL_EVALUATIONS_CLOSED",
            "protocol_sha256": sha(protocol),
            "closed_stage_sha256": {s: sha(v) for s, v in stages.items()},
            "frozen_reporter_sha256": digest_file(reporter),
            "renderer_sha256": digest_file(Path(__file__)),
            "exact_replacement": {"before": broken, "after": fixed},
            "experiment_reexecuted": False,
        })
    amendment = read_signed(amendment_path)
    if amendment["renderer_sha256"] != digest_file(Path(__file__)):
        raise ValueError("Reporting amendment renderer changed.")
    namespace = {"__file__": str(reporter), "__name__": "closed_report_renderer"}
    exec(compiled, namespace)
    namespace["run"]()
    import shutil
    archive = OUTPUT / "source_archive" / digest_file(Path(__file__)) / Path(__file__).name
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__), archive)
    write_signed(OUTPUT / "report_render_receipt.json", {
        "amendment_sha256": sha(amendment),
        "renderer_sha256": digest_file(Path(__file__)),
        "generated_sha256": {n: digest_file(OUTPUT / n)
                             for n in ("report.json", "reporte.md", "manifest.json")},
    })


if __name__ == "__main__":
    run()
