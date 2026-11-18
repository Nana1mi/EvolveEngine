"""Command-line entry points for EvolveEngine."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import sys

from .controller import Controller, Lane
from .experiment import MOCK_PROTOCOL, deploy_pair, evaluate_pair, seal_pair, verify_submission
from .journal import Interrupted, Journal
from .mock import MockBackend
from .partitions import Partitions
from .reports import summarize
from .schema import digest
from .spark import SparkBackend, data_partitions


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def mock(args):
    root = args.output
    backend = MockBackend(args.scenario)
    key = f"simulation-only/{args.seed}/{args.replicate}".encode()
    salt = hashlib.sha256(key + b"/development").digest()
    final_salt = hashlib.sha256(key + b"/final").digest()
    partitions = Partitions(args.seed, salt, final_salt=final_salt)

    def interrupt(key):
        if args.interrupt_at == key:
            raise Interrupted(key)

    with Journal(root / "journal.sqlite3", interrupt) as journal:
        report = deploy_pair(
            journal, backend, partitions, MOCK_PROTOCOL,
            ("mock-spark-1", "mock-spark-2", "mock-spark-3", "mock-spark-4"),
            args.replicate, {"backend": backend.version, "scenario": args.scenario, "simulated": True},
            ablation=args.ablation,
        )
    write_json(root / "report.json", report)
    decisions = [event["status"] for event in report["events"]
                 if event["id"].endswith("/decided") and "/window-" in event["id"]]
    return {"simulated": True, "report": str((root / "report.json").resolve()),
            "harness_decisions": decisions,
            "rq1": report["sealed_evaluation"]["rq1"].get("engines", {})}


def read_json(path):
    return Path(path).read_text(encoding="utf-8-sig")


def keygen(args):
    path = args.path
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(key)
    return {"seal_key_path": str(path.resolve()), "commitment": hashlib.sha256(key).hexdigest()}


def token_ids(args):
    tokenizer = json.loads(read_json(args.tokenizer))
    special = {item["id"] for item in tokenizer.get("added_tokens", []) if item.get("special")}
    vocabulary = tokenizer["model"]["vocab"]
    ids = sorted({token for token in vocabulary.values() if type(token) is int and token not in special})
    if len(ids) < 32:
        raise ValueError("Expected a pinned byte-pair vocabulary with ordinary token IDs")
    value = {"token_ids": ids[:2048], "special_prefix": []}
    write_json(args.output, value)
    return {"digest": digest(value), "output": str(args.output), "count": len(value["token_ids"])}


def live(args):
    config = json.loads(read_json(args.config))
    backend = SparkBackend(config)
    partitions = data_partitions(config, args.seed, args.salt_file.read_bytes(),
                                 final_commitment=getattr(args, "final_commitment", None))
    if args.command == "preflight":
        evidence = backend.pilot_preflight(partitions, args.node)
        write_json(args.output, evidence)
        return {"passed": True, "nodes": list(evidence["nodes"]), "receipt": str(args.output)}
    evidence = json.loads(read_json(args.preflight_receipt))
    if evidence["configuration_digest"] != digest(config):
        raise ValueError("Preflight receipt does not match the controller config")
    nodes = [args.node] if args.command == "optimize" else list(backend.nodes)
    if any(node not in evidence["nodes"] for node in nodes):
        raise ValueError("Every selected Spark node requires a passing receipt")
    backend.pi.preflight()
    fingerprints = sorted({fingerprint for row in evidence["agent"]["metered_probe"]
                           for fingerprint in row.get("fingerprints", [])})
    environment = {"backend": backend.version, "configuration": digest(config),
                   "nodes": {name: {key: attrs[key] for key in
                                    ("driver", "runtime", "runtime_image", "engine_revision", "model_digest")}
                             for name, attrs in evidence["nodes"].items()},
                   "agent": {"pi": backend.pi.version, "model": backend.provider.model,
                             "fingerprint": backend.provider.fingerprint,
                             "observed_preflight_fingerprints": fingerprints},
                   "noise_epsilon": config["noise_epsilon"]}
    with Journal(args.output / "journal.sqlite3", enforce_wall=True) as journal:
        if args.command == "optimize":
            if args.steps <= 0:
                raise ValueError("Single-node optimization needs a positive step count")
            controller = Controller(journal, backend, partitions, environment,
                                    epsilon=environment["noise_epsilon"])
            start = controller.checkpoint(backend.initial_engine(), "p1")
            lane = Lane("p1", "fixed", args.node, backend.protocol.main)
            journal.event("p1/protocol", {"partitions": partitions.public_manifest()})
            submission = controller.run(lane, start, args.steps)
            report = {"submission": submission, "events": journal.events(""),
                      "engine_checkpoints": journal.objects("engine"),
                      "budget_ledger": list(journal.budget_rows()), "experience": journal.objects("experience")}
        else:
            bundle, report = seal_pair(journal, backend, partitions, backend.protocol, backend.nodes,
                                       args.replicate, environment, causal_points=args.causal_points,
                                       ablation=args.ablation)
            write_json(args.output / "sealed_submission.json", bundle)
    write_json(args.output / "report.json", report)
    result = {"simulated": False, "report": str(args.output / "report.json"), "submission": report["submission"]}
    if args.command == "pilot":
        result.update(sealed_submission=str(args.output / "sealed_submission.json"),
                      submission_digest=bundle["digest"])
    return result


def evaluate(args):
    bundle = json.loads(read_json(args.submission))
    payload = verify_submission(bundle, args.submission_digest)
    config = json.loads(read_json(args.config))
    if payload["environment"]["configuration"] != digest(config):
        raise ValueError("Judge controller config differs from the sealed configuration")
    manifest = payload["run_manifest"]["partitions"]
    partitions = data_partitions(config, manifest["seed"], final_salt=args.final_salt_file.read_bytes(),
                                 development_commitment=manifest["development_commitment"],
                                 final_commitment=manifest["final_commitment"])
    backend = SparkBackend(config, evaluation_only=True)
    with Journal(args.output / "judge.sqlite3", enforce_wall=True) as journal:
        report = evaluate_pair(journal, backend, partitions, backend.protocol, bundle, args.submission_digest)
    path = args.output / "sealed_evaluation.json"
    write_json(path, report)
    return {"report": str(path), "submission_digest": args.submission_digest,
            "rq1_status": report["sealed_evaluation"]["rq1"]["status"]}


def main():
    parser = argparse.ArgumentParser(prog="python -m evolveengine")
    command = parser.add_subparsers(dest="command", required=True)
    simulate = command.add_parser("mock", help="run the complete offline paired simulation")
    simulate.add_argument("--output", type=Path, default=Path(".evolveengine/mock"))
    simulate.add_argument("--seed", type=int, default=21001)
    simulate.add_argument("--replicate", type=int, default=0)
    simulate.add_argument("--scenario", default="accept",
                          choices=("accept", "reject", "deviation", "quality_failure"))
    simulate.add_argument("--ablation", action="store_true")
    simulate.add_argument("--interrupt-at", help="fault injection at the given operation ID")
    simulate.set_defaults(handler=mock)
    new_key = command.add_parser("new-seal-key", help="create private salts for independent request pools")
    new_key.add_argument("path", type=Path)
    new_key.set_defaults(handler=keygen)
    tokens = command.add_parser("make-token-ids", help="freeze an ordinary token set from the Qwen tokenizer")
    tokens.add_argument("--tokenizer", type=Path, required=True)
    tokens.add_argument("--output", type=Path, required=True)
    tokens.set_defaults(handler=token_ids)
    hash_json = command.add_parser("hash-config", help="print the canonical digest of a JSON manifest")
    hash_json.add_argument("path", type=Path)
    hash_json.set_defaults(handler=lambda args: {"digest": digest(json.loads(read_json(args.path)))})
    stats = command.add_parser("summarize", help="aggregate sealed independent replicate reports")
    stats.add_argument("reports", type=Path, nargs="+")
    stats.set_defaults(handler=lambda args: summarize([json.loads(read_json(path)) for path in args.reports]))
    for name in ("preflight", "optimize", "pilot"):
        stage = command.add_parser(name, help={"preflight": "check pinned Spark/Pi runtime",
                                               "optimize": "run P1 on a single Spark",
                                               "pilot": "run a paired Spark trial"}[name])
        stage.add_argument("--config", type=Path, required=True)
        stage.add_argument("--salt-file", type=Path, required=True)
        stage.add_argument("--seed", type=int, default=21001)
        stage.add_argument("--output", type=Path, required=True)
        if name == "preflight":
            stage.add_argument("--node", action="append", help="probe only these configured nodes")
        else:
            stage.add_argument("--preflight-receipt", type=Path, required=True)
        if name == "optimize":
            stage.add_argument("--node", required=True)
            stage.add_argument("--steps", type=int, default=2)
        if name == "pilot":
            stage.add_argument("--final-commitment", required=True,
                               help="public SHA-256 of an independent key held by the judge")
            stage.add_argument("--replicate", type=int, default=0)
            stage.add_argument("--causal-points", type=int, choices=(1, 3), default=1)
            stage.add_argument("--ablation", action="store_true")
        stage.set_defaults(handler=live)
    judge = command.add_parser("evaluate", help="judge only precommitted engine submissions with private Q/T")
    judge.add_argument("--config", type=Path, required=True)
    judge.add_argument("--submission", type=Path, required=True)
    judge.add_argument("--submission-digest", required=True)
    judge.add_argument("--final-salt-file", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.set_defaults(handler=evaluate)
    args = parser.parse_args()
    try:
        result = args.handler(args)
    except Interrupted as failure:
        print(f"Simulated interruption after reservation: {failure}", file=sys.stderr)
        return 75
    except (ValueError, OSError, KeyError, TypeError, sqlite3.Error, subprocess.SubprocessError) as failure:
        print(f"EvolveEngine: {failure}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
