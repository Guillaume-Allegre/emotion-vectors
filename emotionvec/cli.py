"""`emotionvec` — local CLI that shells out to Modal.

Commands:
    run          extract + build in one call
    extract      activation extraction only
    build        bank + layer selection only (reads existing activations)
    probe        analyse a text against a run's bank (returns JSON)
    steer        steering α-sweep
    behav-eval   classifier-based behavioural eval over steering results
    upload-corpus   seed /vol/corpus/ from local data/
    gen-corpus   generate a fresh story corpus via OpenAI
    pull         pull a run's artefacts from the Modal volume to local runs/<name>/
    list-vol     list files on the Modal volume
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import os

from .config import DEFAULT_EMOTIONS, MODAL_VOL, PACKAGE_ROOT, RUNS_DIR
from .corpus import local_emotion_stories, local_neutral_stories

MODAL_APP_FILE = Path(__file__).parent / "modal_runner.py"


def _modal_run(entrypoint: str, args: list[str], gpu: str | None = None) -> int:
    cmd = ["modal", "run", f"{MODAL_APP_FILE}::{entrypoint}", *args]
    env = {**os.environ}
    if gpu:
        env["EMOTIONVEC_GPU"] = gpu
    return subprocess.call(cmd, cwd=str(PACKAGE_ROOT), env=env)


def _emotions_arg(em: str | None) -> str:
    if em in (None, "", "default"):
        return "default"
    return em


def cmd_run(args: argparse.Namespace) -> int:
    ex = cmd_extract(args)
    if ex != 0:
        return ex
    return cmd_build(args)


def cmd_extract(args: argparse.Namespace) -> int:
    return _modal_run("cli_extract", [
        "--run-name", args.run_name,
        "--model-name", args.model,
        "--emotions", _emotions_arg(args.emotions),
        "--token-skip", str(args.token_skip),
    ], gpu=getattr(args, "gpu", None))


def cmd_build(args: argparse.Namespace) -> int:
    return _modal_run("cli_build", [
        "--run-name", args.run_name,
        "--emotions", _emotions_arg(args.emotions),
    ])


def cmd_probe(args: argparse.Namespace) -> int:
    if args.generate:
        entry, payload_arg, payload = "cli_generate_probe", "--prompt", args.generate
    else:
        text = args.text
        if text == "-" or (text is None and not sys.stdin.isatty()):
            text = sys.stdin.read()
        if args.file:
            text = Path(args.file).read_text()
        if not text:
            print("need --text, --file, or --generate", file=sys.stderr)
            return 2
        entry, payload_arg, payload = "cli_probe", "--text", text

    modal_args = [
        "--run-name", args.run_name,
        "--model-name", args.model,
        payload_arg, payload,
        "--layer", str(args.layer if args.layer is not None else -1),
    ]
    if args.generate:
        modal_args += ["--max-new-tokens", str(args.max_new_tokens)]

    cmd = ["modal", "run", f"{MODAL_APP_FILE}::{entry}", *modal_args]
    proc = subprocess.run(cmd, cwd=str(PACKAGE_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write("\n---STDERR---\n")
        sys.stderr.write(proc.stderr)
        return proc.returncode
    raw = proc.stdout
    try:
        js = raw.split("<<BEGIN_JSON>>", 1)[1].split("<<END_JSON>>", 1)[0]
        result = json.loads(js)
    except Exception as e:
        sys.stderr.write(raw)
        print(f"failed to parse JSON: {e}", file=sys.stderr)
        return 1

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False))
    sys.stdout.write("\n")
    return 0


def cmd_steer(args: argparse.Namespace) -> int:
    return _modal_run("cli_steer", [
        "--run-name", args.run_name,
        "--model-name", args.model,
        "--emotions", args.emotions or "",
        "--alphas", args.alphas,
        "--layer", str(args.layer if args.layer is not None else -1),
        "--max-new-tokens", str(args.max_new_tokens),
    ])


def cmd_behav(args: argparse.Namespace) -> int:
    return _modal_run("cli_behav", ["--run-name", args.run_name])


def cmd_upload_corpus(args: argparse.Namespace) -> int:
    emo = Path(args.emotion) if args.emotion else local_emotion_stories()
    neu = Path(args.neutral) if args.neutral else local_neutral_stories()
    if not emo.exists() or not neu.exists():
        print(f"missing corpus: {emo}, {neu}", file=sys.stderr)
        return 2
    return _modal_run("cli_upload_corpus", [
        "--emotion-path", str(emo),
        "--neutral-path", str(neu),
    ])


def cmd_gen_corpus(args: argparse.Namespace) -> int:
    script = PACKAGE_ROOT / "data" / "generate_stories.py"
    if not script.exists():
        print(f"{script} not found", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(script)]
    if args.only:
        cmd += ["--only", args.only, "--append"]
    if args.no_neutral:
        cmd += ["--no-neutral"]
    return subprocess.call(cmd)


def cmd_pull(args: argparse.Namespace) -> int:
    dest = Path(args.out) if args.out else PACKAGE_ROOT / RUNS_DIR / args.run_name
    dest.mkdir(parents=True, exist_ok=True)
    # List, then pull each file. `modal volume get` only copies one file at a
    # time but handles the path conversion for us.
    list_proc = subprocess.run(
        ["modal", "run", f"{MODAL_APP_FILE}::cli_list_vol",
         "--prefix", f"runs/{args.run_name}/"],
        cwd=str(PACKAGE_ROOT), capture_output=True, text=True,
    )
    if list_proc.returncode != 0:
        sys.stderr.write(list_proc.stdout + list_proc.stderr)
        return list_proc.returncode
    # modal's local-entrypoint stdout mixes logs + print. Parse lines that look
    # like paths under the volume.
    files = [ln.strip() for ln in list_proc.stdout.splitlines()
             if ln.strip().startswith(f"runs/{args.run_name}/")]
    if not files:
        print(f"(no files under runs/{args.run_name}/ on volume `{MODAL_VOL}`)",
              file=sys.stderr)
        return 1
    print(f"pulling {len(files)} file(s) → {dest}")
    for rel in files:
        local = dest / Path(rel).relative_to(f"runs/{args.run_name}")
        local.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["modal", "volume", "get", MODAL_VOL, rel, str(local), "--force"],
            cwd=str(PACKAGE_ROOT), capture_output=True, text=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(f"[warn] failed {rel}: {proc.stderr}\n")
        else:
            print(f"  <- {rel}")
    return 0


def cmd_list_vol(args: argparse.Namespace) -> int:
    return _modal_run("cli_list_vol", ["--prefix", args.prefix or ""])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emotionvec",
                                description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def _common_run(sp):
        sp.add_argument("--run-name", required=True,
                        help="Short name used under /vol/runs/<NAME>/ on the Modal volume.")
        sp.add_argument("--model", required=True,
                        help="HuggingFace repo id, e.g. Qwen/Qwen3-8B.")
        sp.add_argument("--emotions", default="default",
                        help=f"'default' ({len(DEFAULT_EMOTIONS)} emotions) or "
                             "comma-separated custom list.")

    sp_run = sub.add_parser("run", help="Full pipeline: extract + build.")
    _common_run(sp_run)
    sp_run.add_argument("--token-skip", type=int, default=50)
    sp_run.add_argument("--gpu", default=None,
                        help="Modal GPU spec (e.g. 'A100-40GB', 'A100-80GB', "
                             "'H100', 'H200'). Default: A100-40GB. Use H100 for ≥20B.")
    sp_run.set_defaults(func=cmd_run)

    sp_ex = sub.add_parser("extract", help="Activation extraction only.")
    _common_run(sp_ex)
    sp_ex.add_argument("--token-skip", type=int, default=50)
    sp_ex.add_argument("--gpu", default=None)
    sp_ex.set_defaults(func=cmd_extract)

    sp_bu = sub.add_parser("build", help="Build emotion bank + pick best layer.")
    sp_bu.add_argument("--run-name", required=True)
    sp_bu.add_argument("--emotions", default="default")
    sp_bu.set_defaults(func=cmd_build)

    sp_pr = sub.add_parser("probe", help="Analyse text against a run's bank.")
    sp_pr.add_argument("--run-name", required=True)
    sp_pr.add_argument("--model", required=True)
    group = sp_pr.add_mutually_exclusive_group()
    group.add_argument("--text", nargs="?")
    group.add_argument("--file", type=Path)
    group.add_argument("--generate", metavar="PROMPT",
                       help="Have the model respond, then analyse the response.")
    sp_pr.add_argument("--layer", type=int, default=None)
    sp_pr.add_argument("--max-new-tokens", type=int, default=200)
    sp_pr.add_argument("--json-out", type=Path, default=None)
    sp_pr.set_defaults(func=cmd_probe)

    sp_st = sub.add_parser("steer", help="Steering α-sweep.")
    sp_st.add_argument("--run-name", required=True)
    sp_st.add_argument("--model", required=True)
    sp_st.add_argument("--emotions", default="")
    sp_st.add_argument("--alphas", default="-8,-4,-2,0,2,4,8,12")
    sp_st.add_argument("--layer", type=int, default=None)
    sp_st.add_argument("--max-new-tokens", type=int, default=120)
    sp_st.set_defaults(func=cmd_steer)

    sp_be = sub.add_parser("behav-eval",
                           help="Classifier eval over steering_results.json.")
    sp_be.add_argument("--run-name", required=True)
    sp_be.set_defaults(func=cmd_behav)

    sp_up = sub.add_parser("upload-corpus",
                           help="Upload the local data/ corpus to /vol/corpus/.")
    sp_up.add_argument("--emotion", type=Path, default=None)
    sp_up.add_argument("--neutral", type=Path, default=None)
    sp_up.set_defaults(func=cmd_upload_corpus)

    sp_gc = sub.add_parser("gen-corpus",
                           help="Generate stories via data/generate_stories.py "
                                "(requires OPENAI_API_KEY).")
    sp_gc.add_argument("--only", default=None,
                       help="Comma-separated emotions; appends to existing corpus.")
    sp_gc.add_argument("--no-neutral", action="store_true",
                       help="Keep existing neutral corpus untouched.")
    sp_gc.set_defaults(func=cmd_gen_corpus)

    sp_pl = sub.add_parser("pull",
                           help="Download /vol/runs/<NAME>/ to local runs/<NAME>/.")
    sp_pl.add_argument("--run-name", required=True)
    sp_pl.add_argument("--out", type=Path, default=None)
    sp_pl.set_defaults(func=cmd_pull)

    sp_lv = sub.add_parser("list-vol", help="List files on the Modal volume.")
    sp_lv.add_argument("--prefix", default="")
    sp_lv.set_defaults(func=cmd_list_vol)

    return p


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
