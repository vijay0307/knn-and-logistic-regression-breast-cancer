"""Deploy the RAG chatbot + dashboard to a Hugging Face Space (run on YOUR machine).

  pip install -U huggingface_hub
  hf auth login                                  # or: export HF_TOKEN=hf_...
  export ANTHROPIC_API_KEY=sk-ant-...
  python deploy/huggingface/deploy.py            # --public to make the Space public

What it does
  1. Builds a small bundle: rag/ + requirements + this Dockerfile + the Space card.
  2. Creates the Space (private by default) and uploads the bundle.
  3. Stores ANTHROPIC_API_KEY and RAG_API_KEYS as encrypted Space *secrets*
     (sent straight from your machine to Hugging Face; never written to the repo).
  4. Prints the chat URL, the dashboard URL and the access key to share.
"""
from __future__ import annotations

import argparse
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build_bundle(dst: Path) -> None:
    shutil.copy(HERE / "Dockerfile", dst / "Dockerfile")
    shutil.copy(HERE / "SPACE_README.md", dst / "README.md")   # Space card (front-matter config)
    shutil.copy(ROOT / "README.md", dst / "PROJECT_README.md")  # indexed as knowledge
    shutil.copy(ROOT / "requirements-rag.txt", dst / "requirements-rag.txt")
    shutil.copytree(ROOT / "rag", dst / "rag", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Ship the latest evaluation results so the dashboard's quality section is populated.
    (dst / "reports").mkdir()
    subprocess.run([sys.executable, "-m", "rag.evaluate", "--offline", "--report",
                    str(dst / "reports" / "rag_eval_offline.json")], cwd=ROOT, check=False,
                   stdout=subprocess.DEVNULL)
    for name in ("rag_eval.json",):
        if (ROOT / "reports" / name).exists():
            shutil.copy(ROOT / "reports" / name, dst / "reports" / name)
    hist = dst / "reports" / "rag_eval_history.jsonl"
    if (ROOT / "reports" / "rag_eval_history.jsonl").exists() and not hist.exists():
        shutil.copy(ROOT / "reports" / "rag_eval_history.jsonl", hist)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="breast-cancer-docs-chatbot", help="Space name")
    ap.add_argument("--public", action="store_true", help="make the Space public (keys are still required)")
    ap.add_argument("--access-key", help="key users enter in the UI (default: random)")
    args = ap.parse_args()

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        sys.exit("Set ANTHROPIC_API_KEY first (export ANTHROPIC_API_KEY=sk-ant-...).")

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.name}"
    access_key = args.access_key or secrets.token_urlsafe(18)

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=not args.public, exist_ok=True)
    api.add_space_secret(repo_id, "ANTHROPIC_API_KEY", anthropic_key, description="Claude API key")
    api.add_space_secret(repo_id, "RAG_API_KEYS", access_key, description="Keys clients send to /chat and /dashboard")

    with tempfile.TemporaryDirectory() as d:
        build_bundle(Path(d))
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=d,
                          commit_message="Deploy RAG chatbot and dashboard")

    host = f"https://{repo_id.replace('/', '-').replace('_', '-').lower()}.hf.space"
    print(f"""
Deployed. The Space builds for 2-4 minutes; watch progress at
  https://huggingface.co/spaces/{repo_id}

  Chat:       {host}/
  Dashboard:  {host}/dashboard
  Access key: {access_key}      (paste into the key box; share only with people you trust)

{"The Space is PRIVATE: viewers must be signed in to Hugging Face with access to it. Re-run with --public to share it with anyone who has the access key." if not args.public else "The Space is public, but chat and metrics still require the access key."}
Rotate the key any time: re-run with --access-key NEW_KEY (or edit the RAG_API_KEYS secret in Space settings).""")


if __name__ == "__main__":
    main()
