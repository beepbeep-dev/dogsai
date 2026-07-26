#!/usr/bin/env python3
"""Rent a GPU on Vast.ai, train dogsai on it, bring the checkpoint home.

    # what would this cost?
    python scripts/vast_train.py offers

    # launch (prints the price and asks for --yes before spending anything)
    python scripts/vast_train.py launch --dataset dogbehaviour --preset small --yes

    # follow it, then collect
    python scripts/vast_train.py status
    python scripts/vast_train.py fetch-run
    python scripts/vast_train.py destroy

The API key is read from ``$VAST_API_KEY`` or ``~/.vast_api_key`` and is never
written to the repo or printed. Credentials do not belong in version control:
keep the key in the environment or that file, both of which are gitignored here.

Design notes
------------
* The instance downloads the dataset itself, straight from HuggingFace. Pushing
  ~10 GB up from a laptop is slower and costs GPU-hours while it uploads.
* ``--interruptible`` bids on spare capacity at roughly a third of the on-demand
  price. Training here checkpoints every epoch, so an interruption costs one
  epoch, not the run.
* Nothing is created without ``--yes``. Renting hardware spends real money, and a
  script that can do that silently is a script that eventually does.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://console.vast.ai/api/v0"
STATE_PATH = Path.home() / ".dogsai_vast_run.json"

# Provisioning script run on the instance at boot. Kept dependency-light and
# idempotent so a re-run after an interruption resumes rather than restarts.
ONSTART = r"""#!/bin/bash
set -euo pipefail
exec > >(tee -a /root/dogsai_run.log) 2>&1
echo "=== dogsai provisioning $(date -u) ==="

export DEBIAN_FRONTEND=noninteractive
export PIP_ROOT_USER_ACTION=ignore
python3 -m pip install -q --upgrade pip
python3 -m pip install -q av opencv-python-headless huggingface_hub tqdm numpy

cd /root
if [ ! -d dogsai ]; then
  git clone --depth 1 --branch "__BRANCH__" "__REPO__" dogsai
fi
cd dogsai
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# Fetch + convert the dataset (resumable; skips what is already there).
python3 -m dogsai.cli fetch "__DATASET__" \
  --raw-root /root/data/raw --out /root/data/prepared --workers 16

python3 -m dogsai.cli audit --data-root /root/data/prepared \
  --behaviours /root/data/prepared/behaviours.txt --no-duplicate-check || true

python3 -m dogsai.cli train \
  --data-root /root/data/prepared \
  --behaviours /root/data/prepared/behaviours.txt \
  --out-dir /root/runs/dognet \
  --set model.preset=__PRESET__ \
  --set train.epochs=__EPOCHS__ \
  --set train.batch_size=__BATCH__ \
  --set data.num_workers=__WORKERS__ \
  --set train.amp=true \
  --set train.compile=false

python3 -m dogsai.cli eval /root/runs/dognet/best.pt \
  --data-root /root/data/prepared --split val \
  --json /root/runs/dognet/val_metrics.json || true

python3 -m dogsai.cli export /root/runs/dognet/best.pt /root/runs/dognet/dognet.ts.pt || true
touch /root/DOGSAI_DONE
echo "=== dogsai finished $(date -u) ==="
"""


# ---------------------------------------------------------------------------
# api plumbing
# ---------------------------------------------------------------------------
def api_key() -> str:
    key = os.environ.get("VAST_API_KEY", "").strip()
    if not key:
        path = Path.home() / ".vast_api_key"
        if path.exists():
            key = path.read_text().strip()
    if not key:
        raise SystemExit(
            "no API key: set VAST_API_KEY or write it to ~/.vast_api_key\n"
            "(never commit it — both locations are outside the repo)"
        )
    return key


def call(path: str, method: str = "GET", payload: dict | None = None) -> dict:
    url = f"{API}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {api_key()}")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:400]
        raise SystemExit(f"vast api {exc.code} on {method} {path}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach the vast api: {exc.reason}") from exc
    return json.loads(body) if body.strip() else {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")


def load_state() -> dict:
    if not STATE_PATH.exists():
        raise SystemExit("no recorded run; launch one first")
    return json.loads(STATE_PATH.read_text())


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def find_offers(args) -> list[dict]:
    """Search bookable machines matching the requirements, cheapest first."""
    query = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "num_gpus": {"eq": args.gpus},
        "gpu_ram": {"gte": args.min_vram * 1000},
        "disk_space": {"gte": args.disk},
        "inet_down": {"gte": 100},
        "cuda_max_good": {"gte": 12.0},
        "type": "bid" if args.interruptible else "on-demand",
        "order": [["dph_total", "asc"]],
        "limit": 40,
    }
    if args.gpu_name:
        query["gpu_name"] = {"eq": args.gpu_name}
    result = call("/bundles/", method="PUT", payload={"q": query})
    offers = result.get("offers", [])
    return offers


def cmd_offers(args) -> int:
    offers = find_offers(args)
    if not offers:
        print("no matching offers; try --min-vram 16 or --gpu-name ''")
        return 1
    print(f"{len(offers)} offers (cheapest first)\n")
    print(f"{'id':>10}  {'gpu':<22}{'n':>2} {'vram':>6} {'disk':>6} "
          f"{'$/hr':>7} {'down':>7}  location")
    for offer in offers[:15]:
        print(
            f"{offer['id']:>10}  {offer.get('gpu_name', '?'):<22}"
            f"{offer.get('num_gpus', 1):>2} "
            f"{offer.get('gpu_ram', 0) / 1000:>5.0f}G "
            f"{offer.get('disk_space', 0):>5.0f}G "
            f"{offer.get('dph_total', 0):>7.3f} "
            f"{offer.get('inet_down', 0):>6.0f}M  "
            f"{offer.get('geolocation', '?')}"
        )
    cheapest = offers[0]
    hours = args.hours
    print(
        f"\ncheapest: ${cheapest.get('dph_total', 0):.3f}/hr "
        f"-> ~${cheapest.get('dph_total', 0) * hours:.2f} for {hours}h "
        f"on a {cheapest.get('gpu_name')}"
    )
    print("storage is billed separately and is usually a few cents per hour.")
    return 0


def cmd_launch(args) -> int:
    offers = find_offers(args)
    if not offers:
        print("no matching offers", file=sys.stderr)
        return 1
    offer = offers[0]
    price = offer.get("dph_total", 0)
    estimate = price * args.hours

    print(f"offer {offer['id']}: {offer.get('num_gpus')}x {offer.get('gpu_name')}, "
          f"{offer.get('disk_space', 0):.0f}G disk, {offer.get('geolocation')}")
    print(f"price: ${price:.3f}/hr  ->  ~${estimate:.2f} for an estimated {args.hours}h run")
    if not args.yes:
        print("\nthis will rent real hardware and cost real money.")
        print("re-run with --yes to actually create the instance.")
        return 0

    onstart = (
        ONSTART.replace("__REPO__", args.repo)
        .replace("__BRANCH__", args.branch)
        .replace("__DATASET__", args.dataset)
        .replace("__PRESET__", args.preset)
        .replace("__EPOCHS__", str(args.epochs))
        .replace("__BATCH__", str(args.batch))
        .replace("__WORKERS__", str(args.loader_workers))
    )
    payload = {
        "client_id": "me",
        "image": args.image,
        "disk": args.disk,
        "onstart": onstart,
        "runtype": "ssh",
        "label": "dogsai-train",
    }
    if args.interruptible:
        payload["price"] = round(price * 1.15, 4)  # small headroom over the ask

    created = call(f"/asks/{offer['id']}/", method="PUT", payload=payload)
    if not created.get("success", False):
        print(f"launch failed: {created}", file=sys.stderr)
        return 1
    instance_id = created.get("new_contract")
    save_state({
        "instance_id": instance_id,
        "offer": offer["id"],
        "price_per_hour": price,
        "started": time.time(),
        "gpu": offer.get("gpu_name"),
        "preset": args.preset,
        "dataset": args.dataset,
    })
    print(f"\ninstance {instance_id} created. it boots, provisions, then trains.")
    print("  follow:   python scripts/vast_train.py status")
    print("  collect:  python scripts/vast_train.py fetch-run")
    print("  STOP THE BILLING when done:  python scripts/vast_train.py destroy")
    return 0


def _instance(instance_id: int) -> dict:
    payload = call(f"/instances/{instance_id}/")
    return payload.get("instances", payload) or {}


def cmd_status(args) -> int:
    state = load_state()
    instance = _instance(state["instance_id"])
    if not instance:
        print("instance not found (already destroyed?)")
        return 1
    elapsed = (time.time() - state["started"]) / 3600
    print(f"instance {state['instance_id']}  {instance.get('actual_status', '?')}  "
          f"({instance.get('status_msg') or 'no message'})")
    print(f"  gpu      : {instance.get('gpu_name')} x{instance.get('num_gpus', 1)}")
    print(f"  ssh      : ssh -p {instance.get('ssh_port')} root@{instance.get('ssh_host')}")
    print(f"  elapsed  : {elapsed:.2f}h   spent so far ~${elapsed * state['price_per_hour']:.2f}")
    print(f"\n  training log:  ssh ... 'tail -f /root/dogsai_run.log'")
    print(f"  done marker :  /root/DOGSAI_DONE")
    return 0


def cmd_fetch_run(args) -> int:
    """Print the scp commands to pull the run directory back."""
    state = load_state()
    instance = _instance(state["instance_id"])
    host, port = instance.get("ssh_host"), instance.get("ssh_port")
    if not host:
        print("instance has no ssh endpoint yet; try again once it is running")
        return 1
    print("copy the trained model back with:\n")
    print(f"  scp -P {port} -r root@{host}:/root/runs/dognet ./runs/")
    print(f"  scp -P {port} root@{host}:/root/dogsai_run.log ./runs/")
    print("\nthen locally:")
    print("  dogsai feeling runs/dognet/best.pt my_dog.mp4")
    return 0


def cmd_destroy(args) -> int:
    state = load_state()
    instance_id = state["instance_id"]
    if not args.yes:
        print(f"this will destroy instance {instance_id} and delete its disk.")
        print("copy anything you need first (fetch-run), then re-run with --yes.")
        return 0
    call(f"/instances/{instance_id}/", method="DELETE", payload={})
    elapsed = (time.time() - state["started"]) / 3600
    print(f"destroyed {instance_id}. ran {elapsed:.2f}h, "
          f"~${elapsed * state['price_per_hour']:.2f} of GPU time.")
    STATE_PATH.unlink(missing_ok=True)
    return 0


def cmd_whoami(args) -> int:
    me = call("/users/current/")
    print(f"user    : {me.get('username') or me.get('email')}")
    balance = me.get("credit")
    if balance is not None:
        print(f"credit  : ${balance:.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared(sub):
        sub.add_argument("--gpus", type=int, default=1)
        sub.add_argument("--min-vram", type=int, default=16, help="GB per GPU")
        sub.add_argument("--gpu-name", default=None, help="e.g. 'RTX 4090'")
        sub.add_argument("--disk", type=int, default=60, help="GB of instance disk")
        sub.add_argument("--interruptible", action="store_true",
                         help="bid on spare capacity, ~3x cheaper")
        sub.add_argument("--hours", type=float, default=3.0,
                         help="expected run length, for the cost estimate")

    offers = subparsers.add_parser("offers", help="list matching GPUs and prices")
    add_shared(offers)
    offers.set_defaults(func=cmd_offers)

    launch = subparsers.add_parser("launch", help="create an instance and start training")
    add_shared(launch)
    launch.add_argument("--dataset", default="dogbehaviour")
    launch.add_argument("--preset", default="small", choices=["nano", "small", "base"])
    launch.add_argument("--epochs", type=int, default=40)
    launch.add_argument("--batch", type=int, default=32)
    launch.add_argument("--loader-workers", type=int, default=8)
    launch.add_argument("--image", default="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
    launch.add_argument("--repo", default="https://github.com/beepbeep-dev/dogsai.git")
    launch.add_argument("--branch", default="claude/dog-behavior-detection-ai-xt072k")
    launch.add_argument("--yes", action="store_true", help="actually spend money")
    launch.set_defaults(func=cmd_launch)

    status = subparsers.add_parser("status", help="check the running instance")
    status.set_defaults(func=cmd_status)

    fetch = subparsers.add_parser("fetch-run", help="print commands to copy results back")
    fetch.set_defaults(func=cmd_fetch_run)

    destroy = subparsers.add_parser("destroy", help="destroy the instance and stop billing")
    destroy.add_argument("--yes", action="store_true")
    destroy.set_defaults(func=cmd_destroy)

    whoami = subparsers.add_parser("whoami", help="verify the key and show credit")
    whoami.set_defaults(func=cmd_whoami)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
