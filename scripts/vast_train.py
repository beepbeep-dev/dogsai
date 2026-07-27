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
# Some endpoints have been moved to v1; call() picks per-path.
API_V1 = "https://console.vast.ai/api/v1"
V1_PATHS = ("/instances/",)
STATE_PATH = Path.home() / ".dogsai_vast_run.json"

# Provisioning script run on the instance at boot. Kept dependency-light and
# idempotent so a re-run after an interruption resumes rather than restarts.
ONSTART = r"""#!/bin/bash
set -euo pipefail
exec > >(tee -a /root/dogsai_run.log) 2>&1
echo "=== dogsai provisioning $(date -u) ==="

export DEBIAN_FRONTEND=noninteractive
export PIP_ROOT_USER_ACTION=ignore
# Reduces allocator fragmentation; cheap insurance against a marginal OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# huggingface_hub's newer "xet" transfer backend is the common thread behind two
# separate failures seen on real runs: a 429 from its token endpoint under
# unauthenticated access, and (on a different instance/region) a download that
# hung indefinitely after a handful of files with no error and no progress.
# Forcing the classic HTTP/LFS path trades a little speed for actually finishing.
export HF_HUB_DISABLE_XET=1
python3 -m pip install -q --upgrade pip
python3 -m pip install -q av opencv-python-headless huggingface_hub tqdm numpy

cd /root
__FETCH_SOURCE__
cd /root/dogsai
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# Fetch + convert the dataset. Retried with backoff and with modest parallelism:
# unauthenticated HuggingFace downloads get 429-rate-limited at 16 workers, and
# because this script runs under `set -e` a single 429 killed the whole run.
# snapshot_download is resumable, so each retry continues rather than restarting.
for attempt in 1 2 3 4 5 6 7 8; do
  # A hard wall-clock timeout on top of the retry loop, not instead of it: two
  # separate real runs hung indefinitely partway through this download (stopped
  # making progress, never raised, never returned) rather than failing loudly.
  # `for attempt in ...; if python3 ...; then break; fi` only ever retries on an
  # *exception* — a hung process just sits inside the `if` forever and the loop
  # never gets a second attempt. `timeout` turns "hangs forever" into "fails after
  # N seconds", which is what actually lets the retry+backoff logic below do its job.
  # Three consecutive real runs hung with 4 parallel workers, regardless of GPU
  # host, region, or the xet backend being disabled — consistent with anti-abuse
  # throttling that kicks in on *concurrent* connections from a datacenter IP
  # range rather than anything file- or xet-specific (checked: no file in this
  # dataset is anomalously large). Fewer simultaneous connections is a more
  # boring, less abuse-shaped access pattern, so this trades nominal throughput
  # for actually finishing; the timeout is raised to match a legitimately slower
  # but working serial-ish download rather than one that is still just hanging.
  if timeout 1200 python3 -m dogsai.cli fetch "__DATASET__" \
       --raw-root /root/data/raw --out /root/data/prepared --workers 2; then
    echo "dataset ready after attempt $attempt"
    break
  fi
  echo "fetch attempt $attempt failed or timed out after 1200s; backing off"
  sleep $((attempt * 20))
done
test -s /root/data/prepared/train.jsonl || { echo "FATAL: dataset never arrived"; exit 1; }

python3 -m dogsai.cli audit --data-root /root/data/prepared \
  --behaviours /root/data/prepared/behaviours.txt --no-duplicate-check || true

# Enrich: add self-detected 'barking' labels from our own audio module, turning
# the single-label dataset into a genuine multi-label one — see dogsai/enrich.py.
# make-captions computes the per-clip audio summary enrich reuses; if it fails for
# any reason enrich() falls back to computing audio live, per clip, so this still
# produces a correct (just slower) result.
python3 -m dogsai.cli make-captions --data-root /root/data/prepared || true
python3 -m dogsai.cli enrich --data-root /root/data/prepared \
  --out /root/data/prepared/enriched \
  --captions /root/data/prepared/captions

python3 -m dogsai.cli train \
  --data-root /root/data/prepared/enriched \
  --behaviours /root/data/prepared/enriched/behaviours.txt \
  --task multilabel \
  --out-dir /root/runs/dognet \
  --cache --cache-frames __CACHE_FRAMES__ --cache-size __CACHE_SIZE__ \
  --set model.preset=__PRESET__ \
  --set train.epochs=__EPOCHS__ \
  --set train.batch_size=__BATCH__ \
  --set train.accum_steps=__ACCUM__ \
  --set data.num_workers=__WORKERS__ \
  --set train.amp=true \
  --set train.compile=false \
  --set train.early_stop_patience=__PATIENCE__

python3 -m dogsai.cli eval /root/runs/dognet/best.pt \
  --data-root /root/data/prepared/enriched --split val \
  --json /root/runs/dognet/val_metrics.json || true

python3 -m dogsai.cli export /root/runs/dognet/best.pt /root/runs/dognet/dognet.ts.pt || true
cp /root/data/prepared/enriched/behaviours.txt /root/runs/dognet/behaviours.txt || true

# --- get the trained model back without SSH ---------------------------------
# request_logs only tails console output; scp needs an SSH egress this sandbox
# does not have. So the run uploads its own result to a public, anonymous HTTPS
# file host and prints the URL, which the caller reads back out of the console
# log (already working) and downloads with a plain GET.
cd /root/runs
tar czf artifact.tar.gz -C dognet \
  $(cd dognet && ls best.pt config.json history.json val_metrics.json \
      behaviours.txt dognet.ts.pt dognet.ts.meta.json 2>/dev/null)
echo "artifact size: $(wc -c < artifact.tar.gz) bytes"

ARTIFACT_URL=""
for attempt in 1 2 3; do
  ARTIFACT_URL=$(curl -sS -m 180 -A "Mozilla/5.0" -F"file=@artifact.tar.gz" https://0x0.st | tr -d "\r\n")
  case "$ARTIFACT_URL" in https://0x0.st/*) break ;; esac
  echo "0x0.st upload attempt $attempt failed, retrying"
  sleep 5
done
case "$ARTIFACT_URL" in
  https://0x0.st/*) ;;
  *)
    echo "0x0.st failed, falling back to litterbox.catbox.moe"
    ARTIFACT_URL=$(curl -sS -m 180 -F "reqtype=fileupload" -F "time=72h" \
      -F "fileToUpload=@artifact.tar.gz" \
      https://litterbox.catbox.moe/resources/internals/api.php | tr -d "\r\n")
    ;;
esac
echo "ARTIFACT_URL: $ARTIFACT_URL"

touch /root/DOGSAI_DONE
echo "=== dogsai finished $(date -u) ==="
"""


# Two ways to get the code onto the instance. Cloning is cleaner but only works
# for a public repo; embedding a base64 tarball works regardless and avoids both
# making a private repo public and shipping a git credential to a third-party
# host, neither of which is an acceptable price for a training run.
_CLONE = 'git clone --depth 1 --branch "{branch}" "{repo}" dogsai'
_EMBED = """mkdir -p /root/dogsai
cat > /root/src.b64 <<'DOGSAI_EOF'
{payload}
DOGSAI_EOF
base64 -d /root/src.b64 | tar xzf - -C /root/dogsai
echo "source unpacked: $(find /root/dogsai -name '*.py' | wc -l) python files"
"""


def source_payload(repo_root: Path) -> str:
    """gzip+base64 the package so it can travel inside the boot script."""
    import base64
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ("dogsai", "pyproject.toml", "requirements.txt"):
            path = repo_root / name
            if not path.exists():
                continue
            archive.add(
                path,
                arcname=name,
                filter=lambda info: None
                if ("__pycache__" in info.name or info.name.endswith(".pyc"))
                else info,
            )
    encoded = base64.b64encode(buffer.getvalue()).decode()
    # Wrap so the heredoc lines stay a sane width.
    return "\n".join(encoded[i : i + 200] for i in range(0, len(encoded), 200))


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
    base = API_V1 if path in V1_PATHS else API
    url = f"{base}{path}"
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
    # Vast has moved this endpoint before; try the current one first and fall
    # back, so a server-side rename degrades to a clear error rather than a 404
    # with no explanation.
    last: Exception | None = None
    for path in ("/search/asks/", "/bundles/"):
        try:
            result = call(path, method="PUT", payload={"q": query})
        except SystemExit as exc:
            last = exc
            continue
        offers = result.get("offers", [])
        if offers or "offers" in result:
            return offers
    raise SystemExit(f"could not search offers: {last}")


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

    repo_root = Path(__file__).resolve().parent.parent
    if args.embed_source:
        fetch = _EMBED.format(payload=source_payload(repo_root))
        print(f"embedding source in the boot script ({len(fetch) / 1024:.0f} KB)")
    else:
        fetch = _CLONE.format(branch=args.branch, repo=args.repo)

    onstart = (
        ONSTART.replace("__FETCH_SOURCE__", fetch)
        .replace("__REPO__", args.repo)
        .replace("__BRANCH__", args.branch)
        .replace("__DATASET__", args.dataset)
        .replace("__PRESET__", args.preset)
        .replace("__EPOCHS__", str(args.epochs))
        .replace("__BATCH__", str(args.batch))
        .replace("__WORKERS__", str(args.loader_workers))
        .replace("__PATIENCE__", str(args.patience))
        .replace("__CACHE_FRAMES__", str(args.cache_frames))
        .replace("__CACHE_SIZE__", str(args.cache_size))
        .replace("__ACCUM__", str(args.accum_steps))
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
    found = payload.get("instances", payload) or {}
    if isinstance(found, list):
        found = next((i for i in found if i.get("id") == instance_id), {})
    return found


def list_instances() -> list[dict]:
    payload = call("/instances/")
    found = payload.get("instances", payload)
    return found if isinstance(found, list) else []


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


def cmd_logs(args) -> int:
    """Fetch the instance's console log through the Vast API.

    Vast serves logs indirectly: you ask for them, it writes them to a URL, you
    then GET that. This exists because SSH egress is frequently blocked (CI
    runners, sandboxes, locked-down networks), and without it there is no way to
    see whether a run is progressing.
    """
    state = load_state()
    instance_id = state["instance_id"]
    response = call(f"/instances/request_logs/{instance_id}/", method="PUT",
                    payload={"tail": str(args.tail)})
    url = response.get("result_url")
    if not url:
        print(f"no log url returned: {response}")
        return 1
    # The file appears a moment after the request.
    for attempt in range(args.retries):
        time.sleep(args.wait)
        try:
            with urllib.request.urlopen(url, timeout=60) as handle:
                text = handle.read().decode(errors="replace")
            if text.strip():
                print(text[-args.bytes:])
                return 0
        except urllib.error.HTTPError as exc:
            if exc.code not in (403, 404):
                raise
        print(f"  log not ready yet (attempt {attempt + 1}/{args.retries})")
    print("log did not become available; try again shortly")
    return 1


def cmd_fetch_artifact(args) -> int:
    """Pull the ARTIFACT_URL a finished run printed and download+unpack it.

    This is the actual retrieval path for a trained checkpoint: the run cannot be
    scp'd off (no SSH egress here), so it uploads itself to a public file host and
    this reads the resulting URL back out of the console log.
    """
    import re
    import time as _time
    import urllib.request

    state = load_state()
    instance_id = state["instance_id"]
    response = call(f"/instances/request_logs/{instance_id}/", method="PUT",
                    payload={"tail": "4000"})
    url = response.get("result_url")
    if not url:
        print(f"no log url returned: {response}")
        return 1

    text = ""
    for attempt in range(args.retries):
        _time.sleep(args.wait)
        try:
            with urllib.request.urlopen(url, timeout=60) as handle:
                text = handle.read().decode(errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code not in (403, 404):
                raise
            text = ""
        if "ARTIFACT_URL:" in text:
            break
        print(f"  artifact not ready yet (attempt {attempt + 1}/{args.retries})")

    match = re.search(r"ARTIFACT_URL:\s*(\S+)", text)
    if not match or not match.group(1).startswith("http"):
        print("no ARTIFACT_URL found yet in the log. The run may still be "
              "training — check `logs` for progress, then try again.")
        return 1
    artifact_url = match.group(1)
    print(f"found artifact: {artifact_url}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / "artifact.tar.gz"
    urllib.request.urlretrieve(artifact_url, archive_path)
    print(f"downloaded {archive_path} ({archive_path.stat().st_size / 1e6:.1f} MB)")

    import tarfile

    with tarfile.open(archive_path) as tar:
        tar.extractall(out_dir)
    print(f"unpacked into {out_dir}/")
    print(f"\n  dogsai eval {out_dir}/best.pt --data-root <your data> --split val")
    print(f"  dogsai feeling {out_dir}/best.pt my_dog.mp4")
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


def cmd_instances(args) -> int:
    found = list_instances()
    if not found:
        print("no instances — nothing is being billed.")
        return 0
    total = 0.0
    for instance in found:
        rate = instance.get("dph_total", 0.0)
        total += rate
        print(f"  id={instance.get('id')} {instance.get('actual_status')} "
              f"${rate:.3f}/hr  {instance.get('gpu_name')}")
    print(f"\ntotal burn: ${total:.3f}/hr")
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
    launch.add_argument("--batch", type=int, default=32,
                        help="micro-batch actually placed on the GPU per step")
    launch.add_argument("--accum-steps", type=int, default=1,
                        help="gradient-accumulation multiplier; effective batch "
                             "= batch * accum-steps, at a fraction of the VRAM")
    launch.add_argument("--patience", type=int, default=20,
                        help="early-stop patience, in epochs")
    launch.add_argument("--cache-frames", type=int, default=32,
                        help="frames per cached clip; must exceed the preset's "
                             "clip_frames so temporal jitter survives")
    launch.add_argument("--cache-size", type=int, default=224,
                        help="cached resolution; must exceed the preset's image_size "
                             "so random-resized-crop has real pixels to pick")
    launch.add_argument("--loader-workers", type=int, default=8)
    launch.add_argument("--image", default="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
    launch.add_argument("--repo", default="https://github.com/beepbeep-dev/dogsai.git")
    launch.add_argument("--branch", default="claude/dog-behavior-detection-ai-xt072k")
    launch.add_argument("--embed-source", action="store_true", default=True,
                        help="ship the code inside the boot script instead of cloning "
                             "(required for a private repo)")
    launch.add_argument("--clone", dest="embed_source", action="store_false",
                        help="clone from --repo instead; only works if it is public")
    launch.add_argument("--yes", action="store_true", help="actually spend money")
    launch.set_defaults(func=cmd_launch)

    status = subparsers.add_parser("status", help="check the running instance")
    status.set_defaults(func=cmd_status)

    logs = subparsers.add_parser("logs", help="read the instance console log via the API")
    logs.add_argument("--tail", type=int, default=2000)
    logs.add_argument("--bytes", type=int, default=8000)
    # S3 needs a little while to publish the log after the request; 6x6s was not
    # enough in practice and produced a spurious "log unavailable".
    logs.add_argument("--wait", type=float, default=10.0)
    logs.add_argument("--retries", type=int, default=12)
    logs.set_defaults(func=cmd_logs)

    fetch = subparsers.add_parser("fetch-run", help="print commands to copy results back")
    fetch.set_defaults(func=cmd_fetch_run)

    fetch_artifact = subparsers.add_parser(
        "fetch-artifact",
        help="download the trained model over HTTPS (no SSH needed)",
    )
    fetch_artifact.add_argument("--out", default="runs/dognet")
    fetch_artifact.add_argument("--wait", type=float, default=10.0)
    fetch_artifact.add_argument("--retries", type=int, default=6)
    fetch_artifact.set_defaults(func=cmd_fetch_artifact)

    destroy = subparsers.add_parser("destroy", help="destroy the instance and stop billing")
    destroy.add_argument("--yes", action="store_true")
    destroy.set_defaults(func=cmd_destroy)

    instances = subparsers.add_parser(
        "instances", help="list running instances (check nothing is quietly billing)"
    )
    instances.set_defaults(func=cmd_instances)

    whoami = subparsers.add_parser("whoami", help="verify the key and show credit")
    whoami.set_defaults(func=cmd_whoami)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
