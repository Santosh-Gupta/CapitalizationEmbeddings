#!/usr/bin/env python
"""Poll RunPod for an acceptable GPU, create a pod, and optionally start work.

The polling policy is intentionally conservative:

* tier 1 is checked every 2 minutes immediately;
* after 20 minutes, tiers 2-5 are added at progressively slower intervals;
* expensive datacenter GPUs are not included in the default tier list;
* a max hourly price is enforced before a create request is sent.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REST_URL = "https://rest.runpod.io/v1"
GRAPHQL_URL = "https://api.runpod.io/graphql"
USER_AGENT = "CapitalizationEmbeddings/0.1 (+https://github.com/Santosh-Gupta/CapitalizationEmbeddings)"

DEFAULT_TIERS = (
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX A5000",
)

BLACKWELL_OR_5090 = (
    "5090",
    "Blackwell",
)


@dataclass(frozen=True)
class Tier:
    index: int
    gpu_type_id: str
    interval_seconds: int
    start_after_seconds: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY", ""))
    parser.add_argument(
        "--network-volume-id",
        default=os.environ.get("RUNPOD_NETWORK_VOLUME_ID", ""),
    )
    parser.add_argument("--name", default="capitalization-ablation")
    parser.add_argument(
        "--gpu-tiers",
        nargs="+",
        default=list(DEFAULT_TIERS),
        help="GPU type IDs in preference order. Defaults exclude A100/H100/H200/B200.",
    )
    parser.add_argument("--max-price", type=float, default=1.05)
    parser.add_argument("--cloud-type", choices=["COMMUNITY", "SECURE"], default="COMMUNITY")
    parser.add_argument("--data-center-id", default="US-IL-1")
    parser.add_argument(
        "--image-name",
        default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    )
    parser.add_argument("--container-disk-gb", type=int, default=40)
    parser.add_argument("--volume-gb", type=int, default=20)
    parser.add_argument("--min-vcpu-per-gpu", type=int, default=6)
    parser.add_argument("--min-ram-per-gpu", type=int, default=30)
    parser.add_argument("--initial-tier1-only-minutes", type=int, default=20)
    parser.add_argument("--tier1-interval-seconds", type=int, default=120)
    parser.add_argument("--tier2-interval-seconds", type=int, default=240)
    parser.add_argument("--tier3-interval-seconds", type=int, default=480)
    parser.add_argument("--tier4-interval-seconds", type=int, default=960)
    parser.add_argument("--tier5-interval-seconds", type=int, default=1920)
    parser.add_argument("--max-runtime-minutes", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--auto-run-ablation", action="store_true")
    parser.add_argument("--ssh-key", default=str(Path.home() / ".ssh" / "id_ed25519"))
    parser.add_argument("--ssh-timeout-seconds", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key and not args.dry_run:
        raise SystemExit("RUNPOD_API_KEY is required.")
    if not args.network_volume_id and not args.dry_run:
        raise SystemExit("RUNPOD_NETWORK_VOLUME_ID is required.")

    tiers = make_tiers(args)
    last_attempt: dict[int, float] = {tier.index: -1e12 for tier in tiers}
    start = time.monotonic()
    print("RunPod GPU poller started.", flush=True)
    print(
        json.dumps(
            {
                "tiers": [tier.__dict__ for tier in tiers],
                "max_price": args.max_price,
                "cloud_type": args.cloud_type,
                "data_center_id": args.data_center_id,
                "network_volume_id": args.network_volume_id,
                "dry_run": args.dry_run,
            },
            indent=2,
        ),
        flush=True,
    )

    while True:
        elapsed = time.monotonic() - start
        if args.max_runtime_minutes and elapsed > args.max_runtime_minutes * 60:
            raise SystemExit("Max runtime reached without creating a pod.")

        any_due = False
        for tier in tiers:
            if elapsed < tier.start_after_seconds:
                continue
            if elapsed - last_attempt[tier.index] < tier.interval_seconds:
                continue
            any_due = True
            last_attempt[tier.index] = elapsed
            pod = try_tier(args, tier)
            if pod is not None:
                pod_id = pod.get("id", "")
                print(f"CREATED pod_id={pod_id} gpu={tier.gpu_type_id}", flush=True)
                if args.auto_run_ablation:
                    wait_for_ssh_and_start(args, pod_id)
                return

        if args.once:
            return

        if not any_due:
            time.sleep(5)
        else:
            time.sleep(10)


def make_tiers(args: argparse.Namespace) -> list[Tier]:
    intervals = [
        args.tier1_interval_seconds,
        args.tier2_interval_seconds,
        args.tier3_interval_seconds,
        args.tier4_interval_seconds,
        args.tier5_interval_seconds,
    ]
    delay = args.initial_tier1_only_minutes * 60
    tiers = []
    for index, gpu_type_id in enumerate(args.gpu_tiers[:5], start=1):
        tiers.append(
            Tier(
                index=index,
                gpu_type_id=gpu_type_id,
                interval_seconds=intervals[index - 1],
                start_after_seconds=0 if index == 1 else delay,
            )
        )
    return tiers


def try_tier(args: argparse.Namespace, tier: Tier) -> dict[str, Any] | None:
    print(
        f"CHECK tier={tier.index} gpu={tier.gpu_type_id} "
        f"time={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        flush=True,
    )
    info = gpu_info(args, tier.gpu_type_id)
    if info is None:
        print(f"  unavailable metadata for {tier.gpu_type_id}", flush=True)
        return None

    price = info.get("price")
    stock = info.get("dataCenterStockStatus")
    print(
        "  global_stock={} datacenter_stock={} price={}".format(
            info.get("stockStatus"),
            info.get("dataCenterStockStatus"),
            price,
        ),
        flush=True,
    )
    if stock in (None, "None"):
        return None
    if price is None or float(price) > args.max_price:
        return None

    if args.dry_run:
        print("  DRY RUN: would create pod.", flush=True)
        return None

    return create_pod(args, tier.gpu_type_id)


def gpu_info(args: argparse.Namespace, gpu_type_id: str) -> dict[str, Any] | None:
    secure = "true" if args.cloud_type == "SECURE" else "false"
    query = """
    query GPUInfo($id: String!) {
      gpuTypes(input: { id: $id }) {
        id
        displayName
        memoryInGb
        lowestPrice(input: { gpuCount: 1, secureCloud: SECURE_CLOUD }) {
          stockStatus
          uninterruptablePrice
          availableGpuCounts
        }
      }
      dataCenters {
        id
        gpuAvailability {
          id
          stockStatus
        }
      }
    }
    """.replace("SECURE_CLOUD", secure)
    payload = graphql(args.api_key, query, {"id": gpu_type_id})
    gpu_types = payload.get("data", {}).get("gpuTypes", [])
    if not gpu_types:
        return None
    gpu = gpu_types[0]
    lowest = gpu.get("lowestPrice") or {}
    datacenter_stock = None
    for data_center in payload.get("data", {}).get("dataCenters", []):
        if data_center.get("id") != args.data_center_id:
            continue
        for availability in data_center.get("gpuAvailability", []):
            if availability.get("id") == gpu_type_id:
                datacenter_stock = availability.get("stockStatus")
                break
    return {
        "id": gpu.get("id"),
        "displayName": gpu.get("displayName"),
        "memoryInGb": gpu.get("memoryInGb"),
        "stockStatus": lowest.get("stockStatus"),
        "dataCenterStockStatus": datacenter_stock,
        "price": lowest.get("uninterruptablePrice"),
        "availableGpuCounts": lowest.get("availableGpuCounts"),
    }


def graphql(api_key: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{GRAPHQL_URL}?api_key={api_key}",
        data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
        headers={"content-type": "application/json", "user-agent": USER_AGENT},
        method="POST",
    )
    return read_json(request)


def create_pod(args: argparse.Namespace, gpu_type_id: str) -> dict[str, Any] | None:
    payload: dict[str, Any] = {
        "name": args.name,
        "cloudType": args.cloud_type,
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": [gpu_type_id],
        "gpuTypePriority": "custom",
        "dataCenterIds": [args.data_center_id],
        "dataCenterPriority": "custom",
        "imageName": args.image_name,
        "containerDiskInGb": args.container_disk_gb,
        "volumeInGb": args.volume_gb,
        "networkVolumeId": args.network_volume_id,
        "volumeMountPath": "/workspace",
        "ports": ["8888/http", "22/tcp"],
        "supportPublicIp": True,
        "minVCPUPerGPU": args.min_vcpu_per_gpu,
        "minRAMPerGPU": args.min_ram_per_gpu,
        "interruptible": False,
        "locked": False,
    }
    if any(marker in gpu_type_id for marker in BLACKWELL_OR_5090):
        payload["allowedCudaVersions"] = ["12.8", "12.9", "13.0"]

    request = urllib.request.Request(
        f"{REST_URL}/pods",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    pod = read_json(request)
    if not pod.get("id"):
        return None
    return pod


def get_pod(args: argparse.Namespace, pod_id: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{REST_URL}/pods/{pod_id}",
        headers={"Authorization": f"Bearer {args.api_key}", "User-Agent": USER_AGENT},
        method="GET",
    )
    return read_json(request)


def read_json(request: urllib.request.Request) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        print(f"HTTP {error.code}: {body}", flush=True)
        return {}
    except urllib.error.URLError as error:
        print(f"URL error: {error}", flush=True)
        return {}


def wait_for_ssh_and_start(args: argparse.Namespace, pod_id: str) -> None:
    deadline = time.monotonic() + args.ssh_timeout_seconds
    while time.monotonic() < deadline:
        pod = get_pod(args, pod_id)
        ip = pod.get("publicIp")
        mappings = pod.get("portMappings") or {}
        port = mappings.get("22") or mappings.get(22)
        if ip and port and ssh_ready(args, ip, int(port)):
            print(f"SSH ready: root@{ip} -p {port}", flush=True)
            launch_ablation(args, ip, int(port))
            return
        print("Waiting for SSH...", flush=True)
        time.sleep(15)
    raise SystemExit(f"Timed out waiting for SSH on pod {pod_id}.")


def ssh_ready(args: argparse.Namespace, ip: str, port: int) -> bool:
    command = ssh_base(args, ip, port) + ["true"]
    return subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def launch_ablation(args: argparse.Namespace, ip: str, port: int) -> None:
    remote = r"""
set -euo pipefail
mkdir -p /workspace/repos /workspace/capitalization_embeddings/logs
if [ -d /workspace/repos/CapitalizationEmbeddings/.git ]; then
  cd /workspace/repos/CapitalizationEmbeddings
  git pull --ff-only
else
  git clone https://github.com/Santosh-Gupta/CapitalizationEmbeddings.git /workspace/repos/CapitalizationEmbeddings
  cd /workspace/repos/CapitalizationEmbeddings
fi
python -m pip install --upgrade pip
python -m pip install -r requirements-colab.txt
python -m pip install -e .
bash -n scripts/run_case_ablation_batch.sh
LOG="/workspace/capitalization_embeddings/logs/auto_case_ablation_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash scripts/run_case_ablation_batch.sh > "$LOG" 2>&1 &
echo "started_pid=$!"
echo "log=$LOG"
"""
    command = ssh_base(args, ip, port) + ["bash -lc " + sh_quote(remote)]
    subprocess.run(command, check=True)


def ssh_base(args: argparse.Namespace, ip: str, port: int) -> list[str]:
    return [
        "ssh",
        "-i",
        args.ssh_key,
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ConnectTimeout=10",
        f"root@{ip}",
    ]


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()
