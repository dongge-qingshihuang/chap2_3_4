#!/usr/bin/env python3
"""Run the B0-B4 Chapter 4 ablations sequentially across GPUs."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SCRIPT = (
    REPO_ROOT
    / "legged_gym"
    / "legged_gym"
    / "scripts"
    / "train_ablation.py"
)
DEFAULT_LOG_ROOT = REPO_ROOT / "legged_gym" / "logs" / "chapter4_ablation_queue"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes", default="B0,B1,B2,B3,B4"
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--experiment-name", default="chapter4_ablation"
    )
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    return parser.parse_args()


def write_manifest(path, jobs):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "jobs": jobs,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_status(manifest_path):
    if not manifest_path.is_file():
        print(f"No manifest: {manifest_path}")
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"updated_at: {payload.get('updated_at')}")
    for job in payload.get("jobs", []):
        print(
            f"{job['run_name']:<12} "
            f"GPU {str(job.get('gpu') or '-'): <2} "
            f"{job['status']:<10} "
            f"{job.get('started_at') or '-'} -> "
            f"{job.get('finished_at') or '-'} "
            f"rc={job.get('returncode')}"
        )


def main():
    args = parse_args()
    manifest_path = args.log_root / "manifest.json"
    if args.status:
        print_status(manifest_path)
        return

    modes = [value.strip().upper() for value in args.modes.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not modes or not seeds or not gpus:
        raise ValueError("modes, seeds, and gpus must be non-empty")

    args.log_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for mode in modes:
        for seed in seeds:
            run_name = f"{mode}_v2_s{seed}"
            log_path = args.log_root / f"{run_name}.log"
            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--num_envs",
                str(args.num_envs),
                "--seed",
                str(seed),
            ]
            if args.max_iterations is not None:
                command.extend(
                    ["--max_iterations", str(args.max_iterations)]
                )
            jobs.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "run_name": run_name,
                    "command": command,
                    "environment": {
                        "GO2_ARM_EXPERIMENT": mode,
                        "CHAPTER4_EXPERIMENT_NAME": args.experiment_name,
                        "CHAPTER4_RUN_NAME": run_name,
                        "CHAPTER4_LOG_ROOT": str(
                            args.log_root / "runs"
                        ),
                    },
                    "log_path": str(log_path),
                    "status": "pending",
                    "gpu": None,
                    "pid": None,
                    "started_at": None,
                    "finished_at": None,
                    "returncode": None,
                }
            )

    if args.dry_run:
        for job in jobs:
            print(
                f"{job['run_name']}: "
                f"GO2_ARM_EXPERIMENT={job['mode']} "
                + " ".join(job["command"])
            )
        return

    write_manifest(manifest_path, jobs)
    processes = {}
    pending = list(range(len(jobs)))
    available_gpus = list(gpus)
    completed = 0
    while pending or processes:
        while pending and available_gpus:
            job_index = pending.pop(0)
            gpu = available_gpus.pop(0)
            job = jobs[job_index]
            environment = os.environ.copy()
            environment.update(job["environment"])
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            job_command = list(job["command"])
            # CUDA_VISIBLE_DEVICES remaps the selected physical GPU to cuda:0
            # inside the child process.
            job_command[2:2] = ["--rl_device", "cuda:0"]
            log_file = open(job["log_path"], "a", encoding="utf-8")
            log_file.write(
                f"\n[{datetime.now().isoformat(timespec='seconds')}] "
                + " ".join(job_command)
                + "\n"
            )
            log_file.flush()
            process = subprocess.Popen(
                job_command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes[job_index] = (process, log_file, gpu)
            job.update(
                status="running",
                gpu=gpu,
                pid=process.pid,
                started_at=datetime.now().isoformat(timespec="seconds"),
            )
            write_manifest(manifest_path, jobs)
            print(f"STARTED {job['run_name']} on GPU {gpu} pid={process.pid}", flush=True)

        time.sleep(5)
        for job_index, (process, log_file, gpu) in list(processes.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            log_file.close()
            job = jobs[job_index]
            job.update(
                status="finished" if returncode == 0 else "failed",
                finished_at=datetime.now().isoformat(timespec="seconds"),
                returncode=returncode,
            )
            processes.pop(job_index)
            available_gpus.append(gpu)
            completed += 1
            write_manifest(manifest_path, jobs)
            print(
                f"{job['status'].upper()} {job['run_name']} rc={returncode} "
                f"({completed}/{len(jobs)})",
                flush=True,
            )

    write_manifest(manifest_path, jobs)
    print(f"QUEUE FINISHED: {manifest_path}")
    if any(job["status"] != "finished" for job in jobs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
