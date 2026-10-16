#!/usr/bin/env python3
"""Fixed-token OpenAI completions benchmark with incremental progress output."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import signal
import threading
import time

import requests


_thread_local = threading.local()
_stop = threading.Event()


def handle_stop(signum, frame):
    _stop.set()


def session():
    value = getattr(_thread_local, "session", None)
    if value is None:
        value = requests.Session()
        value.trust_env = False
        _thread_local.session = value
    return value


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct / 100.0
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def distribution(values):
    values = [float(item) for item in values if item is not None]
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None,
                "p50": None, "p90": None, "p95": None, "p99": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def atomic_json(path, value):
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def make_prompt(length, seed, request_index):
    rng = random.Random(seed + request_index * 1000003)
    prefix = [1000 + ((request_index >> shift) & 0xFFFF)
              for shift in (0, 8, 16, 24)]
    prompt = prefix[:length]
    while len(prompt) < length:
        prompt.append(rng.randint(1000, 220000))
    return prompt


def make_job(args, phase, round_index, request_index):
    global_index = round_index * 1000000 + request_index
    seed = args.seed + (100000000 if phase == "warmup" else 0)
    return {
        "request_index": request_index,
        "global_index": global_index,
        "body": {
            "model": args.model,
            "prompt": make_prompt(args.input_len, seed, global_index),
            "max_tokens": args.output_len,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
            "seed": 0,
            "cache_salt": "%s-%d-%d-%d" %
                          (phase, round_index, args.input_len, request_index),
        },
    }


def run_request(url, job, timeout):
    started = time.monotonic()
    first_token_at = None
    output_parts = []
    usage = {}
    event_count = 0
    error = None
    status_code = None
    done = False
    try:
        response = session().post(
            url, json=job["body"], stream=True, timeout=(30, timeout)
        )
        status_code = response.status_code
        if status_code != 200:
            error = "HTTP %s: %s" % (status_code, response.text[:2000])
        else:
            for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
                if _stop.is_set():
                    error = "terminated by capacity/timeout guard"
                    response.close()
                    break
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    break
                try:
                    event = json.loads(payload)
                except Exception:
                    continue
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if choices:
                    choice = choices[0]
                    if first_token_at is None and choice.get("finish_reason") is None:
                        first_token_at = time.monotonic()
                    piece = choice.get("text") or ""
                    if piece:
                        output_parts.append(piece)
                    event_count += 1
    except Exception as exc:
        error = "%s: %s" % (type(exc).__name__, exc)
    ended = time.monotonic()
    text = "".join(output_parts)
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    e2e = ended - started
    ttft = first_token_at - started if first_token_at is not None else None
    tpot_ms = None
    if ttft is not None and completion_tokens and completion_tokens > 1 and e2e > ttft:
        tpot_ms = 1000.0 * (e2e - ttft) / float(completion_tokens - 1)
    return {
        "request_index": job["request_index"],
        "status_code": status_code,
        "done": done,
        "error": error,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens"),
        "e2e_s": e2e,
        "ttft_s": ttft,
        "tpot_ms": tpot_ms,
        "stream_events": event_count,
        "output_chars": len(text),
        "replacement_chars": text.count("\ufffd"),
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "output_sample": text[:240],
    }


def run_round(args, phase, round_index, request_count, round_dir):
    os.makedirs(round_dir, exist_ok=True)
    progress_path = os.path.join(args.output_dir, "progress.json")
    request_path = os.path.join(round_dir, "requests.jsonl")
    jobs = [make_job(args, phase, round_index, index)
            for index in range(request_count)]
    url = args.base_url.rstrip("/") + "/v1/completions"
    started_mono = time.monotonic()
    started_epoch = time.time()
    results = []
    with open(request_path, "w", encoding="utf-8", buffering=1) as output:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency) as pool:
            futures = [pool.submit(run_request, url, job, args.timeout)
                       for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                output.write(json.dumps(result, ensure_ascii=False,
                                        sort_keys=True) + "\n")
                atomic_json(progress_path, {
                    "status": "RUNNING" if not _stop.is_set() else "TERMINATING",
                    "phase": phase,
                    "round": round_index + 1,
                    "completed": len(results),
                    "request_count": request_count,
                    "updated_at": time.time(),
                    "started_at": started_epoch,
                })
                if _stop.is_set():
                    for pending in futures:
                        pending.cancel()
                    break
    wall = time.monotonic() - started_mono
    results.sort(key=lambda item: item["request_index"])
    valid = [item for item in results if item["error"] is None and item["done"]]
    exact = [item for item in valid
             if item["prompt_tokens"] == args.input_len and
             item["completion_tokens"] == args.output_len]
    sum_prompt = sum(item.get("prompt_tokens") or 0 for item in valid)
    sum_completion = sum(item.get("completion_tokens") or 0 for item in valid)
    summary = {
        "status": "ABORTED" if _stop.is_set() else
                  ("PASS" if len(exact) == request_count else "FAIL"),
        "phase": phase,
        "round": round_index + 1,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "concurrency": args.concurrency,
        "requested": request_count,
        "successful": len(valid),
        "exact_length": len(exact),
        "wall_time_s": wall,
        "request_throughput_rps": len(valid) / wall if wall else None,
        "input_throughput_tps": sum_prompt / wall if wall else None,
        "output_throughput_tps": sum_completion / wall if wall else None,
        "total_throughput_tps": (sum_prompt + sum_completion) / wall if wall else None,
        "e2e_s": distribution([item["e2e_s"] for item in valid]),
        "ttft_s": distribution([item["ttft_s"] for item in valid]),
        "tpot_ms": distribution([item["tpot_ms"] for item in valid]),
        "replacement_chars": sum(item["replacement_chars"] for item in valid),
    }
    atomic_json(os.path.join(round_dir, "summary.json"), summary)
    return summary


def aggregate(args, measured):
    valid = [item for item in measured if item["status"] == "PASS"]
    result = {
        "status": "PASS" if len(valid) == args.measure_rounds else
                  ("ABORTED" if _stop.is_set() else "FAIL"),
        "base_url": args.base_url,
        "model": args.model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "concurrency": args.concurrency,
        "warmup_rounds": args.warmup_rounds,
        "measure_rounds": args.measure_rounds,
        "requests_per_round": args.requests_per_round,
        "successful_rounds": len(valid),
        "output_throughput_tps": distribution(
            [item["output_throughput_tps"] for item in valid]),
        "ttft_p50_s": distribution([item["ttft_s"]["p50"] for item in valid]),
        "ttft_p95_s": distribution([item["ttft_s"]["p95"] for item in valid]),
        "tpot_p50_ms": distribution([item["tpot_ms"]["p50"] for item in valid]),
        "tpot_p95_ms": distribution([item["tpot_ms"]["p95"] for item in valid]),
        "rounds": measured,
        "method": "fixed random token IDs; two warmup waves; three measured rounds",
    }
    atomic_json(os.path.join(args.output_dir, "summary.json"), result)
    atomic_json(os.path.join(args.output_dir, "progress.json"), {
        "status": result["status"], "updated_at": time.time()
    })
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--measure-rounds", type=int, default=3)
    parser.add_argument("--requests-per-round", type=int)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    args.requests_per_round = args.requests_per_round or max(8, 2 * args.concurrency)
    os.makedirs(args.output_dir, exist_ok=True)
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    atomic_json(os.path.join(args.output_dir, "config.json"), vars(args))

    for round_index in range(args.warmup_rounds):
        summary = run_round(
            args, "warmup", round_index, args.concurrency,
            os.path.join(args.output_dir, "warmup-%02d" % (round_index + 1)))
        if summary["status"] != "PASS":
            aggregate(args, [])
            return 2

    measured = []
    for round_index in range(args.measure_rounds):
        summary = run_round(
            args, "measure", round_index, args.requests_per_round,
            os.path.join(args.output_dir, "round-%02d" % (round_index + 1)))
        measured.append(summary)
        if summary["status"] != "PASS":
            break
    result = aggregate(args, measured)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
