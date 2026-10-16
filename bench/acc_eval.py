"""四项精度评测（赛题门槛：GSM8K / LiveCodeBench v6 / RealWorldQA / MathVision），走服务的 /v1/chat/completions。
用于「优化后 vs 厂商默认」同口径配对对比（规则：降幅 ≤2pp）；思考模式开（Qwen3.8 默认），采样按官方推荐、每题固定种子。
用法：python3 acc_eval.py <任务> <输出 jsonl> [--n N] [--c 并发] [--max-tokens M]
  任务：gsm8k | rwqa | mathvision | lcb ；中断后重跑会跳过已完成的题（按 id 续跑）。
  汇总：python3 acc_eval.py sum <jsonl>...
数据：/workspace/q38/datasets/{gsm8k_test.jsonl, eval/...}（hf-mirror 下载，见优化清单 §9）。"""
import argparse, base64, concurrent.futures as cf, io, json, os, pickle, re, subprocess, sys, tempfile, threading, time, zlib

import requests

D = os.environ.get("Q38_DATASETS", "/workspace/q38/datasets")
URL = os.environ.get("URL", "http://127.0.0.1:30000/v1/chat/completions")
MODEL = os.environ.get("SERVED", "qwen38")
SAMP = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}       # Qwen3 思考模式推荐采样
MAXTOK = {"gsm8k": 8192, "rwqa": 8192, "mathvision": 32768, "lcb": 32768}   # mathvision 16384 冒烟 2/5 截断（10-11），放到 32768


# ---------------------------------------------------------------- 数据
def _parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _img_url(b):
    if isinstance(b, dict):
        b = b.get("bytes")
    mime = "image/png" if b[:4] == b"\x89PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(b).decode()


def load(task):
    items = []
    if task == "gsm8k":
        for i, l in enumerate(open(f"{D}/gsm8k_test.jsonl")):
            r = json.loads(l)
            items.append({"id": i, "msgs": [{"role": "user", "content": r["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                          "gold": r["answer"].split("####")[-1].strip().replace(",", "")})
    elif task == "rwqa":
        rows = []
        for f in sorted(os.listdir(f"{D}/eval/xai-org_RealworldQA")):
            rows += _parquet(f"{D}/eval/xai-org_RealworldQA/{f}")
        for i, r in enumerate(rows):
            items.append({"id": i, "msgs": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _img_url(r["image"])}},
                                                                          {"type": "text", "text": r["question"]}]}],
                          "gold": str(r["answer"]).strip()})
    elif task == "mathvision":
        d = f"{D}/eval/MathLLMs_MathVision"
        f = [x for x in os.listdir(d) if x.startswith(os.environ.get("MV_SPLIT", "testmini"))][0]
        for r in _parquet(f"{d}/{f}"):
            q = r["question"]
            opts = list(r.get("options") or [])
            if opts:
                q += "\n\nOptions:\n" + "\n".join(f"{chr(65 + j)}. {o}" for j, o in enumerate(opts))
                q += "\nPlease reason step by step, and put the letter of the correct option within \\boxed{}."
            else:
                q += "\nPlease reason step by step, and put your final answer within \\boxed{}."
            img = r.get("decoded_image") or r.get("image")
            content = ([{"type": "image_url", "image_url": {"url": _img_url(img)}}] if isinstance(img, (bytes, dict)) else []) + [{"type": "text", "text": q}]
            items.append({"id": str(r["id"]), "msgs": [{"role": "user", "content": content}], "gold": str(r["answer"]).strip(), "mc": bool(opts)})
    elif task == "lcb":
        for l in open(f"{D}/eval/livecodebench_code_generation_lite/test6.jsonl"):
            r = json.loads(l)
            if r.get("starter_code"):
                fmt = ("### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
                       f"```python\n{r['starter_code']}\n```\n\n")
            else:
                fmt = ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). "
                       "Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
                       "```python\n# YOUR CODE HERE\n```\n\n")
            sysmsg = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."
            user = f"### Question:\n{r['question_content']}\n\n{fmt}### Answer: (use the provided format with backticks)\n\n"
            items.append({"id": r["question_id"], "msgs": [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
                          "lcb": {k: r[k] for k in ("public_test_cases", "private_test_cases", "metadata", "starter_code")}})
    return items


# ---------------------------------------------------------------- 判分
def _boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    dep, k = 0, j
    while k < len(s):
        dep += (s[k] == "{") - (s[k] == "}")
        if dep == 0:
            return s[j + 1:k]
        k += 1
    return None


def _mv(pred, gold):
    from math_verify import parse, verify
    try:
        return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
    except Exception:
        return False


def _num(s):
    # 10-11 冒烟发现：模型写 \boxed{\$70{,}000}（LaTeX 千分位 {,}），只删逗号会剩 "70{}000" → 取到 000 判错
    for a in ("{,}", "\\,", "\\!", ",", "$", "{}"):
        s = s.replace(a, "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return m[-1] if m else None


def score(task, it, text):
    if task == "gsm8k":
        p = _boxed(text) or _num(text) or ""
        n = _num(p)
        return n is not None and abs(float(n) - float(it["gold"])) < 1e-6
    if task == "rwqa":
        g = it["gold"].strip().rstrip(".").lower()
        t = re.sub(r"\\boxed\{([^}]*)\}", r"\1", text).strip().rstrip(".").lower()
        if len(g) == 1 and g.isalpha():
            m = re.match(r"^\(?([a-z])\)?(?:[.):\s]|$)", t)
            return bool(m) and m.group(1) == g
        return t == g or t.split("\n")[-1].strip().rstrip(".") == g
    if task == "mathvision":
        p = (_boxed(text) or "").strip()
        if it["mc"]:
            m = re.search(r"\b([A-E])\b", p) or re.search(r"\b([A-E])\b", text[-200:])
            return bool(m) and m.group(1) == it["gold"].upper()
        # 10-11：(3, -4) 与金标 (3,-4) 只差空格被 math_verify 判错 → 先去空白比较
        return bool(p) and (re.sub(r"\s+", "", p) == re.sub(r"\s+", "", it["gold"]) or _mv(p, it["gold"]))
    if task == "lcb":
        return lcb_run(it, text)


# ---------------------------------------------------------------- LCB 执行（stdin / 函数两型；输出逐行逐词比较，数值容差 1e-6）
_FUNC_TMPL = """
import json, sys
from typing import *
_lines = [x for x in sys.stdin.read().split("\\n") if x.strip()]
_args = [json.loads(x) for x in _lines]
print(json.dumps(Solution().{fn}(*_args)))
"""


def _tests(lc):
    pub = json.loads(lc["public_test_cases"])
    try:
        prv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(lc["private_test_cases"].encode()))))
    except Exception:
        prv = json.loads(lc["private_test_cases"])
    return pub + prv


def _same(out, exp):
    a, b = out.strip().split(), exp.strip().split()
    if a == b:
        return True
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            if abs(float(x) - float(y)) > 1e-6:
                return False
        except ValueError:
            return False
    return True


def lcb_run(it, text):
    blocks = re.findall(r"```python\n(.*?)```", text, re.S) or re.findall(r"```\n(.*?)```", text, re.S)
    if not blocks:
        return False
    code = blocks[-1]
    meta = json.loads(it["lcb"]["metadata"] or "{}")
    fn = meta.get("func_name")
    if it["lcb"]["starter_code"] and fn:
        code = "from typing import *\nimport collections, heapq, math, itertools, bisect, functools\nfrom collections import *\n" + code + _FUNC_TMPL.format(fn=fn)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        for t in _tests(it["lcb"]):
            try:
                r = subprocess.run([sys.executable, path], input=t["input"], capture_output=True, text=True, timeout=10)
            except subprocess.TimeoutExpired:
                return False
            if r.returncode != 0:
                return False
            if t.get("testtype") == "functional":
                try:
                    if json.loads(r.stdout.strip().split("\n")[-1]) != json.loads(t["output"]):
                        return False
                except Exception:
                    return False
            elif not _same(r.stdout, t["output"]):
                return False
        return True
    finally:
        os.unlink(path)


# ---------------------------------------------------------------- 请求
def ask(task, it, maxtok, seed):
    body = {"model": MODEL, "messages": it["msgs"], "max_tokens": maxtok, "seed": seed, **SAMP,
            "chat_template_kwargs": {"enable_thinking": os.environ.get("THINK", "1") == "1"}}
    for k in range(3):
        try:
            r = requests.post(URL, json=body, timeout=int(os.environ.get("ACC_TIMEOUT", "7200"))).json()
            ch = r["choices"][0]
            msg = ch["message"]
            return msg.get("content") or "", msg.get("reasoning_content") or "", ch.get("finish_reason"), r.get("usage", {}).get("completion_tokens")
        except Exception as e:
            err = e
            time.sleep(5)
    return "", "", f"error {err}", 0


def run(a):
    items = load(a.task)[: a.n or None]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    todo = [it for it in items if it["id"] not in done]
    print(f"[{a.task}] 共 {len(items)} 题，已完成 {len(done)}，待做 {len(todo)}，并发 {a.c}", flush=True)
    lock = threading.Lock()
    t0 = time.time()

    def one(it):
        seed = (zlib.crc32(str(it["id"]).encode()) & 0x7FFFFFFF) if os.environ.get("SEED_BY_ID", "1") == "1" else 0
        content, reasoning, fin, ntok = ask(a.task, it, a.max_tokens or MAXTOK[a.task], seed)
        ok = bool(score(a.task, it, content)) if fin != "length" or content else False
        rec = {"id": it["id"], "ok": ok, "finish": fin, "ntok": ntok, "content": content[-4000:], "reasoning_tail": reasoning[-1000:]}
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return ok

    n_ok = n = 0
    with cf.ThreadPoolExecutor(a.c) as ex:
        for ok in ex.map(one, todo):
            n += 1; n_ok += ok
            if n % 50 == 0:
                print(f"  {n}/{len(todo)} 本次正确率 {n_ok / n:.3f} 用时 {time.time() - t0:.0f}s", flush=True)
    summarize([a.out])


def summarize(paths):
    for p in paths:
        rs = [json.loads(l) for l in open(p)]
        n = len(rs); ok = sum(r["ok"] for r in rs)
        fin = {}
        for r in rs:
            fin[r["finish"]] = fin.get(r["finish"], 0) + 1
        nt = [r["ntok"] or 0 for r in rs]
        think = sum(bool(r.get("reasoning_tail")) for r in rs)
        print(f"{p}：{ok}/{n} = {100 * ok / max(n, 1):.2f}%  结束原因 {fin}  平均生成 {sum(nt) / max(n, 1):.0f} token  思考块非空 {think}/{n}")


def rescore(task, paths):
    """用已存的回答按当前判分器重判（判分器修正后不必重跑）。原地改写 ok 字段。"""
    its = {str(i["id"]): i for i in load(task)}
    for p in paths:
        rs = [json.loads(l) for l in open(p)]
        ch = 0
        for r in rs:
            ok = bool(score(task, its[str(r["id"])], r["content"])) if r["finish"] != "length" or r["content"] else False
            ch += ok != r["ok"]; r["ok"] = ok
        with open(p, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
        print(f"{p}：重判改变 {ch} 题")
    summarize(paths)


def pair(ours, ref, gate=2.0):
    """逐题配对：本方案 vs 参照。翻转拆两个方向并标出其中的截断；精确符号检验 + 配对差 95% 区间（正态近似）。"""
    from math import comb, sqrt
    A = {str(r["id"]): r for r in map(json.loads, open(ours))}
    B = {str(r["id"]): r for r in map(json.loads, open(ref))}
    ids = sorted(set(A) & set(B)); n = len(ids)
    if not n:
        print("无共同题号"); return
    up = [i for i in ids if A[i]["ok"] and not B[i]["ok"]]      # 本方案对、参照错
    dn = [i for i in ids if B[i]["ok"] and not A[i]["ok"]]      # 本方案错、参照对
    tr = lambda L, R: sum(R[i]["finish"] == "length" for i in L)
    pa = 100 * sum(A[i]["ok"] for i in ids) / n; pb = 100 * sum(B[i]["ok"] for i in ids) / n
    k, m = len(up), len(up) + len(dn)
    pv = min(1.0, 2 * sum(comb(m, j) for j in range(min(k, m - k) + 1)) / 2 ** m) if m else 1.0
    d = (k - len(dn)) / n
    se = sqrt(max(m / n - d * d, 0) / n) * 100
    print(f"{os.path.basename(ours)}  共同 {n} 题（1/n={100 / n:.2f}pp）  本方案 {pa:.2f}%  参照 {pb:.2f}%  差 {pa - pb:+.2f}pp  95%区间 [{100 * d - 1.96 * se:+.2f}, {100 * d + 1.96 * se:+.2f}]")
    print(f"  翻转：本方案对/参照错 {k}（其中参照截断 {tr(up, B)}）  本方案错/参照对 {len(dn)}（其中本方案截断 {tr(dn, A)}）  符号检验 p={pv:.3f}")
    print(f"  截断：本方案 {sum(A[i]['finish'] == 'length' for i in ids)}  参照 {sum(B[i]['finish'] == 'length' for i in ids)}")
    print(f"  判定（≤{gate}pp）：{'满足' if pa - pb >= -gate else '不满足'}" + ("；区间下界越过门槛，需更多样本" if 100 * d - 1.96 * se < -gate <= pa - pb else ""))


if __name__ == "__main__":
    if sys.argv[1] == "pair":
        pair(sys.argv[2], sys.argv[3]); sys.exit()
    if sys.argv[1] == "sum":
        summarize(sys.argv[2:]); sys.exit()
    if sys.argv[1] == "rescore":
        rescore(sys.argv[2], sys.argv[3:]); sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("task"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=0); ap.add_argument("--c", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=0)
    run(ap.parse_args())
