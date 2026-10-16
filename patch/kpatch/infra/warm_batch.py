"""启动预热补上框架预热覆盖不到的内核特化。

原状：框架的 --warmups prefill_shapes 只逐条发送长度为 64 整倍数的单个请求。Triton 按整型参数「能否被 16 整除」分别编译内核，
      而实际请求的 token 数是任意的；全新容器（Triton 磁盘缓存为空）里，第一次出现 token 数为奇数的图外预填充批时，
      fused_moe_kernel、_layer_norm_fwd_1pass_kernel 要即时编译约 3 秒，落在该请求之后所有排队请求的首响上
      （实测 2048×16 测点：1.5 s → 4.3 s；同一容器第二次启动即恢复，编译时间窗内新增的缓存项即上述两个内核）。
改动：在 prefill_shapes 之后补发 token 数为奇数的请求（覆盖各 MoE 配置档），以及一组并发请求，把编译放到服务就绪之前。
      只影响启动（约 15 秒），不影响任何请求的计算。
"""
import asyncio
import os

_GROUPS = os.environ.get("SGLANG_WARM_BATCH_GROUPS", "1x2501,1x3071,1x4287,1x6143,1x8191,1x12287,1x16383,1x24575,1x32767,16x2111,4x4159")


def install(mod):
    reg = getattr(mod, "_warmup_registry", None)
    if reg is None or "prefill_shapes" not in reg or getattr(mod, "_warm_batch_installed", False):
        return
    orig = reg["prefill_shapes"]
    np = mod.np
    GenerateReqInput = mod.GenerateReqInput

    async def one(tm, size, dis):
        req = GenerateReqInput(input_ids=np.random.randint(2**16, size=[size]).tolist(),
                               sampling_params={"max_new_tokens": 16, "temperature": 0.0, "ignore_eos": True})
        if dis != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = mod.FAKE_BOOTSTRAP_HOST
        async for _ in tm.generate_request(req, None):
            pass

    async def prefill_shapes(disaggregation_mode, tokenizer_manager):
        await orig(disaggregation_mode, tokenizer_manager)
        import time
        t0 = time.time()
        try:
            for g in _GROUPS.split(","):
                n, size = (int(x) for x in g.split("x"))
                await asyncio.gather(*[one(tokenizer_manager, size, disaggregation_mode) for _ in range(n)])
            print(f"[warm_batch] ✅ 奇数长度与并发批预热完成：{_GROUPS}，用时 {time.time() - t0:.1f} s", flush=True)
        except Exception as e:
            print(f"[warm_batch] ⚠ 奇数长度与并发批预热失败（{e!r}）：不影响服务，但全新容器里第一个高并发测点的首响会偏慢", flush=True)

    reg["prefill_shapes"] = prefill_shapes
    mod._warm_batch_installed = True
    print(f"[warm_batch] 已安装 (pid={os.getpid()})", flush=True)
