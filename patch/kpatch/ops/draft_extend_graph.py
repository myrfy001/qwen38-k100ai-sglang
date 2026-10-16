"""草稿扩展（draft extend）进 CUDA 图。

原状：eagle_worker_v2.py:492 `supports_cuda_draft_extend_graph = (_is_cuda or _is_musa) and 后端在白名单`，
  本机 is_hip 且 FlashAttentionBackend 不在白名单，草稿扩展每轮 eager（源码审查 A2，约 1.6 ms/轮）。
  但 flashattention_backend.py 已实现 DRAFT_EXTEND_V2 的捕获（:2592）与回放（:3023）元数据，缺的只是白名单。
改法：原 _capture_cuda_graphs 跑完后，若草稿扩展图未建、后端是 FlashAttentionBackend、未设 SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH，
  按原代码同样方式建 EAGLEDraftExtendCudaGraphRunner。选词仍由 worker 在图外对选中行做（:1026 注释），与 eager 路径同一段代码。
开关 SGLANG_DRAFT_EXTEND_GRAPH=1。验收看接受长度与 gsm8k。
"""
import os
import time


def install(mod):
    cls = mod.EagleDraftWorker
    if getattr(cls, "_dx_graph_installed", False):
        return
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
    # 旧镜像叫 _capture_cuda_graphs；sglang 0.5.12（Qwen3.8 镜像）改名 init_cuda_graphs，白名单只有 Triton/TRTLLM-MLA/NSA（10-10）
    name = "_capture_cuda_graphs" if hasattr(cls, "_capture_cuda_graphs") else "init_cuda_graphs"
    orig = getattr(cls, name)

    def _capture_cuda_graphs(self):
        orig(self)
        if self.server_args.disable_cuda_graph:
            return
        if self.cuda_graph_runner_for_draft_extend is not None or self.cuda_graph_runner is None:
            return
        be = self.draft_extend_attn_backend
        if not isinstance(be, FlashAttentionBackend):
            print(f"[draft_extend_graph] 未接管：后端 {type(be).__name__}", flush=True)
            return
        if os.environ.get("SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH", "0") not in ("0", "", "false", "False"):
            return
        tic = time.perf_counter()
        self.cuda_graph_runner_for_draft_extend = mod.EAGLEDraftExtendCudaGraphRunner(self)
        print(f"[draft_extend_graph] ✅ 草稿扩展图已捕获 {time.perf_counter() - tic:.1f} s pid={os.getpid()}", flush=True)

    setattr(cls, name, _capture_cuda_graphs)
    cls._dx_graph_installed = True
    print(f"[draft_extend_graph] 已安装（挂在 {name}） pid={os.getpid()}", flush=True)
