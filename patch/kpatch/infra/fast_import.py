"""消除 vllm_hcu 的 import 钩子在热路径上的开销。

★ 实测依据 (cProfile, 20 次 prefill 前向 × 40 个段间 eager 断点, 共 2.275 s):
    _custom_import                     37,840 次   累计 0.432 s  = **19%**
    _import_targets_exchange_candidate 37,840 次   自身 0.150 s
    str.startswith                    302,720 次   自身 0.076 s
  每次前向要走 1,892 次这个钩子。

★ 热路径为什么有这么多 import:
  Triton 海光后端 `backends/hcu/compiler.py:246 is_within_4gb` **函数体内写了 `import torch`**,
  而它在每次内核发射的参数特化里对每个张量调用一次(实测 32,400 次/20 前向)。
  命中 sys.modules 缓存本应约 1 us, 但 __import__ 被换成 _custom_import 后慢一个量级。
  对账: is_within_4gb 累计 0.680 - 真正干活的 get_tensor_physical_size 0.232 = 0.448 s ≈ _custom_import 的 0.432 s。

★ 安全性: 钩子的全部映射共 8 条, **无一例外以 "vllm." 开头**
    MODULE_EXCHANGE_MAP 4 条 + _custom_import 内的 module_mappings 4 条。
  `_import_targets_exchange_candidate` 对**祖先包**也返回 True(canonical.startswith(name+".")),
  但祖先包同样以 "vllm" 开头 -> `name[:4]=="vllm"` 是安全的**超集**。
  以 vllm 开头 -> 原样交给厂商的 _custom_import(语义一字不差); 否则 -> 直通原始 __import__。

★ 安装方式 (踩过两次坑):
  1) 钩子**不是**在 import vllm_hcu 时装的 —— 运行时实测 `import vllm_hcu` 前后 builtins.__import__ 不变;
     真正装它的是 vllm_hcu/__init__.py:7 `hcu_platform_plugin()`, 那是 vLLM 平台插件入口点。
  2) 所以不替换 builtins.__import__, 而是**替换 _custom_import 函数本身**:
     import_hook() 里 `builtins.__import__ = _custom_import` 按模块全局名在调用时解析,
     无论它何时被调用, 装上去的都已经是快版本。若钩子已装, 一并就地替换。

不修改任何厂商库文件。开关 SGLANG_FAST_IMPORT=0 关闭。
"""
import os, builtins

_enabled = os.environ.get("SGLANG_FAST_IMPORT", "1") != "0"

def install(mod):
    """mod = vllm_hcu.patch_utils"""
    if getattr(mod, "_fast_import_installed", False) or not _enabled:
        return
    orig = getattr(mod, "OLD_IMPORT_HOOK", None)
    slow = getattr(mod, "_custom_import", None)
    if orig is None or slow is None:
        print("[fast_import] 未找到 OLD_IMPORT_HOOK/_custom_import, 跳过", flush=True)
        return

    def fast(name, globals=None, locals=None, fromlist=(), level=0):
        if name[:4] == "vllm":
            return slow(name, globals, locals, fromlist, level)
        return orig(name, globals, locals, fromlist, level)

    mod._custom_import = fast                      # 之后任何 import_hook() 装的都是快版本
    already = builtins.__import__ is slow
    if already:
        builtins.__import__ = fast                 # 已经装上的, 就地替换
    mod._fast_import_installed = True
    print("[fast_import] 已安装 (pid=%d, 钩子此前%s装上): vllm.* 走原钩子, 其余直通"
          % (os.getpid(), "已" if already else "未"), flush=True)
