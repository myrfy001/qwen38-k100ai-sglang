"""覆盖层补丁包。

补丁按作用分为 sched / kv / ops / comm / infra 五个子包，由上一级目录的 `sitecustomize.py`
在目标模块首次被导入时调用各自的 `install(module)`。

`PATCH_DIR` 指向本包的上一级目录（即 `patch/`），内核 `.so` 与其他随包资源均以它为基准定位，
使补丁文件可以移动到子目录而不影响资源查找。
"""
import os

PATCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
