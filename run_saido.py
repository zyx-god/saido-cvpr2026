#!/usr/bin/env python3
"""SAIDO 启动脚本 - 注入缺失的 avalanche 模块"""
import sys
import types

# ===== 提前注入所有缺失的模块，避免 avalanche/benchmarks/classic 导入报错 =====

# 1. core50 子包 (classic/core50.py 需要)
_core50_core = types.ModuleType("core50")
_core50_core.CORe50Dataset = type("CORe50Dataset", (), {})
_core50_pkg = types.ModuleType("core50")
_core50_pkg.core50 = _core50_core
sys.modules["avalanche.benchmarks.datasets.core50"] = _core50_pkg
sys.modules["avalanche.benchmarks.datasets.core50.core50"] = _core50_core

# 2. datasets 中缺失的类 (在 avalanche.benchmarks.datasets 命名空间下注册)
sys.modules["avalanche.benchmarks.datasets.cub200"] = type(sys)("cub200")
sys.modules["avalanche.benchmarks.datasets.cub200"].CUB200 = type("CUB200", (), {})

sys.modules["avalanche.benchmarks.datasets.endless_cl_sim"] = type(sys)("endless_cl_sim")
sys.modules["avalanche.benchmarks.datasets.endless_cl_sim"].EndlessCLSim = type("EndlessCLSim", (), {})

sys.modules["avalanche.benchmarks.datasets.mini_imagenet"] = type(sys)("mini_imagenet")
sys.modules["avalanche.benchmarks.datasets.mini_imagenet"].MiniImageNet = type("MiniImageNet", (), {})

sys.modules["avalanche.benchmarks.datasets.openloris"] = type(sys)("openloris")
sys.modules["avalanche.benchmarks.datasets.openloris"].OpenLoris = type("OpenLoris", (), {})

sys.modules["avalanche.benchmarks.datasets.stream51"] = type(sys)("stream51")
sys.modules["avalanche.benchmarks.datasets.stream51"].Stream51 = type("Stream51", (), {})

sys.modules["avalanche.benchmarks.datasets.tiny_imagenet"] = type(sys)("tiny_imagenet")
sys.modules["avalanche.benchmarks.datasets.tiny_imagenet"].TinyImagenet = type("TinyImagenet", (), {})

sys.modules["avalanche.benchmarks.datasets.inaturalist"] = type(sys)("inaturalist")
sys.modules["avalanche.benchmarks.datasets.inaturalist"].INATURALIST2018 = type("INATURALIST2018", (), {})

# ===== 现在可以安全导入并运行训练 =====
import avalanche

# 补上 datasets 命名空间中的类
from avalanche.benchmarks import datasets as ds
for mod_name in ["cub200", "endless_cl_sim", "mini_imagenet", "openloris",
                  "stream51", "tiny_imagenet", "inaturalist"]:
    m = sys.modules[f"avalanche.benchmarks.datasets.{mod_name}"]
    for attr in dir(m):
        if not attr.startswith("_"):
            setattr(ds, attr, getattr(m, attr))

exec(open("training.py", encoding="utf-8").read())
