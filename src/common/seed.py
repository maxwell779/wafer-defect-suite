"""재현성 시드 고정 (leak-free 평가의 전제)."""
import os, random
import numpy as np


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True  # 속도 (결과 결정성보다 우선)
    except ImportError:
        pass
