from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_prediction_csv(prediction: np.ndarray, output_path: str | Path, prefix: str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred = np.asarray(prediction, dtype=np.float32)
    df = pd.DataFrame(
        pred,
        index=[f"sample_{i}" for i in range(pred.shape[0])],
        columns=[f"{prefix}_{i}" for i in range(pred.shape[1])],
    )
    df.to_csv(output_path)
    return output_path
