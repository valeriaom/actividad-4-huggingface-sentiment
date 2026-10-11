import numpy as np
import pandas as pd

def minmax_benefit(series):
    s = series.astype(float)
    if np.isclose(s.max(), s.min()):
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def minmax_cost(series):
    s = series.astype(float)
    if np.isclose(s.max(), s.min()):
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s.max() - s) / (s.max() - s.min())

def multicriteria_score(df):
    result = df.copy()
    result["score_f1"] = minmax_benefit(result["F1_macro"])
    result["score_accuracy"] = minmax_benefit(result["Accuracy"])
    result["score_latencia"] = minmax_cost(result["Latencia_media_ms"])
    result["score_memoria"] = minmax_cost(result["VRAM_pico_MB"])
    result["Score_multicriterio"] = (
        0.50 * result["score_f1"]
        + 0.20 * result["score_accuracy"]
        + 0.20 * result["score_latencia"]
        + 0.10 * result["score_memoria"]
    )
    return result.sort_values("Score_multicriterio", ascending=False)
