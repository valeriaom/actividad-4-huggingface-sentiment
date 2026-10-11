from pathlib import Path
import numpy as np
from datasets import load_dataset

def to_sentiment(label):
    stars=int(label)+1
    return "NEG" if stars<=2 else ("NEU" if stars==3 else "POS")

ds=load_dataset("SetFit/amazon_reviews_multi_es",split="test")
df=ds.to_pandas().reset_index(drop=True)
df["id"]=np.arange(len(df)); df["sentiment"]=df["label"].apply(to_sentiment)
Path("data").mkdir(exist_ok=True)
df[["id","text","label","sentiment"]].to_csv("data/dataset_evaluacion.csv",index=False)
