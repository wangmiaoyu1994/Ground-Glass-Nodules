import pickle

import pandas as pd

ant_path = r"C:\Users\301PBDC\Desktop\dataset\annotation.xlsx"
ant_label = "CsqTexTop-project"

df = pd.read_excel(ant_path)

annotation = {l: {"set": s, "class": c} for s, l, c in zip(df[ant_label], df["lid"], df["class"])}

with open(ant_path.replace(".xlsx", f"_{ant_label}.pkf" if ant_label != "" else ".pkf"), "wb") as pkf:
    pickle.dump(annotation, pkf)
