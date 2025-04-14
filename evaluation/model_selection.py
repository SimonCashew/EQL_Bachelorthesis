import pandas as pd

# === Pfad zur CSV anpassen ===
input_csv = "normal_sweep_2_size_4_metrics.csv"
output_csv = "normal_sweep_2_size_4_ms_1.csv"

# CSV laden
df = pd.read_csv(input_csv)

# Relevante Spalten extrahieren
val_col = "Validation Loss"
ex_col = "Extrapolation Validation Loss"
param_col = "Parameter"

# Min-Max-Normalisierung
df["val_norm"] = (df[val_col] - df[val_col].min()) / (df[val_col].max() - df[val_col].min())
df["param_norm"] = (df[param_col] - df[param_col].min()) / (df[param_col].max() - df[param_col].min())
df["exval_norm"] = (df[ex_col] - df[ex_col].min()) / (df[ex_col].max() - df[ex_col].min())

# Score berechnen (nur aus val + param, wie gewünscht)
df["score"] = (df["val_norm"] + df["param_norm"]) / 2

# Nach Score sortieren (aufsteigend)
df_sorted = df.sort_values(by="score", ascending=True)

# Unnötige Normalisierungs-Spalten entfernen (optional)
df_sorted = df_sorted.drop(columns=["val_norm", "param_norm", "exval_norm"])

# Neue CSV speichern
df_sorted.to_csv(output_csv, index=False)
