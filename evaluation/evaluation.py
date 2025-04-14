import wandb
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

ENTITY = "simonb03-university-of-t-bingen"  
PROJECT = "OPR_sweep_2_size_2"
SWEEP_ID = "cv7g3y6t"  

api = wandb.Api()
sweep = api.sweep(f"{ENTITY}/{PROJECT}/{SWEEP_ID}")
runs = sweep.runs

loss_dict = {}
max_steps = 0

summary_data = []

for run in tqdm(runs, desc="Runs verarbeiten"):
    run_name = run.name
    run_id = run.id

    try:
        history = run.history(keys=["loss"], pandas=True)
    except wandb.CommError:
        print("Fehler")
        continue

    if "loss" in history.columns:
        losses = history["loss"].dropna().values
        loss_dict[run_name] = losses
        max_steps = max(max_steps, len(losses))

    summary_data.append({
        "Run": run_name,
        "Validation Loss": run.summary.get("Validation Loss"),
        "Extrapolation Validation Loss": run.summary.get("Extrapolation Validation Loss"),
        "Parameter": run.summary.get("Parameter"),
    })

loss_wide_df = pd.DataFrame({name: pd.Series(losses) for name, losses in loss_dict.items()})
loss_wide_df.index.name = "Step"
loss_wide_df.reset_index(inplace=True)

loss_wide_df.to_csv(f"{PROJECT}_loss.csv", index=False)
pd.DataFrame(summary_data).to_csv(f"{PROJECT}_metrics.csv", index=False)
