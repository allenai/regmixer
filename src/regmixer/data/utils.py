import wandb

api = wandb.Api()

all_runs = api.runs(path="ai2-llm/regmixer")
print(len(all_runs))
filtered = [run.config["trainer"]["callbacks"]["wandb"]["group"] for run in all_runs]
print(filtered)
