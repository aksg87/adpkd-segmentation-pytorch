# %%
import json
import os
import random

# enable lib loading even if not installed as a pip package or in PYTHONPATH
# also convenient for relative paths in example config files
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)


# %%
with open("./stratification/strat_split_2020_09_06.json", "r") as f:
    full_split = json.load(f)
# %%
random.seed = 42
original = full_split["train"]
# %%
# PERCENT = 0.5
PERCENT = 0.9
count = int(len(original) * PERCENT)

# %%
for i in range(4):
    split = random.sample(original, count)
    new = {}
    new["val"] = full_split["val"]
    new["test"] = full_split["test"]
    new["train"] = split
    with open(
        f"./stratification/strat_split_2020_09_06_train_sample_{PERCENT}_{i}.json", # noqa
        "w",
    ) as f:
        json.dump(new, f, indent=4)

# %%
