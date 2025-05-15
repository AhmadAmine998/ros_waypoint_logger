import pandas as pd


def load_waypoints(csv_path):
    with open(csv_path, "r") as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") and ";" in stripped:
            header_idx = i
            raw = stripped.lstrip("#")
            break
    if header_idx is None:
        raise ValueError(f"No header line found in {csv_path}")
    cols = [c.strip() for c in raw.split(";")]
    df = (
        pd.read_csv(
            csv_path,
            skiprows=header_idx + 1,
            header=None,
            names=cols,
            sep=";",
            skipinitialspace=True,
        )
        .dropna()
        .astype(float)
    )
    return df
