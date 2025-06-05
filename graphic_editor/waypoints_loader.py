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

    # Define all possible columns we might encounter
    default_cols = {
        "s_m": float,
        "x_m": float,
        "y_m": float,
        "z_m": float,
        "vx_mps": float,
        "ax_mps2": float,
        "psi_rad": float,
        "kappa_radpm": float,
        "roll_rad": float,
        "pitch_rad": float,
        "yaw_rad": float,
    }

    # Read CSV with all potential columns, then select only those present
    df_raw = pd.read_csv(
        csv_path,
        skiprows=header_idx + 1,
        header=None,
        names=cols,  # Use actual columns from file for initial read
        sep=";",
        skipinitialspace=True,
    )

    # Create an empty DataFrame with the desired schema
    df = pd.DataFrame(columns=default_cols.keys())

    # Fill the DataFrame with data from df_raw, casting to float
    # and adding missing columns with NaN
    for col in default_cols.keys():
        if col in df_raw.columns:
            df[col] = df_raw[col].astype(float)
        else:
            df[col] = pd.NA  # Use pandas NA for missing data to allow float type

    df = df.dropna(
        subset=["s_m", "x_m", "y_m"]
    )  # Drop rows if essential 2D path data is missing

    # Fill NaNs for optional columns with 0.0 after ensuring they are float
    optional_cols = [
        "z_m",
        "vx_mps",
        "ax_mps2",
        "psi_rad",
        "kappa_radpm",
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
    ]
    for col in optional_cols:
        if col in df.columns:
            # Ensure column is numeric before fillna. If it became object due to all NAs, convert.
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0.0)
        else:  # Should not happen if we added all default_cols keys before
            df[col] = 0.0

    return df
