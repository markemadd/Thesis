"""
Shared utility functions for the AD Proteomics Pipeline.
"""
import os
import json
import pandas as pd
import numpy as np


def save_csv(df, filename, output_dir):
    """Save a DataFrame to CSV in the output directory."""
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=True)
    print(f"  ✓ Saved {path}")
    return path


def save_json(obj, filename, output_dir):
    """Save a dict/list to JSON in the output directory."""
    path = os.path.join(output_dir, filename)

    def _convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.Series):
            return o.to_dict()
        raise TypeError(f"Type {type(o)} not serializable")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_convert)
    print(f"  ✓ Saved {path}")
    return path


def load_csv(filename, output_dir, index_col=0):
    """Load a CSV from the output directory."""
    path = os.path.join(output_dir, filename)
    return pd.read_csv(path, index_col=index_col)


def load_json(filename, output_dir):
    """Load a JSON from the output directory."""
    path = os.path.join(output_dir, filename)
    with open(path) as f:
        return json.load(f)


def print_header(title):
    """Print a formatted section header."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_step(step_num, description):
    """Print a numbered step."""
    print(f"\n  [{step_num}] {description}")
    print("  " + "-" * 60)
