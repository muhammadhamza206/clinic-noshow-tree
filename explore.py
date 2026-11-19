"""Summarize clinic appointment no-shows."""

from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda value: f"{value:.6f}")

df = pd.read_csv(Path(__file__).resolve().parent / "clinic_appointments.csv")

print(df.shape)
print(df.describe())

no_show_rows = int((df["no_show"] == "no-show").sum())
no_show_share = no_show_rows / len(df)
print(f"no-show rows: {no_show_rows}")
print(f"no-show share of all rows: {no_show_share:.6f}")

group_columns = ["prior_no_shows", "sms_reminder", "new_patient", "appointment_hour"]
is_no_show = df["no_show"].eq("no-show")

for column in group_columns:
    table = (
        df.assign(is_no_show=is_no_show)
        .groupby(column, dropna=False)
        .agg(row_count=("is_no_show", "size"), no_show_rate=("is_no_show", "mean"))
        .sort_index()
    )
    print(f"\nno-show rate by {column}")
    print(table)
