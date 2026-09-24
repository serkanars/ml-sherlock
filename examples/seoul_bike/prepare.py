"""Prepare the official UCI Seoul Bike Sharing Demand dataset."""

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd


SOURCE = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
COLUMNS = [
    "date", "rented_bike_count", "hour", "temperature_c", "humidity_pct",
    "wind_speed_m_s", "visibility_10m", "dew_point_temperature_c",
    "solar_radiation_mj_m2", "rainfall_mm", "snowfall_cm", "season",
    "holiday", "functioning_day",
]


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "seoul_bike.zip"
    if not archive.exists():
        print(f"Downloading {SOURCE}")
        urlretrieve(SOURCE, archive)
    with ZipFile(archive) as zipped:
        csv_name = next(name for name in zipped.namelist() if name.lower().endswith(".csv"))
        with zipped.open(csv_name) as stream:
            raw = pd.read_csv(stream, encoding="unicode_escape")
    if len(raw.columns) != len(COLUMNS):
        raise ValueError(f"Unexpected Seoul Bike schema: {list(raw.columns)}")
    raw.columns = COLUMNS
    dates = pd.to_datetime(raw.pop("date"), format="%d/%m/%Y", errors="raise")
    raw["month"] = dates.dt.month
    raw["day_of_week"] = dates.dt.dayofweek
    raw["is_weekend"] = dates.dt.dayofweek.ge(5).astype(int)
    train = raw.loc[dates.lt("2018-09-01")].reset_index(drop=True)
    production = raw.loc[dates.ge("2018-09-01")].reset_index(drop=True)
    for name, frame in (("train", train), ("production", production)):
        output = data_dir / f"{name}.csv"
        frame.to_csv(output, index=False)
        print(f"Wrote {output} ({len(frame):,} rows)")


if __name__ == "__main__":
    main()
