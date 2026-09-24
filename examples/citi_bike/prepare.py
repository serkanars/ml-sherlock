"""Prepare a reproducible temporal drift sample from official Citi Bike data."""

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd


SOURCES = {
    "train": "https://s3.amazonaws.com/tripdata/JC-202401-citibike-tripdata.csv.zip",
    "production": "https://s3.amazonaws.com/tripdata/JC-202501-citibike-tripdata.csv.zip",
}
MAX_ROWS = 40_000
SEED = 42


def transform(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    started = pd.to_datetime(raw["started_at"], errors="coerce")
    ended = pd.to_datetime(raw["ended_at"], errors="coerce")
    duration = (ended - started).dt.total_seconds().div(60)
    frame = pd.DataFrame({
        "start_hour": started.dt.hour,
        "start_day_of_week": started.dt.dayofweek,
        "is_weekend": started.dt.dayofweek.ge(5).astype(int),
        "rideable_type": raw["rideable_type"].astype("string"),
        "member_type": raw["member_casual"].astype("string"),
        "start_station": raw["start_station_name"].astype("string"),
        "end_station": raw["end_station_name"].astype("string"),
        "start_lat": raw["start_lat"],
        "start_lng": raw["start_lng"],
        "end_lat": raw["end_lat"],
        "end_lng": raw["end_lng"],
        "ride_duration_minutes": duration,
    })
    valid = started.dt.year.eq(year) & duration.between(1, 120)
    return frame.loc[valid].dropna().reset_index(drop=True)


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for index, (split, url) in enumerate(SOURCES.items()):
        raw_path = data_dir / f"{split}.csv.zip"
        if not raw_path.exists():
            print(f"Downloading {url}")
            urlretrieve(url, raw_path)
        with ZipFile(raw_path) as zipped:
            csv_name = next(
                name for name in zipped.namelist()
                if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")
            )
            with zipped.open(csv_name) as stream:
                raw = pd.read_csv(stream)
        frame = transform(raw, 2024 + index)
        if len(frame) > MAX_ROWS:
            frame = frame.sample(MAX_ROWS, random_state=SEED).reset_index(drop=True)
        output = data_dir / f"{split}.csv"
        frame.to_csv(output, index=False)
        print(f"Wrote {output} ({len(frame):,} rows)")


if __name__ == "__main__":
    main()
