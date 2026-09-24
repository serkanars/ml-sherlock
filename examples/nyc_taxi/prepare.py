"""Prepare a reproducible temporal drift sample from official NYC TLC data."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


SOURCES = {
    "train": "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-01.parquet",
    "production": "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2025-01.parquet",
}
MAX_ROWS = 30_000
SEED = 42


def transform(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    pickup = pd.to_datetime(raw["lpep_pickup_datetime"], errors="coerce")
    dropoff = pd.to_datetime(raw["lpep_dropoff_datetime"], errors="coerce")
    duration = (dropoff - pickup).dt.total_seconds().div(60)
    frame = pd.DataFrame({
        "pickup_hour": pickup.dt.hour,
        "pickup_day_of_week": pickup.dt.dayofweek,
        "is_weekend": pickup.dt.dayofweek.ge(5).astype(int),
        "passenger_count": raw["passenger_count"],
        "trip_distance": raw["trip_distance"],
        "pickup_zone": raw["PULocationID"].astype("Int64").astype("string"),
        "dropoff_zone": raw["DOLocationID"].astype("Int64").astype("string"),
        "rate_code": raw["RatecodeID"].astype("Int64").astype("string"),
        "store_and_forward": raw["store_and_fwd_flag"].astype("string"),
        "trip_duration_minutes": duration,
    })
    valid = (
        pickup.dt.year.eq(year)
        & duration.between(1, 120)
        & frame["trip_distance"].between(0.1, 100)
        & frame["passenger_count"].between(1, 6)
    )
    return frame.loc[valid].dropna().reset_index(drop=True)


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "lpep_pickup_datetime", "lpep_dropoff_datetime", "passenger_count",
        "trip_distance", "PULocationID", "DOLocationID", "RatecodeID",
        "store_and_fwd_flag",
    ]
    for index, (split, url) in enumerate(SOURCES.items()):
        raw_path = data_dir / f"{split}.parquet"
        if not raw_path.exists():
            print(f"Downloading {url}")
            urlretrieve(url, raw_path)
        frame = transform(pd.read_parquet(raw_path, columns=columns), 2024 + index)
        if len(frame) > MAX_ROWS:
            frame = frame.sample(MAX_ROWS, random_state=SEED).reset_index(drop=True)
        output = data_dir / f"{split}.csv"
        frame.to_csv(output, index=False)
        print(f"Wrote {output} ({len(frame):,} rows)")


if __name__ == "__main__":
    main()
