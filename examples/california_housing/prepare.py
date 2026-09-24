"""Prepare a geographic domain-shift example from sklearn's housing data."""

from pathlib import Path

from sklearn.datasets import fetch_california_housing


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = fetch_california_housing(as_frame=True)
    frame = dataset.frame.rename(columns={dataset.target.name: "median_house_value"})
    train = frame.loc[frame["Latitude"].ge(36)].reset_index(drop=True)
    production = frame.loc[frame["Latitude"].lt(36)].reset_index(drop=True)
    for name, split in (("train", train), ("production", production)):
        output = data_dir / f"{name}.csv"
        split.to_csv(output, index=False)
        print(f"Wrote {output} ({len(split):,} rows)")


if __name__ == "__main__":
    main()
