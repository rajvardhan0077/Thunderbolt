from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / 'Dataset' / 'royal_enfield_thunderbird_350x_parts_dataset.csv'


def inspect_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    print('Dataset shape:', df.shape)
    print('Columns:', list(df.columns))
    print('\nSample rows:')
    print(df.head(5).to_string(index=False))
    print('\nMissing values:')
    print(df.isna().sum().to_string())
    print('\nDuplicate rows:', int(df.duplicated().sum()))
    print('\nAvailability distribution:')
    print(df['Availability status'].value_counts().to_string())
    return df


if __name__ == '__main__':
    inspect_dataset()
