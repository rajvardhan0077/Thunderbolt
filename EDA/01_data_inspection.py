from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parents[1]
dataset_dir = project_root / "Dataset"
source_path = dataset_dir / "royal_enfield_thunderbird_350x_parts_dataset.csv"
report_path = project_root / "reports" / "data_inspection_report.txt"


def main():
    if not source_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: {source_path}\n"
            "Please place the CSV in the project Dataset folder before running this script."
        )

    print("Loading dataset...")
    df = pd.read_csv(source_path)

    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    print("\nLast 5 rows:")
    print(df.tail().to_string(index=False))

    print("\nShape:")
    print(df.shape)

    print("\nColumn names:")
    print(list(df.columns))

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    duplicate_count = df.duplicated().sum()
    print("\nDuplicate rows:")
    print(duplicate_count)

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    print("\nUnique values for categorical columns:")
    for column in categorical_columns:
        print(f"\n{column}: {sorted(df[column].dropna().unique().tolist())}")

    print("\nStatistical summary:")
    print(df.describe(include='all').to_string())

    report_lines = []
    report_lines.append("Royal Enfield Thunderbird 350X Spare Parts Dataset Inspection Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Source file: {source_path}")
    report_lines.append(f"Shape: {df.shape}")
    report_lines.append("")
    report_lines.append("Columns:")
    report_lines.extend(f"- {column}" for column in df.columns)
    report_lines.append("")
    report_lines.append("Data types:")
    report_lines.append(df.dtypes.to_string())
    report_lines.append("")
    report_lines.append("Missing values:")
    report_lines.append(df.isnull().sum().to_string())
    report_lines.append("")
    report_lines.append(f"Duplicate rows: {duplicate_count}")
    report_lines.append("")
    report_lines.append("Unique values for categorical columns:")
    for column in categorical_columns:
        report_lines.append(f"{column}: {sorted(df[column].dropna().unique().tolist())}")
    report_lines.append("")
    report_lines.append("Statistical summary:")
    report_lines.append(df.describe(include='all').to_string())
    report_lines.append("")
    report_lines.append("First 10 rows:")
    report_lines.append(df.head(10).to_string(index=False))
    report_lines.append("")
    report_lines.append("Last 5 rows:")
    report_lines.append(df.tail().to_string(index=False))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding='utf-8')

    print(f"\nInspection report saved to: {report_path}")


if __name__ == "__main__":
    main()
