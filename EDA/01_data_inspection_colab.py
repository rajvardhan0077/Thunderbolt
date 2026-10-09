from pathlib import Path

import pandas as pd


def main():
    project_root = Path.cwd()
    dataset_folder = project_root / "Dataset"
    dataset_path = dataset_folder / "royal_enfield_thunderbird_350x_parts_dataset.csv"
    report_path = project_root / "reports" / "data_inspection_report.txt"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}\n"
            "Upload the CSV to the project folder or create a Dataset folder with the file."
        )

    df = pd.read_csv(dataset_path)

    print("FIRST 10 ROWS")
    print(df.head(10).to_string(index=False))
    print("\nLAST 5 ROWS")
    print(df.tail().to_string(index=False))
    print("\nSHAPE")
    print(df.shape)
    print("\nCOLUMN NAMES")
    print(list(df.columns))
    print("\nDATA TYPES")
    print(df.dtypes.to_string())
    print("\nMISSING VALUES")
    print(df.isnull().sum().to_string())
    print("\nDUPLICATE ROWS")
    print(df.duplicated().sum())

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    print("\nUNIQUE VALUES FOR CATEGORICAL COLUMNS")
    for column in categorical_columns:
        print(f"\n{column}: {sorted(df[column].dropna().unique().tolist())}")

    print("\nSTATISTICAL SUMMARY")
    print(df.describe(include='all').to_string())

    report_lines = [
        "Royal Enfield Thunderbird 350X Spare Parts Dataset Inspection Report",
        "=" * 80,
        f"Source file: {dataset_path}",
        f"Shape: {df.shape}",
        "",
        "Columns:",
        *[f"- {c}" for c in df.columns],
        "",
        "Data types:",
        df.dtypes.to_string(),
        "",
        "Missing values:",
        df.isnull().sum().to_string(),
        "",
        f"Duplicate rows: {df.duplicated().sum()}",
        "",
        "Unique values for categorical columns:",
    ]

    for column in categorical_columns:
        report_lines.append(f"{column}: {sorted(df[column].dropna().unique().tolist())}")

    report_lines.extend([
        "",
        "Statistical summary:",
        df.describe(include='all').to_string(),
        "",
        "First 10 rows:",
        df.head(10).to_string(index=False),
        "",
        "Last 5 rows:",
        df.tail().to_string(index=False),
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
