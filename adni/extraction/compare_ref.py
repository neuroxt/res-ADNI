"""compare_ref.py — REF CSV vs NEW CSV 자동 비교 유틸리티

PTID+VISCODE 기준 inner join → 컬럼별 일치율, 불일치 건수, Pearson r (수치형).

Usage:
    python adni/extraction/compare_ref.py csv/ref/ADNIMERGE_240821.csv csv/ADNIMERGE_260213.csv
    python adni/extraction/compare_ref.py REF.csv NEW.csv --cols DX_bl MMSE MOCA
"""

import argparse
import sys

import numpy as np
import pandas as pd


def compare_csvs(ref_path: str, new_path: str, cols: list[str] | None = None,
                 tolerance: float = 1e-9) -> pd.DataFrame:
    """Compare REF and NEW CSV on PTID+VISCODE inner join.

    Args:
        ref_path: Path to reference CSV
        new_path: Path to new (built) CSV
        cols: Specific columns to compare (None = all common columns)
        tolerance: Absolute tolerance for numeric equality

    Returns:
        DataFrame with columns: column, compared, match, match_pct, mismatch, pearson_r
    """
    ref = pd.read_csv(ref_path, low_memory=False)
    new = pd.read_csv(new_path, low_memory=False)

    # Standardize join keys
    for df in [ref, new]:
        df['RID'] = pd.to_numeric(df['RID'], errors='coerce').astype('Int64')
        df['VISCODE'] = df['VISCODE'].astype(str).str.strip()

    # Inner join on PTID + VISCODE (or RID + VISCODE)
    if 'PTID' in ref.columns and 'PTID' in new.columns:
        merged = ref.merge(new, on=['PTID', 'VISCODE'], suffixes=('_REF', '_NEW'), how='inner')
        join_key = 'PTID+VISCODE'
    else:
        merged = ref.merge(new, on=['RID', 'VISCODE'], suffixes=('_REF', '_NEW'), how='inner')
        join_key = 'RID+VISCODE'

    print(f"REF: {len(ref):,} rows, NEW: {len(new):,} rows")
    print(f"Inner join ({join_key}): {len(merged):,} common rows")

    # Find common columns
    ref_cols = set(ref.columns) - {'PTID', 'RID', 'VISCODE'}
    new_cols = set(new.columns) - {'PTID', 'RID', 'VISCODE'}
    common = sorted(ref_cols & new_cols)

    if cols:
        common = [c for c in cols if c in common]
        missing = [c for c in cols if c not in common]
        if missing:
            print(f"Warning: columns not in common set: {missing}")

    results = []

    for col in common:
        ref_col = f'{col}_REF'
        new_col = f'{col}_NEW'
        if ref_col not in merged.columns or new_col not in merged.columns:
            continue

        r = merged[ref_col]
        n = merged[new_col]

        # Both non-NA
        both_valid = r.notna() & n.notna()
        compared = both_valid.sum()
        if compared == 0:
            results.append({
                'column': col, 'compared': 0, 'match': 0,
                'match_pct': np.nan, 'mismatch': 0, 'pearson_r': np.nan
            })
            continue

        rv = r[both_valid]
        nv = n[both_valid]

        # Try numeric comparison
        rv_num = pd.to_numeric(rv, errors='coerce')
        nv_num = pd.to_numeric(nv, errors='coerce')
        both_numeric = rv_num.notna() & nv_num.notna()

        if both_numeric.sum() > compared * 0.5:
            # Numeric column
            match = (np.abs(rv_num[both_numeric] - nv_num[both_numeric]) <= tolerance).sum()
            # Add exact string matches for non-numeric portion
            non_num = ~both_numeric
            if non_num.any():
                match += (rv[non_num].astype(str) == nv[non_num].astype(str)).sum()

            # Pearson r
            if both_numeric.sum() > 1 and rv_num[both_numeric].std() > 0 and nv_num[both_numeric].std() > 0:
                pearson_r = rv_num[both_numeric].corr(nv_num[both_numeric])
            else:
                pearson_r = np.nan
        else:
            # String/categorical column
            match = (rv.astype(str).str.strip() == nv.astype(str).str.strip()).sum()
            pearson_r = np.nan

        mismatch = compared - match
        match_pct = match / compared * 100 if compared > 0 else np.nan

        results.append({
            'column': col, 'compared': int(compared), 'match': int(match),
            'match_pct': round(match_pct, 2), 'mismatch': int(mismatch),
            'pearson_r': round(pearson_r, 4) if pd.notna(pearson_r) else np.nan
        })

    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values('match_pct', ascending=True).reset_index(drop=True)
    return df_result


def print_report(df: pd.DataFrame):
    """Print formatted comparison report."""
    print(f"\n{'='*80}")
    print(f"{'Column':<30} {'Compared':>8} {'Match':>8} {'Match%':>8} {'Mismatch':>8} {'r':>8}")
    print(f"{'='*80}")

    for _, row in df.iterrows():
        r_str = f"{row['pearson_r']:.4f}" if pd.notna(row['pearson_r']) else '—'
        print(f"{row['column']:<30} {row['compared']:>8,} {row['match']:>8,} "
              f"{row['match_pct']:>7.2f}% {row['mismatch']:>8,} {r_str:>8}")

    print(f"{'='*80}")

    # Summary
    total = len(df)
    perfect = (df['match_pct'] == 100).sum()
    above99 = (df['match_pct'] >= 99).sum()
    above90 = (df['match_pct'] >= 90).sum()

    print(f"\nSummary: {total} columns compared")
    print(f"  100% match: {perfect}")
    print(f"  ≥99% match: {above99}")
    print(f"  ≥90% match: {above90}")
    print(f"  <90% match: {total - above90}")


def main():
    parser = argparse.ArgumentParser(description='Compare REF and NEW ADNIMERGE CSVs')
    parser.add_argument('ref', help='Reference CSV path')
    parser.add_argument('new', help='New (built) CSV path')
    parser.add_argument('--cols', nargs='+', help='Specific columns to compare')
    parser.add_argument('--tolerance', type=float, default=1e-9,
                        help='Numeric tolerance (default: 1e-9)')
    parser.add_argument('--csv', help='Save results to CSV')

    args = parser.parse_args()

    df = compare_csvs(args.ref, args.new, cols=args.cols, tolerance=args.tolerance)
    print_report(df)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nResults saved to: {args.csv}")


if __name__ == '__main__':
    main()
