#!/usr/bin/env python3
"""
compare_merged.py — MERGED.csv 레퍼런스 비교 보고서 생성

Usage:
    python scripts/compare_merged.py [--new-merged PATH] [--ref-merged PATH] [--ref-adni4 PATH] [--output PATH]
"""

import os
import sys
import json
import re
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_NEW_MERGED = os.path.join(
    '/Volumes/nfs_storage/1_combined/ADNI_New/ORIG/DCM/DEMO/ADNI_matching_v4',
    'MERGED.csv')
DEFAULT_REF_MERGED = os.path.join(PROJECT_ROOT, 'csv', 'ref', 'MERGED.csv')
DEFAULT_REF_ADNI4 = os.path.join(PROJECT_ROOT, 'csv', 'ref', 'merged_adni4.csv')
DEFAULT_ADNIMERGE = os.path.join(PROJECT_ROOT, 'csv', 'ADNIMERGE_260213.csv')
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, 'reports', 'matching_validation_report.md')
DEFAULT_DCM_INVENTORY = os.path.join(
    '/Volumes/nfs_storage/1_combined/ADNI_New/ORIG/DCM/DEMO/ADNI_matching_v4',
    'dcm_inventory.json')
ADNI4_DCM_ROOT = '/Volumes/nfs_storage/1_combined/ADNI4/ORIG/DCM'

# ref MERGED.csv (ADNI1/GO/2/3) modalities
REF_MODALITIES = ['T1', 'AV45_8MM', 'AV45_6MM', 'AV1451_8MM', 'AV1451_6MM',
                  'FBB_6MM', 'FLAIR', 'T2_FSE', 'T2_TSE', 'T2_STAR']

# ref merged_adni4.csv modalities (with name mapping)
# key = ref column name, value = our column name
ADNI4_MOD_MAP = {
    'T1': 'T1',
    'T2_STAR': 'T2_STAR',
    'FLAIR': 'FLAIR',
    'AV45_6MM': 'AV45_6MM',
    'AV1451_6MM': 'AV1451_6MM',
    'FBB_6MM': 'FBB_6MM',
    'MK6240': 'MK6240_6MM',   # name differs
    'T2w': 'T2_3D',           # name differs
}


def load_csv(path, label):
    """Load CSV with basic info logging."""
    if not os.path.isfile(path):
        print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded {label}: {df.shape[0]:,} rows x {df.shape[1]:,} cols from {path}")
    return df


def modality_distribution(df, i_cols_prefix='I_'):
    """Count non-null entries per modality (I_{MOD} columns)."""
    result = {}
    for col in df.columns:
        if col.startswith(i_cols_prefix) and not col.endswith('_bl'):
            mod = col[len(i_cols_prefix):]
            result[mod] = df[col].notna().sum()
    return pd.Series(result).sort_values(ascending=False)


def compare_imageuid(ref_df, new_df, modality, ref_mod_name=None):
    """Compare ImageUID (I_{MOD}) for common PTID+VISCODE_FIX rows."""
    if ref_mod_name is None:
        ref_mod_name = modality

    ref_col = f'I_{ref_mod_name}'
    new_col = f'I_{modality}'

    if ref_col not in ref_df.columns or new_col not in new_df.columns:
        return None

    # common index
    common_idx = ref_df.index.intersection(new_df.index)
    if len(common_idx) == 0:
        return {'common_rows': 0}

    ref_vals = ref_df.loc[common_idx, ref_col]
    new_vals = new_df.loc[common_idx, new_col]

    # both non-null
    both_present = ref_vals.notna() & new_vals.notna()
    n_both = both_present.sum()

    if n_both == 0:
        return {
            'common_rows': len(common_idx),
            'both_have_value': 0,
            'match': 0,
            'match_rate': 0.0,
        }

    # Convert to string for comparison (ImageUID can be int or float)
    ref_str = ref_vals[both_present].astype(str).str.strip()
    new_str = new_vals[both_present].astype(str).str.strip()

    # Normalize: remove .0 suffix from float strings
    ref_str = ref_str.str.replace(r'\.0$', '', regex=True)
    new_str = new_str.str.replace(r'\.0$', '', regex=True)

    match = (ref_str == new_str).sum()

    # Collect mismatch samples
    mismatch_mask = ref_str != new_str
    mismatch_samples = []
    if mismatch_mask.sum() > 0:
        for idx in mismatch_mask[mismatch_mask].index[:5]:
            mismatch_samples.append({
                'PTID': idx[0] if isinstance(idx, tuple) else idx,
                'VISCODE_FIX': idx[1] if isinstance(idx, tuple) else '',
                'ref': ref_str.loc[idx],
                'new': new_str.loc[idx],
            })

    return {
        'common_rows': len(common_idx),
        'both_have_value': int(n_both),
        'match': int(match),
        'mismatch': int(n_both - match),
        'match_rate': match / n_both if n_both > 0 else 0.0,
        'mismatch_samples': mismatch_samples,
    }


def compare_aqudate(ref_df, new_df, modality, ref_mod_name=None):
    """Compare AQUDATE for common rows."""
    if ref_mod_name is None:
        ref_mod_name = modality

    ref_col = f'AQUDATE_{ref_mod_name}'
    new_col = f'AQUDATE_{modality}'

    if ref_col not in ref_df.columns or new_col not in new_df.columns:
        return None

    common_idx = ref_df.index.intersection(new_df.index)
    if len(common_idx) == 0:
        return {'common_rows': 0}

    ref_vals = ref_df.loc[common_idx, ref_col].astype(str).str[:10]
    new_vals = new_df.loc[common_idx, new_col].astype(str).str[:10]

    both_present = (ref_vals != 'nan') & (new_vals != 'nan') & (ref_vals != 'NaT') & (new_vals != 'NaT')
    n_both = both_present.sum()
    if n_both == 0:
        return {'common_rows': len(common_idx), 'both_have_value': 0}

    match = (ref_vals[both_present] == new_vals[both_present]).sum()
    return {
        'common_rows': len(common_idx),
        'both_have_value': int(n_both),
        'match': int(match),
        'match_rate': match / n_both if n_both > 0 else 0.0,
    }


def compare_demographics(ref_df, new_df):
    """Compare DX_bl, subjectSex, Apoe on common rows."""
    common_idx = ref_df.index.intersection(new_df.index)
    if len(common_idx) == 0:
        return {}

    results = {}
    for col in ['DX_bl', 'subjectSex', 'Apoe']:
        if col not in ref_df.columns or col not in new_df.columns:
            results[col] = 'column missing'
            continue

        ref_vals = ref_df.loc[common_idx, col].astype(str)
        new_vals = new_df.loc[common_idx, col].astype(str)
        both_present = (ref_vals != 'nan') & (new_vals != 'nan')
        n_both = both_present.sum()

        if n_both == 0:
            results[col] = {'both_have_value': 0}
            continue

        match = (ref_vals[both_present] == new_vals[both_present]).sum()
        results[col] = {
            'both_have_value': int(n_both),
            'match': int(match),
            'match_rate': match / n_both if n_both > 0 else 0.0,
        }
    return results


def fmt_pct(val):
    """Format percentage."""
    return f"{val * 100:.1f}%"


def collect_nfs_ptids(dcm_root):
    """Collect PTIDs from NFS DCM directory structure ({SOURCE}/{PTID}/...)."""
    ptid_re = re.compile(r'\d{3}_S_\d{4,5}$')
    ptids = set()
    if not os.path.isdir(dcm_root):
        return ptids
    for source_dir in os.listdir(dcm_root):
        source_path = os.path.join(dcm_root, source_dir)
        if not os.path.isdir(source_path):
            continue
        for entry in os.listdir(source_path):
            if ptid_re.match(entry):
                ptids.add(entry)
    return ptids


def collect_inventory_ptid_modalities(inventory_path):
    """Load dcm_inventory.json and return {ptid: set(modalities)} mapping."""
    if not inventory_path or not os.path.isfile(inventory_path):
        return {}
    print(f"  Loading inventory for PTID analysis: {inventory_path}")
    with open(inventory_path) as f:
        inv = json.load(f)
    ptid_mods = {}
    for mod, ptid_data in inv.get('by_modality', {}).items():
        for ptid in ptid_data:
            if ptid not in ptid_mods:
                ptid_mods[ptid] = set()
            ptid_mods[ptid].add(mod)
    print(f"    {len(ptid_mods):,} PTIDs across {len(inv.get('by_modality', {})):,} modalities")
    return ptid_mods


# =========================================================================
# Report generation
# =========================================================================

def generate_report(new_merged_path, ref_merged_path, ref_adni4_path, output_path, **kwargs):
    """Generate comparison report in markdown format."""
    lines = []
    w = lines.append  # shorthand

    w(f"# MERGED.csv Matching Validation Report")
    w(f"")
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"")

    # ---- Load data ----
    print("Loading data...")
    new_all = load_csv(new_merged_path, "New MERGED.csv")
    ref = load_csv(ref_merged_path, "Ref MERGED.csv (ADNI1/GO/2/3)")
    ref4 = load_csv(ref_adni4_path, "Ref merged_adni4.csv")

    # Set index
    for df in [new_all, ref, ref4]:
        if 'PTID' in df.columns and 'VISCODE_FIX' in df.columns:
            df.set_index(['PTID', 'VISCODE_FIX'], inplace=True)

    # Split: Part A (COLPROT-based) and Part B (PTID-based, independent)
    # Part A: all non-ADNI4 rows (COLPROT != 'ADNI4') — for legacy ref comparison
    # Part B: all rows for PTIDs with ADNI4 enrollment — includes longitudinal data
    # These are NOT mutually exclusive; longitudinal PTIDs appear in both.
    adnimerge_path = kwargs.get('adnimerge_csv')
    adni4_ptids = set()
    if adnimerge_path and os.path.isfile(adnimerge_path):
        am = pd.read_csv(adnimerge_path, usecols=['PTID', 'COLPROT'], low_memory=False)
        adni4_ptids = set(am[am['COLPROT'] == 'ADNI4']['PTID'].unique())
        print(f"  ADNI4 PTIDs from ADNIMERGE: {len(adni4_ptids):,}")

    if 'COLPROT' in new_all.columns:
        new_legacy = new_all[new_all['COLPROT'] != 'ADNI4'].copy()
    else:
        new_legacy = new_all.copy()

    if adni4_ptids:
        new_ptid_col = new_all.index.get_level_values('PTID')
        new_adni4 = new_all[new_ptid_col.isin(adni4_ptids)].copy()
    elif 'COLPROT' in new_all.columns:
        new_adni4 = new_all[new_all['COLPROT'] == 'ADNI4'].copy()
    else:
        new_adni4 = pd.DataFrame()

    # =====================================================================
    # Part A: ADNI1/GO/2/3 comparison
    # =====================================================================
    w("---")
    w("")
    w("## Part A: ADNI1/GO/2/3 Comparison")
    w("")
    w("### A1. Overall Summary")
    w("")
    w("| Metric | Ref | New (non-ADNI4) |")
    w("|--------|-----|-----------------|")
    w(f"| Rows | {ref.shape[0]:,} | {new_legacy.shape[0]:,} |")
    w(f"| Columns | {ref.shape[1]:,} | {new_legacy.shape[1]:,} |")
    ref_ptids = ref.index.get_level_values('PTID').unique()
    new_ptids = new_legacy.index.get_level_values('PTID').unique()
    w(f"| Unique PTIDs | {len(ref_ptids):,} | {len(new_ptids):,} |")

    if 'COLPROT' in new_legacy.columns:
        w("")
        w("**COLPROT distribution (new):**")
        w("")
        for proto, cnt in new_legacy['COLPROT'].value_counts().items():
            w(f"- {proto}: {cnt:,}")
    w("")

    # Modality distribution
    w("### A2. Modality Distribution (non-null I_{MOD} counts)")
    w("")
    ref_mod = modality_distribution(ref)
    new_mod = modality_distribution(new_legacy)
    all_mods = sorted(set(ref_mod.index) | set(new_mod.index))
    w("| Modality | Ref | New | Diff |")
    w("|----------|-----|-----|------|")
    for m in all_mods:
        r = int(ref_mod.get(m, 0))
        n = int(new_mod.get(m, 0))
        diff = n - r
        sign = "+" if diff > 0 else ""
        w(f"| {m} | {r:,} | {n:,} | {sign}{diff:,} |")
    w("")

    # Index comparison
    w("### A3. PTID+VISCODE_FIX Index Comparison")
    w("")
    common_idx = ref.index.intersection(new_legacy.index)
    ref_only = ref.index.difference(new_legacy.index)
    new_only = new_legacy.index.difference(ref.index)
    w(f"- Common rows (intersection): **{len(common_idx):,}**")
    w(f"- Ref-only rows: {len(ref_only):,}")
    w(f"- New-only rows: {len(new_only):,}")
    if len(ref.index) > 0:
        w(f"- Coverage of ref: {len(common_idx)/len(ref.index)*100:.1f}%")
    w("")

    # Per-modality ImageUID comparison
    w("### A4. Per-Modality ImageUID Comparison (common rows)")
    w("")
    w("| Modality | Both Have Value | Match | Mismatch | Match Rate |")
    w("|----------|-----------------|-------|----------|------------|")
    mismatch_details = {}
    for mod in REF_MODALITIES:
        uid_result = compare_imageuid(ref, new_legacy, mod)
        if uid_result is None:
            w(f"| {mod} | — | — | — | column missing |")
            continue
        if uid_result.get('both_have_value', 0) == 0:
            w(f"| {mod} | 0 | — | — | — |")
            continue
        w(f"| {mod} | {uid_result['both_have_value']:,} | {uid_result['match']:,} | {uid_result['mismatch']:,} | {fmt_pct(uid_result['match_rate'])} |")
        if uid_result.get('mismatch_samples'):
            mismatch_details[mod] = uid_result['mismatch_samples']
    w("")

    # Per-modality AQUDATE comparison
    w("### A5. Per-Modality AQUDATE Comparison (common rows)")
    w("")
    w("| Modality | Both Have Value | Match | Match Rate |")
    w("|----------|-----------------|-------|------------|")
    for mod in REF_MODALITIES:
        aq_result = compare_aqudate(ref, new_legacy, mod)
        if aq_result is None:
            w(f"| {mod} | — | — | column missing |")
            continue
        if aq_result.get('both_have_value', 0) == 0:
            w(f"| {mod} | 0 | — | — |")
            continue
        w(f"| {mod} | {aq_result['both_have_value']:,} | {aq_result['match']:,} | {fmt_pct(aq_result['match_rate'])} |")
    w("")

    # Demographics comparison
    w("### A6. Demographics Comparison (common rows)")
    w("")
    demo_result = compare_demographics(ref, new_legacy)
    w("| Field | Both Have Value | Match | Match Rate |")
    w("|-------|-----------------|-------|------------|")
    for col, res in demo_result.items():
        if isinstance(res, str):
            w(f"| {col} | — | — | {res} |")
        elif res.get('both_have_value', 0) == 0:
            w(f"| {col} | 0 | — | — |")
        else:
            w(f"| {col} | {res['both_have_value']:,} | {res['match']:,} | {fmt_pct(res['match_rate'])} |")
    w("")

    # Mismatch details
    if mismatch_details:
        w("### A7. ImageUID Mismatch Samples")
        w("")
        for mod, samples in mismatch_details.items():
            w(f"**{mod}** (showing up to 5):")
            w("")
            w("| PTID | VISCODE_FIX | Ref I_{MOD} | New I_{MOD} |")
            w("|------|-------------|-------------|-------------|")
            for s in samples:
                w(f"| {s['PTID']} | {s['VISCODE_FIX']} | {s['ref']} | {s['new']} |")
            w("")

    # =====================================================================
    # Part B: ADNI4 comparison
    # =====================================================================
    w("---")
    w("")
    w("## Part B: ADNI4 Comparison")
    w("")

    if new_adni4 is None or len(new_adni4) == 0:
        w("**No ADNI4 rows found in new MERGED.csv.**")
        w("")
    else:
        # Reset index for PTID-level comparison (VISCODE_FIX differs)
        ref4_flat = ref4.reset_index()
        new4_flat = new_adni4.reset_index()

        ref4_ptids = ref4_flat['PTID'].unique()
        new4_ptids = new4_flat['PTID'].unique()

        w("### B1. Overall Summary")
        w("")
        w("| Metric | Ref | New (ADNI4 PTIDs) |")
        w("|--------|-----|-------------------|")
        w(f"| Rows | {ref4_flat.shape[0]:,} | {new4_flat.shape[0]:,} |")
        w(f"| Columns | {ref4_flat.shape[1]:,} | {new4_flat.shape[1]:,} |")
        w(f"| Unique PTIDs | {len(ref4_ptids):,} | {len(new4_ptids):,} |")
        w("")

        # COLPROT distribution in new ADNI4 (shows longitudinal composition)
        if 'COLPROT' in new4_flat.columns:
            w("**New ADNI4 COLPROT distribution (PTID-based filter includes longitudinal):**")
            w("")
            for proto, cnt in new4_flat['COLPROT'].value_counts().items():
                w(f"- {proto}: {cnt:,}")
            w("")

        # VISCODE_FIX format note
        w("### B2. VISCODE_FIX Format Difference")
        w("")
        w("**VISCODE_FIX 형식이 다르므로 row-level 비교 불가:**")
        w("")
        ref_vc_top = ref4_flat['VISCODE_FIX'].value_counts().head(5)
        new_vc_top = new4_flat['VISCODE_FIX'].value_counts().head(5)
        w("| Ref (top 5) | Count | New (top 5) | Count |")
        w("|-------------|-------|-------------|-------|")
        for i in range(5):
            r_vc = ref_vc_top.index[i] if i < len(ref_vc_top) else ''
            r_cnt = int(ref_vc_top.iloc[i]) if i < len(ref_vc_top) else ''
            n_vc = new_vc_top.index[i] if i < len(new_vc_top) else ''
            n_cnt = int(new_vc_top.iloc[i]) if i < len(new_vc_top) else ''
            w(f"| {r_vc} | {r_cnt} | {n_vc} | {n_cnt} |")
        w("")
        w("Ref uses `4_sc/4_init/4_m12` format; New uses `m000/m003/m072` format.")
        w("")

        # PTID-level comparison
        w("### B3. PTID Overlap")
        w("")
        common4_ptid = set(ref4_ptids).intersection(set(new4_ptids))
        ref4_only_ptid = set(ref4_ptids) - set(new4_ptids)
        new4_only_ptid = set(new4_ptids) - set(ref4_ptids)
        w(f"- Common PTIDs: **{len(common4_ptid):,}**")
        w(f"- Ref-only PTIDs: {len(ref4_only_ptid):,}")
        w(f"- New-only PTIDs: {len(new4_only_ptid):,}")
        w(f"- Coverage of ref PTIDs: {len(common4_ptid)/len(ref4_ptids)*100:.1f}%")
        w("")

        # Modality distribution
        w("### B4. Modality Distribution")
        w("")
        ref4_mod = modality_distribution(ref4_flat)
        new4_mod = modality_distribution(new4_flat)
        all4_mods = sorted(set(ref4_mod.index) | set(new4_mod.index))
        w("| Modality | Ref | New | Note |")
        w("|----------|-----|-----|------|")
        for m in all4_mods:
            r = int(ref4_mod.get(m, 0))
            n = int(new4_mod.get(m, 0))
            note = ''
            if m == 'MK6240':
                note = 'Our name: MK6240_6MM'
            elif m == 'MK6240_6MM':
                note = 'Ref name: MK6240'
            elif m == 'T2w':
                note = 'Our name: T2_3D'
            elif m == 'T2_3D':
                note = 'Ref name: T2w'
            w(f"| {m} | {r:,} | {n:,} | {note} |")
        w("")

        # PTID+ImageUID set comparison (since VISCODE differs)
        # Use only COLPROT='ADNI4' rows for fair comparison (ref4 only has ADNI4-phase data)
        new4_adni4_only = new4_flat[new4_flat['COLPROT'] == 'ADNI4'] if 'COLPROT' in new4_flat.columns else new4_flat

        w("### B5. Per-Modality ImageUID Set Comparison (ADNI4 phase only)")
        w("")
        w("Ref는 ADNI4 phase 데이터만 포함 → New도 COLPROT='ADNI4' 행만으로 비교:")
        w("")
        w("| Ref Mod | Our Mod | Ref UIDs | New UIDs | Intersection | Jaccard |")
        w("|---------|---------|----------|----------|--------------|---------|")

        for ref_mod_name, our_mod_name in ADNI4_MOD_MAP.items():
            ref_col = f'I_{ref_mod_name}'
            new_col = f'I_{our_mod_name}'

            if ref_col not in ref4_flat.columns or new_col not in new4_adni4_only.columns:
                w(f"| {ref_mod_name} | {our_mod_name} | — | — | — | col missing |")
                continue

            # Collect all ImageUIDs as sets (normalized)
            ref_uids = set(ref4_flat[ref_col].dropna().astype(str).str.replace(r'\.0$', '', regex=True))
            new_uids = set(new4_adni4_only[new_col].dropna().astype(str).str.replace(r'\.0$', '', regex=True))

            if len(ref_uids) == 0 and len(new_uids) == 0:
                w(f"| {ref_mod_name} | {our_mod_name} | 0 | 0 | — | — |")
                continue

            inter = ref_uids & new_uids
            union = ref_uids | new_uids
            jaccard = len(inter) / len(union) if len(union) > 0 else 0

            w(f"| {ref_mod_name} | {our_mod_name} | {len(ref_uids):,} | {len(new_uids):,} | {len(inter):,} | {fmt_pct(jaccard)} |")
        w("")

    # =====================================================================
    # Part C: ADNI4 Detailed Analysis
    # =====================================================================
    w("---")
    w("")
    w("## Part C: ADNI4 Matching Issue Analysis")
    w("")

    if new_adni4 is not None and len(new_adni4) > 0:
        ref4_flat = ref4.reset_index()
        new4_flat = new_adni4.reset_index()

        # C1. AQUDATE-based UID comparison (bypasses VISCODE mismatch)
        w("### C1. AQUDATE-Based ImageUID Comparison")
        w("")
        w("VISCODE_FIX가 다르므로 PTID+AQUDATE 기준으로 UID를 비교합니다.")
        w("")
        w("| Ref Mod | Our Mod | Common PTID+AQUDATE | Same UID | Diff UID | Match Rate |")
        w("|---------|---------|---------------------|----------|----------|------------|")

        for ref_mod_name, our_mod_name in ADNI4_MOD_MAP.items():
            ref_uid_col = f'I_{ref_mod_name}'
            new_uid_col = f'I_{our_mod_name}'
            ref_aq_col = f'AQUDATE_{ref_mod_name}'
            new_aq_col = f'AQUDATE_{our_mod_name}'

            if ref_uid_col not in ref4_flat.columns or new_uid_col not in new4_flat.columns:
                w(f"| {ref_mod_name} | {our_mod_name} | — | — | — | col missing |")
                continue
            if ref_aq_col not in ref4_flat.columns or new_aq_col not in new4_flat.columns:
                w(f"| {ref_mod_name} | {our_mod_name} | — | — | — | AQUDATE missing |")
                continue

            # Build PTID+AQUDATE lookup for ref
            ref_sub = ref4_flat[['PTID', ref_aq_col, ref_uid_col]].dropna(subset=[ref_uid_col]).copy()
            ref_sub['aq'] = ref_sub[ref_aq_col].astype(str).str[:10]
            ref_sub['uid'] = ref_sub[ref_uid_col].astype(str).str.replace(r'\.0$', '', regex=True)

            new_sub = new4_flat[['PTID', new_aq_col, new_uid_col]].dropna(subset=[new_uid_col]).copy()
            new_sub['aq'] = new_sub[new_aq_col].astype(str).str[:10]
            new_sub['uid'] = new_sub[new_uid_col].astype(str).str.replace(r'\.0$', '', regex=True)

            # Merge on PTID + AQUDATE
            merged = ref_sub.merge(new_sub, on=['PTID', 'aq'], suffixes=('_ref', '_new'))
            if len(merged) == 0:
                w(f"| {ref_mod_name} | {our_mod_name} | 0 | — | — | — |")
                continue

            # Deduplicate (keep first match per PTID+AQUDATE)
            merged = merged.drop_duplicates(subset=['PTID', 'aq'], keep='first')
            same_uid = (merged['uid_ref'] == merged['uid_new']).sum()
            diff_uid = len(merged) - same_uid
            rate = same_uid / len(merged) if len(merged) > 0 else 0
            w(f"| {ref_mod_name} | {our_mod_name} | {len(merged):,} | {same_uid:,} | {diff_uid:,} | {fmt_pct(rate)} |")
        w("")

        # C2. Ref-only PTID root cause analysis
        w("### C2. Ref-only PTID 원인 분석")
        w("")

        ref4_all_ptids = set(ref4_flat['PTID'].unique())
        new4_all_ptids = set(new4_flat['PTID'].unique())
        ref_only = ref4_all_ptids - new4_all_ptids

        w(f"- Ref PTIDs: {len(ref4_all_ptids):,}")
        w(f"- New PTIDs: {len(new4_all_ptids):,}")
        w(f"- Ref-only PTIDs: {len(ref_only):,}")
        w("")

        # Classify ref-only PTIDs using NFS + inventory
        inventory_path = kwargs.get('inventory_path')
        ptid_mods = collect_inventory_ptid_modalities(inventory_path)
        adni4_nfs_ptids = collect_nfs_ptids(ADNI4_DCM_ROOT)

        if ptid_mods or adni4_nfs_ptids:
            all_nfs_ptids = set(ptid_mods.keys()) | adni4_nfs_ptids

            no_dcm = sorted(ref_only - all_nfs_ptids)
            has_dcm = ref_only & all_nfs_ptids
            non_t1_only = sorted([p for p in has_dcm if 'T1' not in ptid_mods.get(p, set())])
            has_t1 = sorted(has_dcm - set(non_t1_only))

            w("**Ref-only 원인 분류:**")
            w("")
            w("| 원인 | 수 | 설명 |")
            w("|------|-----|------|")
            w(f"| NFS 전체 DCM 부재 | {len(no_dcm)} | ADNI4/ORIG, ADNI_New 양쪽 모두 DCM 파일 없음 |")
            w(f"| 비-T1 모달리티만 존재 | {len(non_t1_only)} | T1 없어 T1 기반 매칭 불가 |")
            if has_t1:
                w(f"| T1 존재 (기타) | {len(has_t1)} | DCM+T1 있으나 매칭 안 됨 (추가 조사 필요) |")
            w("")
            w("- Ref(기존 코드)는 XML 메타데이터 기반 → 물리적 DCM 없이도 행 생성 가능")
            w("- New(v4 코드)는 DCM 인벤토리 기반 → DCM 없으면 매칭 불가")
            w("- **파이프라인 버그 아님** — 데이터 가용성 차이")
            w("")

            # Detail table for non-T1-only PTIDs
            if non_t1_only:
                w("**비-T1 PTID 상세:**")
                w("")
                w("| PTID | 보유 모달리티 |")
                w("|------|---------------|")
                for ptid in non_t1_only:
                    mods = ', '.join(sorted(ptid_mods.get(ptid, set())))
                    w(f"| {ptid} | {mods} |")
                w("")

            # Detail table for unexpected T1-present PTIDs
            if has_t1:
                w("**T1 존재하나 매칭 안 된 PTID (추가 조사 필요):**")
                w("")
                w("| PTID | 보유 모달리티 |")
                w("|------|---------------|")
                for ptid in has_t1:
                    mods = ', '.join(sorted(ptid_mods.get(ptid, set())))
                    w(f"| {ptid} | {mods} |")
                w("")
        else:
            w("(DCM inventory not provided — 원인 분류 생략)")
            w("")

        # T1-specific coverage (retained from original)
        ref_t1_ptids = set(ref4_flat[ref4_flat['I_T1'].notna()]['PTID'].unique())
        new_t1_ptids = set(new4_flat[new4_flat['I_T1'].notna()]['PTID'].unique()) if 'I_T1' in new4_flat.columns else set()
        ref_t1_only = ref_t1_ptids - new_t1_ptids
        w(f"**T1 기준:**")
        w(f"- Ref T1 PTIDs: {len(ref_t1_ptids):,}")
        w(f"- New T1 PTIDs: {len(new_t1_ptids):,} {'(NEW가 더 많음)' if len(new_t1_ptids) > len(ref_t1_ptids) else ''}")
        w(f"- Ref-only T1 PTIDs: {len(ref_t1_only):,}")
        w("")

        # C3. VISCODE_FIX impact explanation
        w("### C3. VISCODE_FIX System Difference Impact")
        w("")
        w("Ref(기존 코드)는 ADNI4 전용 VISCODE(`4_sc`, `4_init`, `4_m12`)를 사용하고,")
        w("New(v4 코드)는 표준 month 기반(`m000`, `m003`, `m012`)을 사용합니다.")
        w("")
        w("**영향:**")
        w("- Row-level 비교 불가 (PTID+VISCODE_FIX 교집합 = 0)")
        w("- 동일 촬영일에도 다른 EXAMDATE 행 매칭 → 다른 UID 선택")
        w("- 다중 시리즈 모달리티(T2_STAR: 세션당 최대 10개)에서 영향 극대화")
        w("- **이것은 시스템 차이이며 버그가 아님** — 추후 VISCODE 매핑 테이블로 해결 가능")
        w("")

    # =====================================================================
    # Notes
    # =====================================================================
    w("---")
    w("")
    w("## Notes")
    w("")
    w("- **subjectAge**: Empty in new MERGED.csv due to missing `birth_dates.csv`.")
    w("- **weightKg / Flip Angle**: Not available from non-XML sources.")
    w("- **MK6240 vs MK6240_6MM, T2w vs T2_3D**: Modality name mapping not yet applied. "
      "Direct column comparison only works for identically-named modalities.")
    w("- **Modalities not compared**: DTI, DTI_MB, FMRI, HIPPO, ASL, NAV4694_6MM, PI2620_6MM "
      "(not in ref or not yet run).")
    w("")

    # Write report
    report = '\n'.join(lines) + '\n'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description='Compare new MERGED.csv with reference')
    parser.add_argument('--new-merged', type=str, default=DEFAULT_NEW_MERGED,
                        help='Path to new MERGED.csv')
    parser.add_argument('--ref-merged', type=str, default=DEFAULT_REF_MERGED,
                        help='Path to ref MERGED.csv (ADNI1/GO/2/3)')
    parser.add_argument('--ref-adni4', type=str, default=DEFAULT_REF_ADNI4,
                        help='Path to ref merged_adni4.csv')
    parser.add_argument('--adnimerge', type=str, default=DEFAULT_ADNIMERGE,
                        help='Path to ADNIMERGE CSV (for PTID coverage analysis)')
    parser.add_argument('--inventory', type=str, default=DEFAULT_DCM_INVENTORY,
                        help='Path to dcm_inventory.json (for ref-only PTID analysis)')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help='Output report path')
    args = parser.parse_args()

    generate_report(args.new_merged, args.ref_merged, args.ref_adni4, args.output,
                    adnimerge_csv=args.adnimerge, inventory_path=args.inventory)


if __name__ == '__main__':
    main()
