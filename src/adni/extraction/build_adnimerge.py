"""
build_adnimerge.py — ADNIMERGE2 .rda 데이터를 병합하여 ADNIMERGE CSV 생성

07_extract_adnimerge_csv.R (1043줄)의 전체 로직을 Python/pandas로 1:1 이식.

12 Steps:
 1. Load core datasets
 2. Create base frame (REGISTRY)
 3. Add demographics (PTDEMOG + APOERES + ADSL)
 4. Add diagnosis (DXSUM)
 5. Add cognitive tests (ADAS, MMSE, CDR, MOCA, FAQ, NEUROBAT, ECOG)
 6. Add CSF biomarkers (UPENNBIOMK_MASTER + ROCHE)
 7. Add MRI volumes (UCSDVOL + UCSFFSX51)
 8. Add PET data (FDG, Amyloid, TAU)
 9. Add plasma biomarkers
10. Calculate derived variables
11. Select & order final columns
12. Save to CSV
"""

import os
import re
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import pyreadr

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def load_rda(filepath: str, csv_fallback_dir: str = None) -> pd.DataFrame | None:
    """Load .rda file, return first DataFrame.

    If pyreadr fails, tries loading from CSV fallback (csv/tables/{name}.csv).
    """
    if not os.path.isfile(filepath):
        logger.warning('File not found: %s', filepath)
        return None
    try:
        result = pyreadr.read_r(filepath)
        if not result:
            raise ValueError('Empty .rda file')
        name = list(result.keys())[0]
        return result[name]
    except Exception as e:
        logger.warning('pyreadr failed for %s: %s, trying CSV fallback...', filepath, e)
        # CSV fallback: csv/tables/{basename}.csv
        basename = os.path.splitext(os.path.basename(filepath))[0]
        if csv_fallback_dir:
            csv_path = os.path.join(csv_fallback_dir, basename + '.csv')
        else:
            # (vendor/ADNIMERGE2/data/*.rda → data → ADNIMERGE2 → vendor → project root = 3 levels up)
            rda_dir = os.path.dirname(filepath)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(rda_dir)))
            csv_path = os.path.join(project_root, 'csv', 'tables', basename + '.csv')
        if os.path.isfile(csv_path):
            logger.info('  -> Loading from CSV: %s', csv_path)
            return pd.read_csv(csv_path, low_memory=False)
        logger.error('No CSV fallback found: %s', csv_path)
        return None


def standardize_viscode(viscode: pd.Series, viscode2: pd.Series = None) -> pd.Series:
    """Convert VISCODE2 to standard VISCODE.

    sc -> bl, scmri -> bl, v01 -> bl, nv -> bl
    ADNI4: 4_sc -> sc, 4_bl -> bl, 4_m12 -> m12, etc.
    """
    if viscode2 is not None:
        v = viscode2.where(viscode2.notna() & (viscode2 != ''), viscode)
    else:
        v = viscode.copy()

    v = v.str.replace(r'^scmri$', 'bl', regex=True)
    v = v.str.replace(r'^sc$', 'bl', regex=True)
    v = v.str.replace(r'^v01$', 'bl', regex=True)
    v = v.str.replace(r'^nv$', 'bl', regex=True)
    # ADNI4 VISCODE formats (when VISCODE2 is NaN, fallback to VISCODE)
    v = v.str.replace(r'^4_sc$', 'sc', regex=True)
    v = v.str.replace(r'^4_bl$', 'bl', regex=True)
    v = v.str.replace(r'^4_m(\d+)$', r'm\1', regex=True)
    return v


def convert_ecog_to_numeric(series: pd.Series) -> pd.Series:
    """Convert ECOG text responses to numeric: '1- Better/no change' -> 1."""
    if series.dtype == object:
        return pd.to_numeric(series.str.extract(r'^(\d+)', expand=False), errors='coerce')
    return pd.to_numeric(series, errors='coerce')


def first_non_na(group: pd.DataFrame, cols: list) -> pd.Series:
    """Return first non-NA value for each column in group."""
    result = {}
    for col in cols:
        vals = group[col].dropna()
        result[col] = vals.iloc[0] if len(vals) > 0 else np.nan
    return pd.Series(result)


def _group_first_non_na(df: pd.DataFrame, key_cols: list, value_cols: list) -> pd.DataFrame:
    """Group by key_cols, take first non-NA for each value_col."""
    return df.groupby(key_cols, sort=False).apply(
        lambda g: first_non_na(g, value_cols), include_groups=False
    ).reset_index()


# =============================================================================
# Valid VISCODE set
# =============================================================================

VALID_VISCODES = {'bl', 'sc'} | {
    f'm{str(m).zfill(2)}' for m in range(3, 169, 6)
} | {
    f'm{str(m).zfill(2)}' for m in [3, 6, 12, 18, 24, 30, 36, 42, 48, 54,
                                      60, 66, 72, 78, 84, 90, 96, 102, 108,
                                      114, 120, 126, 132, 138, 144, 150, 156,
                                      162, 168]
} | {
    # ADNI4 extended visits (m174 ~ m360, 6-month intervals)
    f'm{m}' for m in range(174, 361, 6)
}

# R script original: "bl","sc", m03~m168 (6-month intervals)
# Extended for ADNI4: m174~m360 (rollover subjects with 20+ year follow-up)
VALID_VISCODES_R = {'bl', 'sc',
    'm03', 'm06', 'm12', 'm18', 'm24', 'm30', 'm36',
    'm42', 'm48', 'm54', 'm60', 'm66', 'm72', 'm78',
    'm84', 'm90', 'm96', 'm102', 'm108', 'm114', 'm120',
    'm126', 'm132', 'm138', 'm144', 'm150', 'm156', 'm162', 'm168',
    'm174', 'm180', 'm186', 'm192', 'm198', 'm204', 'm210', 'm216',
    'm222', 'm228', 'm234', 'm240', 'm246', 'm252', 'm258', 'm264',
    'm270', 'm276', 'm282', 'm288', 'm294', 'm300', 'm306', 'm312',
    'm318', 'm324', 'm330', 'm336', 'm342', 'm348', 'm354', 'm360'}


# =============================================================================
# Main Build Function
# =============================================================================

def build_adnimerge(rda_dir: str, output_dir: str, date_str: str = None):
    """Build ADNIMERGE CSV from .rda files.

    Args:
        rda_dir: Path to ADNIMERGE2/data/ directory
        output_dir: Path to csv/ directory
        date_str: Date string for output filename (YYMMDD), defaults to today
    """
    if date_str is None:
        date_str = datetime.now().strftime('%y%m%d')

    def _load(name):
        return load_rda(os.path.join(rda_dir, name))

    # =========================================================================
    # Step 1: Load Core Datasets
    # =========================================================================
    logger.info('Step 1: Loading core datasets...')

    ADSL = _load('ADSL.rda')
    REGISTRY = _load('REGISTRY.rda')
    DXSUM = _load('DXSUM.rda')
    PTDEMOG = _load('PTDEMOG.rda')
    APOERES = _load('APOERES.rda')
    ARM = _load('ARM.rda')
    ADAS = _load('ADAS.rda')
    MMSE = _load('MMSE.rda')
    CDR = _load('CDR.rda')
    MOCA_df = _load('MOCA.rda')
    NEUROBAT = _load('NEUROBAT.rda')
    FAQ_df = _load('FAQ.rda')
    ECOGPT = _load('ECOGPT.rda')
    ECOGSP = _load('ECOGSP.rda')
    UPENNBIOMK = _load('UPENNBIOMK_MASTER.rda')
    UPENNBIOMK_ROCHE = _load('UPENNBIOMK_ROCHE_ELECSYS.rda')
    UCSDVOL = _load('UCSDVOL.rda')
    UCSFFSX = _load('UCSFFSX.rda')
    UPENNPLASMA = _load('UPENNPLASMA.rda')
    C2N_PLASMA = _load('C2N_PRECIVITYAD2_PLASMA.rda')
    UPENN_PLASMA_FQ = _load('UPENN_PLASMA_FUJIREBIO_QUANTERIX.rda')
    BLENNOW_NFL = _load('BLENNOWPLASMANFL.rda')
    UGOT_PTAU181 = _load('UGOTPTAU181.rda')

    logger.info('  - ADSL: %d subjects', len(ADSL))
    logger.info('  - REGISTRY: %d visits', len(REGISTRY))
    logger.info('  - DXSUM: %d records', len(DXSUM))

    # =========================================================================
    # Step 2: Create Base Frame
    # =========================================================================
    logger.info('Step 2: Creating base frame from REGISTRY...')

    base = REGISTRY[REGISTRY['RID'].notna() & REGISTRY['VISCODE'].notna()].copy()
    viscode2 = base['VISCODE2'] if 'VISCODE2' in base.columns else None
    # Save original VISCODE before standardization (sc/bl distinction needed)
    base['_VISCODE_ORIG'] = base['VISCODE'].copy()
    base['VISCODE'] = standardize_viscode(base['VISCODE'], viscode2)
    base['EXAMDATE'] = pd.to_datetime(base['EXAMDATE'], errors='coerce')

    # Filter to valid clinical visits
    base = base[base['VISCODE'].isin(VALID_VISCODES_R)].copy()

    cols_keep = ['RID', 'PTID', 'VISCODE', '_VISCODE_ORIG', 'EXAMDATE', 'COLPROT', 'ORIGPROT']
    if 'VISCODE2' in base.columns:
        cols_keep.append('VISCODE2')
    cols_keep = [c for c in cols_keep if c in base.columns]
    base = base[cols_keep].drop_duplicates()

    # Keep one record per RID-VISCODE.
    # For bl: prefer the original 'bl' visit over 'sc' (screening) — LONI behavior.
    # ADNI1: VISCODE itself is 'sc'/'bl'. ADNI2/GO: VISCODE='v01'/'v03', VISCODE2='sc'/'bl'.
    is_sc_v1 = base['_VISCODE_ORIG'].str.lower().isin(['sc', 'scmri'])
    is_sc_v2 = pd.Series(False, index=base.index)
    if 'VISCODE2' in base.columns:
        is_sc_v2 = base['VISCODE2'].str.lower().isin(['sc', 'scmri']).fillna(False)
    base['_is_sc'] = (is_sc_v1 | is_sc_v2).astype(int)
    base = base.sort_values(['_is_sc', 'EXAMDATE'])
    base = base.groupby(['RID', 'VISCODE'], sort=False).first().reset_index()
    base = base.drop(columns=['_is_sc', '_VISCODE_ORIG'] + (['VISCODE2'] if 'VISCODE2' in base.columns else []))
    base = base.sort_values(['RID', 'EXAMDATE']).reset_index(drop=True)

    # Ensure RID is integer
    base['RID'] = base['RID'].astype(int)

    logger.info('  - Base frame: %d visit records for %d subjects',
                len(base), base['RID'].nunique())

    # =========================================================================
    # Step 3: Add Demographics
    # =========================================================================
    logger.info('Step 3: Adding demographics...')

    # Demographics (one per subject)
    demog_cols = ['RID', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'PTDOB']
    demog_cols = [c for c in demog_cols if c in PTDEMOG.columns]
    demog = PTDEMOG[demog_cols].drop_duplicates()
    demog = demog.groupby('RID', sort=False).first().reset_index()

    # Standardize ethnicity
    def _std_eth(x):
        if pd.isna(x):
            return 'Unknown'
        x = str(x)
        if re.search(r'Not', x, re.IGNORECASE) and re.search(r'Hisp|Latino', x, re.IGNORECASE):
            return 'Not Hisp/Latino'
        if re.search(r'Hisp|Latino', x, re.IGNORECASE):
            return 'Hisp/Latino'
        return 'Unknown'

    def _std_race(x):
        if pd.isna(x):
            return x
        x = str(x)
        if re.search(r'White', x, re.IGNORECASE):
            return 'White'
        if re.search(r'Black|African', x, re.IGNORECASE):
            return 'Black'
        if re.search(r'Asian', x, re.IGNORECASE):
            return 'Asian'
        if re.search(r'More than', x, re.IGNORECASE):
            return 'More than one'
        return x

    demog['PTETHCAT'] = demog['PTETHCAT'].apply(_std_eth)
    demog['PTRACCAT'] = demog['PTRACCAT'].apply(_std_race)

    # APOE (one per subject)
    def _apoe4(genotype):
        if pd.isna(genotype):
            return np.nan
        g = str(genotype)
        if '4/4' in g:
            return 2
        if '4' in g:
            return 1
        return 0

    apoe = APOERES[['RID']].copy()
    apoe['APOE4'] = APOERES['GENOTYPE'].apply(_apoe4).astype('Int64')
    apoe = apoe.drop_duplicates().groupby('RID', sort=False).first().reset_index()

    # ADSL baseline characteristics
    adsl_rename = {
        'SUBJID': 'RID',
        'AGE': 'AGE_bl',
        'DX': 'DX_bl',
        'CDRSB': 'CDRSB_bl',
        'MMSCORE': 'MMSE_bl',
        'ADASTT11': 'ADAS11_bl',
        'ADASTT13': 'ADAS13_bl',
        'FAQTOTAL': 'FAQ_bl',
        'MOCA': 'MOCA_bl',
        'RAVLTIMM': 'RAVLT_immediate_bl',
        'RAVLTLRN': 'RAVLT_learning_bl',
        'RAVLTFG': 'RAVLT_forgetting_bl',
        'RAVLTFGP': 'RAVLT_perc_forgetting_bl',
        'DIGITSCR': 'DIGITSCOR_bl',
        'LDELTOTL': 'LDELTOTAL_BL',
        'TRABSCOR': 'TRABSCOR_bl',
        'MPACCDIGIT': 'mPACCdigit_bl',
        'MPACCTRAILSB': 'mPACCtrailsB_bl',
    }
    adsl_cols_needed = [c for c in adsl_rename.keys() if c in ADSL.columns]
    # Also keep APOE and SEX, EDUC, MARISTAT from ADSL (used for reference)
    adsl_bl = ADSL[adsl_cols_needed].rename(columns=adsl_rename).copy()
    adsl_bl['RID'] = pd.to_numeric(adsl_bl['RID'], errors='coerce').astype('Int64')

    # Merge demographics into base
    base = base.merge(demog, on='RID', how='left')
    base = base.merge(apoe, on='RID', how='left')
    base = base.merge(adsl_bl, on='RID', how='left')

    # Calculate EXAMDATE_bl and Years_bl
    # Use the enrollment/baseline visit (VISCODE='bl') EXAMDATE, not min(EXAMDATE).
    # This matches LONI R package behavior: enrollment visit date as baseline reference.
    bl_dates = base[base['VISCODE'] == 'bl'][['RID', 'EXAMDATE']].copy()
    bl_dates.columns = ['RID', 'EXAMDATE_bl']
    bl_dates = bl_dates.drop_duplicates('RID')
    # Fallback for subjects without a bl row: use earliest EXAMDATE
    all_rids = base[['RID']].drop_duplicates()
    bl_dates = all_rids.merge(bl_dates, on='RID', how='left')
    missing_bl = bl_dates['EXAMDATE_bl'].isna()
    if missing_bl.any():
        fallback = base.groupby('RID')['EXAMDATE'].min().reset_index()
        fallback.columns = ['RID', '_fallback_bl']
        bl_dates = bl_dates.merge(fallback, on='RID', how='left')
        bl_dates.loc[missing_bl, 'EXAMDATE_bl'] = bl_dates.loc[missing_bl, '_fallback_bl']
        bl_dates = bl_dates.drop(columns=['_fallback_bl'])
        logger.info('  - %d subjects used fallback EXAMDATE_bl (no bl visit)', missing_bl.sum())
    base = base.merge(bl_dates, on='RID', how='left')
    base['Years_bl'] = (base['EXAMDATE'] - base['EXAMDATE_bl']).dt.days / 365.25

    # AGE = baseline AGE for all visits (original ADNIMERGE behavior)
    base['AGE'] = base['AGE_bl']

    logger.info('  - Demographics merged')

    # =========================================================================
    # Step 4: Add Diagnosis
    # =========================================================================
    logger.info('Step 4: Adding diagnosis...')

    dx = DXSUM.copy()
    vis2 = dx['VISCODE2'] if 'VISCODE2' in dx.columns else None
    dx['VISCODE'] = standardize_viscode(dx['VISCODE'], vis2)

    # DIAGNOSIS -> DX
    def _map_dx(d):
        if pd.isna(d):
            return np.nan
        d = str(d)
        if d == 'CN':
            return 'CN'
        if d == 'MCI':
            return 'MCI'
        if d == 'Dementia':
            return 'Dementia'
        return np.nan

    dx['DX'] = dx['DIAGNOSIS'].apply(_map_dx) if 'DIAGNOSIS' in dx.columns else np.nan
    dx = dx[dx['DX'].notna()][['RID', 'VISCODE', 'DX']].drop_duplicates()
    dx = dx.groupby(['RID', 'VISCODE'], sort=False).first().reset_index()

    # Baseline DX from DXSUM (fallback for ADNI3/4)
    dx_bl = dx[dx['VISCODE'].isin(['bl', 'sc'])][['RID', 'DX']].copy()
    dx_bl = dx_bl.rename(columns={'DX': 'DX_bl_diag'})
    dx_bl = dx_bl.groupby('RID', sort=False).first().reset_index()

    base = base.merge(dx[['RID', 'VISCODE', 'DX']], on=['RID', 'VISCODE'], how='left')
    base = base.merge(dx_bl, on='RID', how='left')

    # Use DX_bl from ADSL if available, else from DXSUM
    base['DX_bl'] = base['DX_bl'].fillna(base['DX_bl_diag'])

    # Convert DEM/Dementia to AD
    base['DX_bl'] = base['DX_bl'].replace({'DEM': 'AD', 'Dementia': 'AD'})
    base = base.drop(columns=['DX_bl_diag'])

    # --- ARM.rda: Refine DX_bl for ADNI1/GO/2 with enrollment group ---
    # ARM.rda has fine-grained enrollment groups (EMCI/LMCI/SMC) for ADNI1/GO/2.
    # ADNI3/4 subjects are not in ARM.rda and keep ADSL/DXSUM-based CN/MCI/AD.
    if ARM is not None and 'ARM' in ARM.columns:
        def _arm_to_dx_bl(arm_val):
            """Parse ARM enrollment group prefix to DX_bl."""
            if pd.isna(arm_val):
                return np.nan
            a = str(arm_val)
            if a.startswith('NL'):
                return 'CN'
            if a.startswith('EMCI'):
                return 'EMCI'
            if a.startswith('LMCI'):
                return 'LMCI'
            if a.startswith('SMC'):
                return 'SMC'
            if a.startswith('MCI'):
                # ADNI1 MCI = LMCI in later nomenclature
                return 'LMCI'
            if a.startswith('AD'):
                return 'AD'
            return np.nan

        arm_df = ARM[ARM['ENROLLED'].notna()].copy()
        # Exclude screen failures and phantoms
        arm_df = arm_df[~arm_df['ENROLLED'].str.contains('phantom|screen failed',
                                                          case=False, na=False)]
        arm_df['DX_bl_arm'] = arm_df['ARM'].apply(_arm_to_dx_bl)
        arm_df = arm_df[arm_df['DX_bl_arm'].notna()][['RID', 'DX_bl_arm']].copy()
        arm_df['RID'] = pd.to_numeric(arm_df['RID'], errors='coerce').astype('Int64')
        # One DX_bl per RID (first enrollment)
        arm_df = arm_df.drop_duplicates('RID')

        # Apply ARM-based DX_bl where available
        base = base.merge(arm_df, on='RID', how='left')
        arm_available = base['DX_bl_arm'].notna()
        base.loc[arm_available, 'DX_bl'] = base.loc[arm_available, 'DX_bl_arm']
        base = base.drop(columns=['DX_bl_arm'])

        logger.info('  - ARM.rda DX_bl applied: %d subjects with EMCI/LMCI/SMC',
                     arm_available.groupby(base['RID']).first().sum())

    logger.info('  - Diagnosis added')

    # =========================================================================
    # Step 5: Add Cognitive Test Scores
    # =========================================================================
    logger.info('Step 5: Adding cognitive test scores...')

    # ADAS
    adas = ADAS.copy()
    vis2 = adas['VISCODE2'] if 'VISCODE2' in adas.columns else None
    adas['VISCODE'] = standardize_viscode(adas['VISCODE'], vis2)
    adas = adas.rename(columns={'TOTSCORE': 'ADAS11', 'TOTAL13': 'ADAS13'})
    adas = adas[adas['ADAS11'].notna() | adas['ADAS13'].notna()]
    adas = _group_first_non_na(adas, ['RID', 'VISCODE'], ['ADAS11', 'ADAS13'])

    # MMSE
    mmse = MMSE.copy()
    vis2 = mmse['VISCODE2'] if 'VISCODE2' in mmse.columns else None
    mmse['VISCODE'] = standardize_viscode(mmse['VISCODE'], vis2)
    mmse = mmse.rename(columns={'MMSCORE': 'MMSE'})
    mmse = mmse[mmse['MMSE'].notna()][['RID', 'VISCODE', 'MMSE']]
    mmse = mmse.groupby(['RID', 'VISCODE'], sort=False).first().reset_index()

    # CDR
    cdr = CDR.copy()
    vis2 = cdr['VISCODE2'] if 'VISCODE2' in cdr.columns else None
    cdr['VISCODE'] = standardize_viscode(cdr['VISCODE'], vis2)
    cdr_cols = ['RID', 'VISCODE', 'CDRSB']
    if 'CDGLOBAL' in cdr.columns:
        cdr_cols.append('CDGLOBAL')
    cdr = cdr[cdr['CDRSB'].notna()][cdr_cols]
    cdr = cdr.groupby(['RID', 'VISCODE'], sort=False).first().reset_index()

    # MOCA
    moca = MOCA_df.copy()
    vis2 = moca['VISCODE2'] if 'VISCODE2' in moca.columns else None
    moca['VISCODE'] = standardize_viscode(moca['VISCODE'], vis2)
    moca['MOCA'] = pd.to_numeric(moca['MOCA'], errors='coerce')
    moca = moca[moca['MOCA'].notna()][['RID', 'VISCODE', 'MOCA']]
    moca = _group_first_non_na(moca, ['RID', 'VISCODE'], ['MOCA'])

    # FAQ
    faq = FAQ_df.copy()
    vis2 = faq['VISCODE2'] if 'VISCODE2' in faq.columns else None
    faq['VISCODE'] = standardize_viscode(faq['VISCODE'], vis2)
    faq = faq.rename(columns={'FAQTOTAL': 'FAQ'})
    faq = faq[faq['FAQ'].notna()][['RID', 'VISCODE', 'FAQ']]
    faq = faq.groupby(['RID', 'VISCODE'], sort=False).first().reset_index()

    # NEUROBAT
    nb = NEUROBAT.copy()
    vis2 = nb['VISCODE2'] if 'VISCODE2' in nb.columns else None
    nb['VISCODE'] = standardize_viscode(nb['VISCODE'], vis2)

    for col in ['AVTOT1', 'AVTOT2', 'AVTOT3', 'AVTOT4', 'AVTOT5', 'AVDELTOT', 'AVDEL30MIN']:
        if col in nb.columns:
            nb[col] = pd.to_numeric(nb[col], errors='coerce')

    nb['RAVLT_immediate'] = nb[['AVTOT1', 'AVTOT2', 'AVTOT3', 'AVTOT4', 'AVTOT5']].sum(axis=1, min_count=5)
    nb['RAVLT_learning'] = nb['AVTOT5'] - nb['AVTOT1']
    nb['RAVLT_forgetting'] = nb['AVTOT5'] - nb['AVDEL30MIN']
    nb['RAVLT_perc_forgetting'] = np.where(
        nb['AVTOT5'] > 0,
        (nb['AVTOT5'] - nb['AVDEL30MIN']) / nb['AVTOT5'] * 100,
        np.nan
    )

    nb_cols = ['RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting',
               'RAVLT_perc_forgetting', 'LDELTOTAL', 'DIGITSCOR', 'TRABSCOR']
    nb_cols = [c for c in nb_cols if c in nb.columns]
    nb = _group_first_non_na(nb, ['RID', 'VISCODE'], nb_cols)

    # ECOG Patient
    ecogpt = ECOGPT.copy()
    vis2 = ecogpt['VISCODE2'] if 'VISCODE2' in ecogpt.columns else None
    ecogpt['VISCODE'] = standardize_viscode(ecogpt['VISCODE'], vis2)

    ecog_items_mem = [f'MEMORY{i}' for i in range(1, 9)]
    ecog_items_lang = [f'LANG{i}' for i in range(1, 10)]
    ecog_items_vis = [f'VISSPAT{i}' for i in range(1, 9)]
    ecog_items_plan = [f'PLAN{i}' for i in range(1, 6)]
    ecog_items_org = [f'ORGAN{i}' for i in range(1, 7)]
    ecog_items_div = [f'DIVATT{i}' for i in range(1, 5)]
    all_ecog_items = ecog_items_mem + ecog_items_lang + ecog_items_vis + ecog_items_plan + ecog_items_org + ecog_items_div

    for col in all_ecog_items:
        if col in ecogpt.columns:
            ecogpt[col] = convert_ecog_to_numeric(ecogpt[col])

    ecogpt['EcogPtMem'] = ecogpt[[c for c in ecog_items_mem if c in ecogpt.columns]].mean(axis=1, skipna=True)
    ecogpt['EcogPtLang'] = ecogpt[[c for c in ecog_items_lang if c in ecogpt.columns]].mean(axis=1, skipna=True)
    ecogpt['EcogPtVisspat'] = ecogpt[[c for c in ecog_items_vis if c in ecogpt.columns]].mean(axis=1, skipna=True)
    ecogpt['EcogPtPlan'] = ecogpt[[c for c in ecog_items_plan if c in ecogpt.columns]].mean(axis=1, skipna=True)
    ecogpt['EcogPtOrgan'] = ecogpt[[c for c in ecog_items_org if c in ecogpt.columns]].mean(axis=1, skipna=True)
    ecogpt['EcogPtDivatt'] = ecogpt[[c for c in ecog_items_div if c in ecogpt.columns]].mean(axis=1, skipna=True)

    pt_domain_cols = ['EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt']
    all_pt_items = [c for c in ecog_items_mem + ecog_items_lang + ecog_items_vis + ecog_items_plan + ecog_items_org + ecog_items_div if c in ecogpt.columns]
    ecogpt['EcogPtTotal'] = ecogpt[all_pt_items].mean(axis=1, skipna=True)

    ecogpt_out_cols = ['RID', 'VISCODE'] + pt_domain_cols + ['EcogPtTotal']
    ecogpt = ecogpt[ecogpt_out_cols].groupby(['RID', 'VISCODE'], sort=False).first().reset_index()

    # ECOG Study Partner
    ecogsp = ECOGSP.copy()
    vis2 = ecogsp['VISCODE2'] if 'VISCODE2' in ecogsp.columns else None
    ecogsp['VISCODE'] = standardize_viscode(ecogsp['VISCODE'], vis2)

    for col in all_ecog_items:
        if col in ecogsp.columns:
            ecogsp[col] = convert_ecog_to_numeric(ecogsp[col])

    ecogsp['EcogSPMem'] = ecogsp[[c for c in ecog_items_mem if c in ecogsp.columns]].mean(axis=1, skipna=True)
    ecogsp['EcogSPLang'] = ecogsp[[c for c in ecog_items_lang if c in ecogsp.columns]].mean(axis=1, skipna=True)
    ecogsp['EcogSPVisspat'] = ecogsp[[c for c in ecog_items_vis if c in ecogsp.columns]].mean(axis=1, skipna=True)
    ecogsp['EcogSPPlan'] = ecogsp[[c for c in ecog_items_plan if c in ecogsp.columns]].mean(axis=1, skipna=True)
    ecogsp['EcogSPOrgan'] = ecogsp[[c for c in ecog_items_org if c in ecogsp.columns]].mean(axis=1, skipna=True)
    ecogsp['EcogSPDivatt'] = ecogsp[[c for c in ecog_items_div if c in ecogsp.columns]].mean(axis=1, skipna=True)

    sp_domain_cols = ['EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt']
    all_sp_items = [c for c in ecog_items_mem + ecog_items_lang + ecog_items_vis + ecog_items_plan + ecog_items_org + ecog_items_div if c in ecogsp.columns]
    ecogsp['EcogSPTotal'] = ecogsp[all_sp_items].mean(axis=1, skipna=True)

    ecogsp_out_cols = ['RID', 'VISCODE'] + sp_domain_cols + ['EcogSPTotal']
    ecogsp = ecogsp[ecogsp_out_cols].groupby(['RID', 'VISCODE'], sort=False).first().reset_index()

    # Merge all cognitive tests
    for df_cog in [adas, mmse, cdr, moca, faq, nb, ecogpt, ecogsp]:
        base = base.merge(df_cog, on=['RID', 'VISCODE'], how='left')

    logger.info('  - Cognitive tests merged')

    # =========================================================================
    # Step 6: Add CSF Biomarkers
    # =========================================================================
    logger.info('Step 6: Adding CSF biomarkers...')

    # UPENNBIOMK_MASTER
    csf_master = UPENNBIOMK.copy()
    csf_master['VISCODE'] = standardize_viscode(csf_master['VISCODE'])
    csf_cols_m = ['RID', 'VISCODE']
    for c in ['ABETA', 'TAU', 'PTAU']:
        if c in csf_master.columns:
            csf_cols_m.append(c)
    csf_master = csf_master[csf_cols_m]
    csf_master = csf_master[csf_master[['ABETA', 'TAU', 'PTAU']].notna().any(axis=1)]

    # UPENNBIOMK_ROCHE_ELECSYS
    csf_roche = UPENNBIOMK_ROCHE.copy()
    vis2 = csf_roche['VISCODE2'] if 'VISCODE2' in csf_roche.columns else None
    csf_roche['VISCODE'] = standardize_viscode(
        csf_roche.get('VISCODE2', csf_roche.get('VISCODE', pd.Series(dtype=str))),
    )
    rename_roche = {}
    if 'ABETA42' in csf_roche.columns:
        rename_roche['ABETA42'] = 'ABETA'
    csf_roche = csf_roche.rename(columns=rename_roche)
    csf_cols_r = ['RID', 'VISCODE']
    for c in ['ABETA', 'TAU', 'PTAU']:
        if c in csf_roche.columns:
            csf_cols_r.append(c)
    csf_roche = csf_roche[csf_cols_r]
    csf_roche = csf_roche[csf_roche[['ABETA', 'TAU', 'PTAU']].notna().any(axis=1)]

    # Combine (prefer MASTER, fill with ROCHE)
    csf = pd.concat([csf_master, csf_roche], ignore_index=True)
    csf = _group_first_non_na(csf, ['RID', 'VISCODE'], ['ABETA', 'TAU', 'PTAU'])

    logger.info('  - Combined CSF biomarkers: %d records', len(csf))

    # Baseline CSF
    csf_bl = csf[csf['VISCODE'] == 'bl'][['RID', 'ABETA', 'TAU', 'PTAU']].copy()
    csf_bl = csf_bl.rename(columns={'ABETA': 'ABETA_bl', 'TAU': 'TAU_bl', 'PTAU': 'PTAU_bl'})
    csf_bl = csf_bl.groupby('RID', sort=False).first().reset_index()

    base = base.merge(csf, on=['RID', 'VISCODE'], how='left')
    base = base.merge(csf_bl, on='RID', how='left')

    logger.info('  - CSF biomarkers merged')

    # =========================================================================
    # Step 7: Add MRI Volumes
    # =========================================================================
    logger.info('Step 7: Adding MRI volumes...')

    # --- Primary source: UCSFFSX51 (FreeSurfer 5.1, highest coverage) ---
    UCSFFSX51 = _load('UCSFFSX51.rda')

    mri_fsx51 = UCSFFSX51.copy()
    vis2 = mri_fsx51['VISCODE2'] if 'VISCODE2' in mri_fsx51.columns else None
    mri_fsx51['VISCODE'] = standardize_viscode(mri_fsx51['VISCODE'], vis2)

    fsx_vol_map = {}
    if 'ST29SV' in mri_fsx51.columns and 'ST88SV' in mri_fsx51.columns:
        fsx_vol_map['Hippocampus'] = pd.to_numeric(mri_fsx51['ST29SV'], errors='coerce') + pd.to_numeric(mri_fsx51['ST88SV'], errors='coerce')
    if 'ST120SV' in mri_fsx51.columns:
        fsx_vol_map['WholeBrain'] = pd.to_numeric(mri_fsx51['ST120SV'], errors='coerce')
    if 'ST10CV' in mri_fsx51.columns:
        fsx_vol_map['ICV'] = pd.to_numeric(mri_fsx51['ST10CV'], errors='coerce')
    if 'ST32CV' in mri_fsx51.columns and 'ST91CV' in mri_fsx51.columns:
        fsx_vol_map['Entorhinal'] = pd.to_numeric(mri_fsx51['ST32CV'], errors='coerce') + pd.to_numeric(mri_fsx51['ST91CV'], errors='coerce')
    if 'ST34CV' in mri_fsx51.columns and 'ST93CV' in mri_fsx51.columns:
        fsx_vol_map['Fusiform'] = pd.to_numeric(mri_fsx51['ST34CV'], errors='coerce') + pd.to_numeric(mri_fsx51['ST93CV'], errors='coerce')
    if 'ST70SV' in mri_fsx51.columns and 'ST129CV' in mri_fsx51.columns:
        fsx_vol_map['MidTemp'] = pd.to_numeric(mri_fsx51['ST70SV'], errors='coerce') + pd.to_numeric(mri_fsx51['ST129CV'], errors='coerce')

    for k, v in fsx_vol_map.items():
        mri_fsx51[k] = v

    fsx_vol_cols = list(fsx_vol_map.keys())
    mri = mri_fsx51[['RID', 'VISCODE'] + fsx_vol_cols].copy()
    mri = mri.groupby(['RID', 'VISCODE'], sort=False).first().reset_index()

    logger.info('  - UCSFFSX51 (primary): %d records, %d subjects', len(mri), mri['RID'].nunique())

    # --- Ventricles only: from UCSDVOL (FreeSurfer에 lateral ventricle 컬럼 없음) ---
    # UCSDVOL과 UCSFFSX51은 거의 비중첩이므로 outer merge로 양쪽 모두 보존.
    # Ventricles 컬럼만 UCSDVOL에서 가져오고, 나머지 용적은 UCSFFSX51 전용.
    mri_ucsd = UCSDVOL.copy()
    mri_ucsd['VISCODE'] = standardize_viscode(mri_ucsd['VISCODE'])
    if 'VENTRICLES' in mri_ucsd.columns:
        ucsd_vent = mri_ucsd[mri_ucsd['VENTRICLES'].notna()][['RID', 'VISCODE', 'VENTRICLES']].copy()
        ucsd_vent = ucsd_vent.rename(columns={'VENTRICLES': 'Ventricles'})
        ucsd_vent = ucsd_vent.groupby(['RID', 'VISCODE'], sort=False).first().reset_index()
        mri = mri.merge(ucsd_vent, on=['RID', 'VISCODE'], how='outer')
        logger.info('  - UCSDVOL Ventricles: %d records', len(ucsd_vent))

    logger.info('  - MRI volumes combined: %d records', len(mri))

    mri_vol_cols = ['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV']
    mri_vol_cols = [c for c in mri_vol_cols if c in mri.columns]

    # MRI baseline
    mri_bl_cols = [c for c in mri_vol_cols if c in mri.columns]
    mri_bl = mri[mri['VISCODE'] == 'bl'][['RID'] + mri_bl_cols].copy()
    mri_bl = mri_bl.rename(columns={c: c + '_bl' for c in mri_bl_cols})
    mri_bl = mri_bl.groupby('RID', sort=False).first().reset_index()

    # FreeSurfer metadata
    fsx_meta = UCSFFSX.copy()
    fsx_meta['VISCODE'] = standardize_viscode(fsx_meta['VISCODE'])
    fsx_meta_cols = ['RID', 'VISCODE']
    if 'FLDSTRENG' in fsx_meta.columns:
        fsx_meta['FLDSTRENG'] = fsx_meta['FLDSTRENG'].apply(
            lambda x: '1.5 Tesla MRI' if x == 1.5 else ('3 Tesla MRI' if x == 3 else str(x)) if pd.notna(x) else x
        )
        fsx_meta_cols.append('FLDSTRENG')
    if 'VERSION' in fsx_meta.columns:
        fsx_meta = fsx_meta.rename(columns={'VERSION': 'FSVERSION'})
        fsx_meta_cols.append('FSVERSION')
    fsx_meta = fsx_meta[fsx_meta_cols].groupby(['RID', 'VISCODE'], sort=False).first().reset_index()

    fsx_meta_bl = fsx_meta[fsx_meta['VISCODE'] == 'bl'].drop(columns=['VISCODE']).copy()
    bl_rename = {}
    if 'FLDSTRENG' in fsx_meta_bl.columns:
        bl_rename['FLDSTRENG'] = 'FLDSTRENG_bl'
    if 'FSVERSION' in fsx_meta_bl.columns:
        bl_rename['FSVERSION'] = 'FSVERSION_bl'
    fsx_meta_bl = fsx_meta_bl.rename(columns=bl_rename)
    fsx_meta_bl = fsx_meta_bl.groupby('RID', sort=False).first().reset_index()

    base = base.merge(mri, on=['RID', 'VISCODE'], how='left')
    base = base.merge(mri_bl, on='RID', how='left')
    base = base.merge(fsx_meta, on=['RID', 'VISCODE'], how='left')
    base = base.merge(fsx_meta_bl, on='RID', how='left')

    logger.info('  - MRI volumes merged')

    # =========================================================================
    # Step 8: Add PET Data
    # =========================================================================
    logger.info('Step 8: Adding PET data...')

    # FDG PET
    UCBERKELEYFDG = _load('UCBERKELEYFDG_8mm.rda')
    fdg = UCBERKELEYFDG[UCBERKELEYFDG['ROINAME'] == 'MetaROI'].copy()
    vis2 = fdg['VISCODE2'] if 'VISCODE2' in fdg.columns else None
    fdg['VISCODE'] = standardize_viscode(fdg['VISCODE'], vis2)
    fdg['FDG'] = pd.to_numeric(fdg['MEAN'], errors='coerce')
    fdg = fdg[fdg['FDG'].notna()][['RID', 'VISCODE', 'FDG']]
    fdg = fdg.groupby(['RID', 'VISCODE'], sort=False).first().reset_index()

    fdg_bl = fdg[fdg['VISCODE'] == 'bl'][['RID', 'FDG']].copy()
    fdg_bl = fdg_bl.rename(columns={'FDG': 'FDG_bl'})
    fdg_bl = fdg_bl.groupby('RID', sort=False).first().reset_index()

    logger.info('  - FDG PET: %d records', len(fdg))

    # Amyloid PET
    UCBERKELEY_AMY = _load('UCBERKELEY_AMY_6MM.rda')
    amy = UCBERKELEY_AMY.copy()
    vis2 = amy['VISCODE2'] if 'VISCODE2' in amy.columns else None
    amy['VISCODE'] = standardize_viscode(amy['VISCODE'], vis2)
    amy['AV45'] = pd.to_numeric(amy.get('SUMMARY_SUVR', pd.Series(dtype=float)), errors='coerce')
    amy_cols = ['RID', 'VISCODE', 'AV45']
    if 'LONIUID' in amy.columns:
        amy['AMY_LONIUID'] = amy['LONIUID']
        amy_cols.append('AMY_LONIUID')
    amy = amy[amy['AV45'].notna()][amy_cols]
    amy = amy.groupby(['RID', 'VISCODE'], sort=False).first().reset_index()

    amy_bl = amy[amy['VISCODE'] == 'bl'][['RID', 'AV45']].copy()
    amy_bl = amy_bl.rename(columns={'AV45': 'AV45_bl'})
    amy_bl = amy_bl.groupby('RID', sort=False).first().reset_index()

    logger.info('  - Amyloid PET: %d records', len(amy))

    # TAU PET
    UCBERKELEY_TAU = _load('UCBERKELEY_TAU_6MM.rda')
    tau_pet = UCBERKELEY_TAU.copy()
    vis2 = tau_pet['VISCODE2'] if 'VISCODE2' in tau_pet.columns else None
    tau_pet['VISCODE'] = standardize_viscode(tau_pet['VISCODE'], vis2)
    tau_cols = ['RID', 'VISCODE']
    if 'LONIUID' in tau_pet.columns:
        tau_pet['TAU_LONIUID'] = tau_pet['LONIUID']
        tau_cols.append('TAU_LONIUID')
    if 'META_TEMPORAL_SUVR' in tau_pet.columns:
        tau_pet['TAU_META_TEMPORAL_SUVR'] = pd.to_numeric(tau_pet['META_TEMPORAL_SUVR'], errors='coerce')
        tau_cols.append('TAU_META_TEMPORAL_SUVR')
    tau_pet = tau_pet[tau_pet.get('TAU_META_TEMPORAL_SUVR', pd.Series(dtype=float)).notna()][tau_cols]
    tau_pet = tau_pet.groupby(['RID', 'VISCODE'], sort=False).first().reset_index()

    # Merge PET
    base = base.merge(fdg, on=['RID', 'VISCODE'], how='left')
    base = base.merge(fdg_bl, on='RID', how='left')
    base = base.merge(amy, on=['RID', 'VISCODE'], how='left')
    base = base.merge(amy_bl, on='RID', how='left')
    base = base.merge(tau_pet, on=['RID', 'VISCODE'], how='left')

    # Placeholder columns
    for col in ['PIB', 'PIB_bl', 'FBB', 'FBB_bl']:
        base[col] = np.nan

    logger.info('  - PET data merged')

    # =========================================================================
    # Step 9: Add Plasma Biomarkers
    # =========================================================================
    logger.info('Step 9: Adding plasma biomarkers...')

    # UPenn Plasma
    p_upenn = UPENNPLASMA.copy()
    p_upenn['VISCODE'] = standardize_viscode(p_upenn['VISCODE'])
    rename_upenn = {}
    if 'AB40' in p_upenn.columns:
        rename_upenn['AB40'] = 'PLASMA_AB40_UPENN'
    if 'AB42' in p_upenn.columns:
        rename_upenn['AB42'] = 'PLASMA_AB42_UPENN'
    p_upenn = p_upenn.rename(columns=rename_upenn)
    p_upenn_cols = ['RID', 'VISCODE'] + list(rename_upenn.values())
    p_upenn = p_upenn[p_upenn_cols].groupby(['RID', 'VISCODE'], sort=False).first().reset_index()

    # C2N PrecivityAD2
    p_c2n = C2N_PLASMA.copy()
    vis2 = p_c2n['VISCODE2'] if 'VISCODE2' in p_c2n.columns else None
    p_c2n['VISCODE'] = standardize_viscode(p_c2n['VISCODE'], vis2)
    rename_c2n = {
        'pT217_C2N': 'PLASMA_PTAU217_C2N',
        'AB42_C2N': 'PLASMA_AB42_C2N',
        'AB40_C2N': 'PLASMA_AB40_C2N',
        'AB42_AB40_C2N': 'PLASMA_AB42_40_RATIO_C2N',
        'APS2_C2N': 'PLASMA_APS2_C2N',
    }
    rename_c2n = {k: v for k, v in rename_c2n.items() if k in p_c2n.columns}
    p_c2n = p_c2n.rename(columns=rename_c2n)
    p_c2n_cols = ['RID', 'VISCODE'] + list(rename_c2n.values())
    p_c2n = p_c2n[p_c2n_cols].groupby(['RID', 'VISCODE'], sort=False).first().reset_index()

    # UPenn Fujirebio/Quanterix
    p_fq = UPENN_PLASMA_FQ.copy()
    vis2 = p_fq['VISCODE2'] if 'VISCODE2' in p_fq.columns else None
    p_fq['VISCODE'] = standardize_viscode(p_fq['VISCODE'], vis2)
    rename_fq = {
        'pT217_F': 'PLASMA_PTAU217_F',
        'AB42_F': 'PLASMA_AB42_F',
        'AB40_F': 'PLASMA_AB40_F',
        'AB42_AB40_F': 'PLASMA_AB42_40_RATIO_F',
        'NfL_Q': 'PLASMA_NFL',
        'GFAP_Q': 'PLASMA_GFAP',
    }
    rename_fq = {k: v for k, v in rename_fq.items() if k in p_fq.columns}
    p_fq = p_fq.rename(columns=rename_fq)
    p_fq_cols = ['RID', 'VISCODE'] + list(rename_fq.values())
    p_fq = p_fq[p_fq_cols].groupby(['RID', 'VISCODE'], sort=False).first().reset_index()

    # Blennow NFL
    p_nfl = BLENNOW_NFL.copy()
    vis2 = p_nfl['VISCODE2'] if 'VISCODE2' in p_nfl.columns else None
    p_nfl['VISCODE'] = standardize_viscode(p_nfl['VISCODE'], vis2)
    rename_nfl = {}
    if 'PLASMA_NFL' in p_nfl.columns:
        rename_nfl['PLASMA_NFL'] = 'PLASMA_NFL_BLENNOW'
    p_nfl = p_nfl.rename(columns=rename_nfl)
    p_nfl_cols = ['RID', 'VISCODE'] + list(rename_nfl.values())
    p_nfl = p_nfl[p_nfl_cols].groupby(['RID', 'VISCODE'], sort=False).first().reset_index()

    # UGOT pTau181
    p_ptau = UGOT_PTAU181.copy()
    vis2 = p_ptau['VISCODE2'] if 'VISCODE2' in p_ptau.columns else None
    p_ptau['VISCODE'] = standardize_viscode(p_ptau['VISCODE'], vis2)
    rename_ptau = {}
    if 'PLASMAPTAU181' in p_ptau.columns:
        rename_ptau['PLASMAPTAU181'] = 'PLASMA_PTAU181'
    p_ptau = p_ptau.rename(columns=rename_ptau)
    p_ptau_cols = ['RID', 'VISCODE'] + list(rename_ptau.values())
    p_ptau = p_ptau[p_ptau_cols].groupby(['RID', 'VISCODE'], sort=False).first().reset_index()

    # Merge all plasma
    for df_p in [p_upenn, p_c2n, p_fq, p_nfl, p_ptau]:
        base = base.merge(df_p, on=['RID', 'VISCODE'], how='left')

    logger.info('  - Plasma biomarkers merged')

    # =========================================================================
    # Step 10: Calculate Derived Variables
    # =========================================================================
    logger.info('Step 10: Calculating derived variables...')

    # Baseline cognitive scores (calculate from visit data if ADSL doesn't have them)
    cog_bl_cols = ['ADAS11', 'ADAS13', 'MMSE', 'CDRSB', 'FAQ', 'MOCA',
                   'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting',
                   'RAVLT_perc_forgetting', 'LDELTOTAL', 'DIGITSCOR', 'TRABSCOR',
                   'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan',
                   'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal',
                   'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 'EcogSPPlan',
                   'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal']
    cog_bl_cols = [c for c in cog_bl_cols if c in base.columns]

    cog_bl_df = base[base['VISCODE'].isin(['bl', 'sc'])][['RID'] + cog_bl_cols].copy()
    cog_bl_df = cog_bl_df.groupby('RID', sort=False).first().reset_index()
    cog_bl_df = cog_bl_df.rename(columns={c: c + '_bl_calc' for c in cog_bl_cols})

    base = base.merge(cog_bl_df, on='RID', how='left')

    # Fill baseline values: coalesce ADSL with calculated
    coalesce_pairs = [
        ('ADAS11_bl', 'ADAS11_bl_calc'),
        ('ADAS13_bl', 'ADAS13_bl_calc'),
        ('MMSE_bl', 'MMSE_bl_calc'),
        ('CDRSB_bl', 'CDRSB_bl_calc'),
        ('FAQ_bl', 'FAQ_bl_calc'),
        ('MOCA_bl', 'MOCA_bl_calc'),
        ('RAVLT_immediate_bl', 'RAVLT_immediate_bl_calc'),
        ('RAVLT_learning_bl', 'RAVLT_learning_bl_calc'),
        ('RAVLT_forgetting_bl', 'RAVLT_forgetting_bl_calc'),
        ('RAVLT_perc_forgetting_bl', 'RAVLT_perc_forgetting_bl_calc'),
        ('LDELTOTAL_BL', 'LDELTOTAL_bl_calc'),
        ('DIGITSCOR_bl', 'DIGITSCOR_bl_calc'),
        ('TRABSCOR_bl', 'TRABSCOR_bl_calc'),
    ]
    for bl_col, calc_col in coalesce_pairs:
        if bl_col in base.columns and calc_col in base.columns:
            base[bl_col] = base[bl_col].fillna(base[calc_col])

    # ECOG baselines (always from calculated)
    ecog_bl_pairs = [
        ('EcogPtMem_bl', 'EcogPtMem_bl_calc'),
        ('EcogPtLang_bl', 'EcogPtLang_bl_calc'),
        ('EcogPtVisspat_bl', 'EcogPtVisspat_bl_calc'),
        ('EcogPtPlan_bl', 'EcogPtPlan_bl_calc'),
        ('EcogPtOrgan_bl', 'EcogPtOrgan_bl_calc'),
        ('EcogPtDivatt_bl', 'EcogPtDivatt_bl_calc'),
        ('EcogPtTotal_bl', 'EcogPtTotal_bl_calc'),
        ('EcogSPMem_bl', 'EcogSPMem_bl_calc'),
        ('EcogSPLang_bl', 'EcogSPLang_bl_calc'),
        ('EcogSPVisspat_bl', 'EcogSPVisspat_bl_calc'),
        ('EcogSPPlan_bl', 'EcogSPPlan_bl_calc'),
        ('EcogSPOrgan_bl', 'EcogSPOrgan_bl_calc'),
        ('EcogSPDivatt_bl', 'EcogSPDivatt_bl_calc'),
        ('EcogSPTotal_bl', 'EcogSPTotal_bl_calc'),
    ]
    for bl_col, calc_col in ecog_bl_pairs:
        if calc_col in base.columns:
            base[bl_col] = base[calc_col]

    # Drop _bl_calc columns
    calc_cols = [c for c in base.columns if c.endswith('_bl_calc')]
    base = base.drop(columns=calc_cols)

    # Month and M
    base['Month_bl'] = 0
    base['Month'] = (base['Years_bl'] * 12).round().astype('Int64')
    base['M'] = base['Month']

    # SITE from PTID
    base['SITE'] = base['PTID'].str.extract(r'^(\d+)_S_').astype('Int64')

    # Timestamp
    base['update_stamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ADASQ4 (NA placeholder)
    base['ADASQ4'] = np.nan
    base['ADASQ4_bl'] = np.nan

    # mPACC (NA placeholder)
    base['mPACCdigit'] = np.nan
    base['mPACCtrailsB'] = np.nan

    logger.info('  - Derived variables calculated')

    # =========================================================================
    # Step 11: Select & Order Final Columns
    # =========================================================================
    logger.info('Step 11: Selecting and ordering final columns...')

    final_cols = [
        # Identifiers
        'RID', 'COLPROT', 'ORIGPROT', 'PTID', 'SITE', 'VISCODE', 'EXAMDATE',
        # Baseline diagnosis
        'DX_bl',
        # Demographics
        'AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'PTMARRY',
        # APOE
        'APOE4',
        # PET
        'FDG', 'PIB', 'AV45', 'FBB',
        # CSF biomarkers
        'ABETA', 'TAU', 'PTAU',
        # Cognitive tests
        'CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'MMSE',
        'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting',
        'LDELTOTAL', 'DIGITSCOR', 'TRABSCOR', 'FAQ', 'MOCA',
        # ECOG Patient
        'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal',
        # ECOG Study Partner
        'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal',
        # MRI metadata
        'FLDSTRENG', 'FSVERSION',
        # MRI volumes
        'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV',
        # Current diagnosis
        'DX',
        # mPACC
        'mPACCdigit', 'mPACCtrailsB',
        # Baseline examdate and mPACC baseline
        'EXAMDATE_bl',
        'mPACCdigit_bl', 'mPACCtrailsB_bl',
        # Baseline cognitive
        'CDRSB_bl', 'ADAS11_bl', 'ADAS13_bl', 'ADASQ4_bl', 'MMSE_bl',
        'RAVLT_immediate_bl', 'RAVLT_learning_bl', 'RAVLT_forgetting_bl', 'RAVLT_perc_forgetting_bl',
        'LDELTOTAL_BL', 'DIGITSCOR_bl', 'TRABSCOR_bl', 'FAQ_bl',
        # Baseline MRI metadata
        'FLDSTRENG_bl', 'FSVERSION_bl',
        # Baseline MRI volumes
        'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl', 'ICV_bl',
        # Baseline MOCA
        'MOCA_bl',
        # Baseline ECOG Patient
        'EcogPtMem_bl', 'EcogPtLang_bl', 'EcogPtVisspat_bl', 'EcogPtPlan_bl', 'EcogPtOrgan_bl', 'EcogPtDivatt_bl', 'EcogPtTotal_bl',
        # Baseline ECOG Study Partner
        'EcogSPMem_bl', 'EcogSPLang_bl', 'EcogSPVisspat_bl', 'EcogSPPlan_bl', 'EcogSPOrgan_bl', 'EcogSPDivatt_bl', 'EcogSPTotal_bl',
        # Baseline CSF
        'ABETA_bl', 'TAU_bl', 'PTAU_bl',
        # Baseline PET
        'FDG_bl', 'PIB_bl', 'AV45_bl', 'FBB_bl',
        # Time variables
        'Years_bl', 'Month_bl', 'Month', 'M',
        # Timestamp
        'update_stamp',
        # PET identifiers
        'AMY_LONIUID',
        'TAU_LONIUID', 'TAU_META_TEMPORAL_SUVR',
        # Plasma biomarkers
        'PLASMA_AB40_UPENN', 'PLASMA_AB42_UPENN',
        'PLASMA_PTAU217_C2N', 'PLASMA_AB42_C2N', 'PLASMA_AB40_C2N', 'PLASMA_AB42_40_RATIO_C2N', 'PLASMA_APS2_C2N',
        'PLASMA_PTAU217_F', 'PLASMA_AB42_F', 'PLASMA_AB40_F', 'PLASMA_AB42_40_RATIO_F',
        'PLASMA_NFL', 'PLASMA_GFAP',
        'PLASMA_NFL_BLENNOW',
        'PLASMA_PTAU181',
    ]

    # Add missing columns as NA
    for col in final_cols:
        if col not in base.columns:
            base[col] = np.nan

    adnimerge = base[final_cols].copy()

    logger.info('  - Final columns selected: %d columns', len(adnimerge.columns))

    # Sort by PTID, then VISCODE (natural order)
    def _viscode_num(v):
        if v == 'bl':
            return 0
        m = re.match(r'^m(\d+)$', str(v))
        if m:
            return int(m.group(1))
        return 999

    adnimerge['_viscode_num'] = adnimerge['VISCODE'].apply(_viscode_num)
    adnimerge = adnimerge.sort_values(['PTID', '_viscode_num']).drop(columns=['_viscode_num'])
    adnimerge = adnimerge.reset_index(drop=True)

    logger.info('  - Sorted by PTID, VISCODE')

    # =========================================================================
    # Step 12: Save to CSV
    # =========================================================================
    logger.info('Step 12: Saving to CSV...')

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'ADNIMERGE_{date_str}.csv')
    adnimerge.to_csv(output_file, index=False, na_rep='')

    logger.info('  - Saved to: %s', output_file)

    # Summary
    logger.info('=' * 60)
    logger.info('SUMMARY')
    logger.info('=' * 60)
    logger.info('Output file: %s', output_file)
    logger.info('Total rows: %d', len(adnimerge))
    logger.info('Total columns: %d', len(adnimerge.columns))
    logger.info('Unique subjects (RID): %d', adnimerge['RID'].nunique())
    logger.info('Unique PTIDs: %d', adnimerge['PTID'].nunique())

    # Visit distribution
    vc = adnimerge['VISCODE'].value_counts().sort_index()
    logger.info('Visit distribution:\n%s', vc.to_string())

    # Diagnosis distribution
    dx_dist = adnimerge['DX'].value_counts(dropna=False)
    logger.info('Diagnosis distribution:\n%s', dx_dist.to_string())

    return adnimerge


# =============================================================================
# UCBerkeley PET CSV builders
# =============================================================================

def build_ucberkeley_fdg(rda_dir: str, output_dir: str, date_str: str = None):
    """Build UCBERKELEYFDG_8MM CSV from .rda files."""
    if date_str is None:
        date_str = datetime.now().strftime('%y%m%d')

    logger.info('Building UCBERKELEYFDG_8MM CSV...')

    FDG = load_rda(os.path.join(rda_dir, 'UCBERKELEYFDG_8mm.rda'))
    REGISTRY = load_rda(os.path.join(rda_dir, 'REGISTRY.rda'))

    if FDG is None or REGISTRY is None:
        logger.error('Could not load UCBERKELEYFDG_8mm or REGISTRY')
        return None

    logger.info('  - UCBERKELEYFDG_8mm: %d records', len(FDG))

    # VISCODE standardization
    if 'VISCODE2' in FDG.columns:
        FDG['VISCODE'] = standardize_viscode(FDG['VISCODE'], FDG['VISCODE2'])
    elif 'VISCODE' in FDG.columns:
        FDG['VISCODE'] = standardize_viscode(FDG['VISCODE'])

    # RID -> PTID mapping (FDG has no PTID)
    rid_ptid = REGISTRY[['RID', 'PTID']].drop_duplicates()
    fdg_clean = FDG.drop(columns=['PTID'], errors='ignore')
    fdg_export = fdg_clean.merge(rid_ptid, on='RID', how='left')

    # Column ordering
    key_cols = ['RID', 'PTID', 'VISCODE', 'EXAMDATE', 'ROINAME', 'MEAN', 'MAX', 'STDEV', 'TOTVOX', 'ORIGPROT']
    key_cols = [c for c in key_cols if c in fdg_export.columns]
    other_cols = [c for c in fdg_export.columns if c not in key_cols]
    fdg_export = fdg_export[key_cols + other_cols]

    fdg_export = fdg_export.sort_values(['RID', 'VISCODE']).reset_index(drop=True)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'UCBERKELEYFDG_8MM_{date_str}.csv')
    fdg_export.to_csv(output_file, index=False, na_rep='')

    logger.info('  - UCBERKELEYFDG_8MM saved: %s (%d rows, %d cols)',
                output_file, len(fdg_export), len(fdg_export.columns))
    return fdg_export


def build_ucberkeley_amy(rda_dir: str, output_dir: str, date_str: str = None):
    """Build UCBERKELEY_AMY_6MM CSV from .rda files."""
    if date_str is None:
        date_str = datetime.now().strftime('%y%m%d')

    logger.info('Building UCBERKELEY_AMY_6MM CSV...')

    AMY = load_rda(os.path.join(rda_dir, 'UCBERKELEY_AMY_6MM.rda'))

    if AMY is None:
        logger.error('Could not load UCBERKELEY_AMY_6MM')
        return None

    logger.info('  - UCBERKELEY_AMY_6MM: %d records', len(AMY))

    # VISCODE standardization
    if 'VISCODE2' in AMY.columns:
        AMY['VISCODE'] = standardize_viscode(AMY['VISCODE'], AMY['VISCODE2'])
    elif 'VISCODE' in AMY.columns:
        AMY['VISCODE'] = standardize_viscode(AMY['VISCODE'])

    # Column ordering
    key_cols = ['RID', 'PTID', 'VISCODE', 'SCANDATE', 'ORIGPROT', 'LONIUID', 'TRACER',
                'AMYLOID_STATUS', 'CENTILOIDS', 'SUMMARY_SUVR', 'SUMMARY_VOLUME']
    key_cols = [c for c in key_cols if c in AMY.columns]
    other_cols = [c for c in AMY.columns if c not in key_cols]
    amy_export = AMY[key_cols + other_cols]

    amy_export = amy_export.sort_values(['RID', 'VISCODE']).reset_index(drop=True)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'UCBERKELEY_AMY_6MM_{date_str}.csv')
    amy_export.to_csv(output_file, index=False, na_rep='')

    logger.info('  - UCBERKELEY_AMY_6MM saved: %s (%d rows, %d cols)',
                output_file, len(amy_export), len(amy_export.columns))
    return amy_export


def build_ucberkeley_tau(rda_dir: str, output_dir: str, date_str: str = None):
    """Build UCBERKELEY_TAU_6MM CSV from .rda files."""
    if date_str is None:
        date_str = datetime.now().strftime('%y%m%d')

    logger.info('Building UCBERKELEY_TAU_6MM CSV...')

    TAU = load_rda(os.path.join(rda_dir, 'UCBERKELEY_TAU_6MM.rda'))

    if TAU is None:
        logger.error('Could not load UCBERKELEY_TAU_6MM')
        return None

    logger.info('  - UCBERKELEY_TAU_6MM: %d records', len(TAU))

    # VISCODE standardization
    if 'VISCODE2' in TAU.columns:
        TAU['VISCODE'] = standardize_viscode(TAU['VISCODE'], TAU['VISCODE2'])
    elif 'VISCODE' in TAU.columns:
        TAU['VISCODE'] = standardize_viscode(TAU['VISCODE'])

    # Column ordering
    key_cols = ['RID', 'PTID', 'VISCODE', 'SCANDATE', 'ORIGPROT', 'LONIUID', 'TRACER',
                'META_TEMPORAL_SUVR']
    key_cols = [c for c in key_cols if c in TAU.columns]
    other_cols = [c for c in TAU.columns if c not in key_cols]
    tau_export = TAU[key_cols + other_cols]

    tau_export = tau_export.sort_values(['RID', 'VISCODE']).reset_index(drop=True)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'UCBERKELEY_TAU_6MM_{date_str}.csv')
    tau_export.to_csv(output_file, index=False, na_rep='')

    logger.info('  - UCBERKELEY_TAU_6MM saved: %s (%d rows, %d cols)',
                output_file, len(tau_export), len(tau_export.columns))
    return tau_export


def build_ucberkeley_taupvc(rda_dir: str, output_dir: str, date_str: str = None):
    """Build UCBERKELEY_TAUPVC_6MM CSV from .rda files."""
    if date_str is None:
        date_str = datetime.now().strftime('%y%m%d')

    logger.info('Building UCBERKELEY_TAUPVC_6MM CSV...')

    TAUPVC = load_rda(os.path.join(rda_dir, 'UCBERKELEY_TAUPVC_6MM.rda'))

    if TAUPVC is None:
        logger.error('Could not load UCBERKELEY_TAUPVC_6MM')
        return None

    logger.info('  - UCBERKELEY_TAUPVC_6MM: %d records', len(TAUPVC))

    # VISCODE standardization
    if 'VISCODE2' in TAUPVC.columns:
        TAUPVC['VISCODE'] = standardize_viscode(TAUPVC['VISCODE'], TAUPVC['VISCODE2'])
    elif 'VISCODE' in TAUPVC.columns:
        TAUPVC['VISCODE'] = standardize_viscode(TAUPVC['VISCODE'])

    # Column ordering
    key_cols = ['RID', 'PTID', 'VISCODE', 'SCANDATE', 'ORIGPROT', 'LONIUID', 'TRACER',
                'META_TEMPORAL_SUVR']
    key_cols = [c for c in key_cols if c in TAUPVC.columns]
    other_cols = [c for c in TAUPVC.columns if c not in key_cols]
    taupvc_export = TAUPVC[key_cols + other_cols]

    taupvc_export = taupvc_export.sort_values(['RID', 'VISCODE']).reset_index(drop=True)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'UCBERKELEY_TAUPVC_6MM_{date_str}.csv')
    taupvc_export.to_csv(output_file, index=False, na_rep='')

    logger.info('  - UCBERKELEY_TAUPVC_6MM saved: %s (%d rows, %d cols)',
                output_file, len(taupvc_export), len(taupvc_export.columns))
    return taupvc_export


def build_all_ucberkeley(rda_dir: str, output_dir: str, date_str: str = None):
    """Build all 4 UCBerkeley PET CSVs."""
    if date_str is None:
        date_str = datetime.now().strftime('%y%m%d')

    results = {}
    results['FDG'] = build_ucberkeley_fdg(rda_dir, output_dir, date_str)
    results['AMY'] = build_ucberkeley_amy(rda_dir, output_dir, date_str)
    results['TAU'] = build_ucberkeley_tau(rda_dir, output_dir, date_str)
    results['TAUPVC'] = build_ucberkeley_taupvc(rda_dir, output_dir, date_str)

    built = sum(1 for v in results.values() if v is not None)
    logger.info('UCBerkeley PET CSVs: %d/4 built', built)
    return results


# =============================================================================
# birth_dates.csv 생성
# =============================================================================

def build_birth_dates(rda_dir: str, output_dir: str):
    """PTDEMOG PTDOB (MM/YYYY) → birth_dates.csv (PTID, est_birth_date) 생성.

    est_birth_date = YYYY-MM-15 (월 중간일 기준, 평균 0.58일 오차).
    """
    tables_dir = os.path.join(output_dir, 'tables')
    ptdemog = load_rda(os.path.join(rda_dir, 'PTDEMOG.rda'),
                       csv_fallback_dir=tables_dir)
    if ptdemog is None:
        logger.error('PTDEMOG not found, cannot build birth_dates.csv')
        return None

    if 'PTID' not in ptdemog.columns or 'PTDOB' not in ptdemog.columns:
        logger.error('PTDEMOG missing PTID or PTDOB columns')
        return None

    # PTID당 첫 번째 유효 PTDOB
    demog = ptdemog[['PTID', 'PTDOB']].dropna(subset=['PTDOB']).drop_duplicates()
    demog = demog.groupby('PTID', sort=False).first().reset_index()

    def _parse_ptdob(dob_str):
        if pd.isna(dob_str) or not isinstance(dob_str, str):
            return ''
        parts = str(dob_str).strip().split('/')
        if len(parts) == 2:
            month, year = parts[0], parts[1]
            try:
                return '%s-%s-15' % (year, month.zfill(2))
            except (ValueError, TypeError):
                return ''
        return ''

    demog['est_birth_date'] = demog['PTDOB'].apply(_parse_ptdob)
    demog = demog[demog['est_birth_date'] != '']

    out = demog[['PTID', 'est_birth_date']]
    out_path = os.path.join(output_dir, 'birth_dates.csv')
    out.to_csv(out_path, index=False)
    logger.info('birth_dates.csv: %d subjects -> %s', len(out), out_path)
    return out
