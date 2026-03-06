"""
clinical.py — A4/LEARN 임상 데이터 통합

개별 A4 CSV 파일들을 로드하여 통합 clinical DataFrame 생성.
BID 기준 피험자 수준 테이블 + 방문별 데이터 피벗.
"""

import os
import logging

import numpy as np
import pandas as pd

from .config import (
    NFS_METADATA_BASE, NFS_CLINICAL_BASE,
    CLINICAL_CSV_FILES, IMAGING_CSV_FILES, TREATMENT_CSV,
    BIOMARKER_CSV_FILES, PTAU217_VISIT_MAP,
    format_apoe_genotype, map_ptgender,
)


def _load_csv(base_dir: str, filename: str, **kwargs) -> pd.DataFrame:
    """CSV 파일 로드. 없으면 빈 DataFrame 반환."""
    path = os.path.join(base_dir, filename)
    if not os.path.isfile(path):
        logging.warning('CSV not found: %s' % path)
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False, **kwargs)
    logging.info('Loaded %s: %d rows, %d cols' % (filename, len(df), len(df.columns)))
    return df


def _build_demographics(metadata_dir: str) -> pd.DataFrame:
    """PTDEMOG + SUBJINFO → 피험자 수준 demographics DataFrame."""
    ptdemog = _load_csv(metadata_dir, CLINICAL_CSV_FILES['ptdemog'])
    subjinfo = _load_csv(metadata_dir, CLINICAL_CSV_FILES['subjinfo'])

    if ptdemog.empty and subjinfo.empty:
        return pd.DataFrame()

    # PTDEMOG: BID, PTGENDER, PTAGE (screening age), PTEDUCAT
    demo_cols = ['BID']
    if 'PTGENDER' in ptdemog.columns:
        demo_cols.append('PTGENDER')
    if 'PTAGE' in ptdemog.columns:
        demo_cols.append('PTAGE')
    if 'PTEDUCAT' in ptdemog.columns:
        demo_cols.append('PTEDUCAT')

    demo = ptdemog[demo_cols].drop_duplicates(subset='BID', keep='first').copy()
    demo.set_index('BID', inplace=True)

    # PTGENDER 변환 (1→Male, 2→Female)
    if 'PTGENDER' in demo.columns:
        demo['PTGENDER'] = demo['PTGENDER'].apply(map_ptgender)

    # SUBJINFO: APOEGN, LRNFLGSNM, AGEYR
    if not subjinfo.empty:
        subj_cols = ['BID']
        for c in ['APOEGN', 'LRNFLGSNM', 'AGEYR']:
            if c in subjinfo.columns:
                subj_cols.append(c)
        subj = subjinfo[subj_cols].drop_duplicates(subset='BID', keep='first').copy()
        subj.set_index('BID', inplace=True)

        # APOEGN 정규화 (E3/E4 → e3/e4)
        if 'APOEGN' in subj.columns:
            subj['APOEGN'] = subj['APOEGN'].apply(format_apoe_genotype)

        demo = demo.join(subj, how='outer')

    # Research_Group 결정: LRNFLGSNM == 'Y' → 'LEARN amyloidNE', else → 'amyloidE'
    # (amyloidNE 스크리닝 탈락은 demography.csv에서 별도 처리)
    if 'LRNFLGSNM' in demo.columns:
        demo['Research_Group'] = np.where(
            demo['LRNFLGSNM'].str.upper() == 'Y',
            'LEARN amyloidNE',
            'amyloidE',
        )
    else:
        demo['Research_Group'] = ''

    return demo


def _build_amyloid_status(metadata_dir: str) -> pd.DataFrame:
    """PETVADATA → 아밀로이드 적격성 (BID 수준)."""
    imaging_dir = os.path.join(metadata_dir, 'A4 Imaging data and docs')
    petvadata = _load_csv(imaging_dir, IMAGING_CSV_FILES['petvadata'])

    if petvadata.empty:
        return pd.DataFrame()

    # BID, PMODSUVR (composite SUVR), SCORE (positive/negative)
    cols = ['BID']
    for c in ['PMODSUVR', 'SCORE']:
        if c in petvadata.columns:
            cols.append(c)
    amy = petvadata[cols].drop_duplicates(subset='BID', keep='first').copy()
    amy.set_index('BID', inplace=True)

    rename = {}
    if 'PMODSUVR' in amy.columns:
        rename['PMODSUVR'] = 'AMY_SUVR'
    if 'SCORE' in amy.columns:
        rename['SCORE'] = 'AMY_STATUS'
    amy.rename(columns=rename, inplace=True)

    return amy


def _build_pet_suvr(metadata_dir: str) -> pd.DataFrame:
    """PETSUVR → amyloid PET composite SUVR + centiloid (BID 수준, screening).

    Long-format → wide-format: Composite_Summary 값만 추출.
    """
    imaging_dir = os.path.join(metadata_dir, 'A4 Imaging data and docs')
    petsuvr = _load_csv(imaging_dir, IMAGING_CSV_FILES['petsuvr'])

    if petsuvr.empty:
        return pd.DataFrame()

    # Composite_Summary 행만 필터
    if 'brain_region' not in petsuvr.columns:
        return pd.DataFrame()

    composite = petsuvr[petsuvr['brain_region'] == 'Composite_Summary'].copy()
    if composite.empty:
        # 다른 이름 시도
        composite = petsuvr[petsuvr['brain_region'].str.contains('Composite', case=False, na=False)].copy()
    if composite.empty:
        return pd.DataFrame()

    cols = ['BID']
    rename = {}
    if 'suvr_cer' in composite.columns:
        cols.append('suvr_cer')
        rename['suvr_cer'] = 'AMY_SUVR_CER'
    if 'centiloid' in composite.columns:
        cols.append('centiloid')
        rename['centiloid'] = 'AMY_CENTILOID'

    result = composite[cols].drop_duplicates(subset='BID', keep='first').copy()
    result.set_index('BID', inplace=True)
    result.rename(columns=rename, inplace=True)

    return result


def _build_vmri(metadata_dir: str) -> pd.DataFrame:
    """VMRI → MRI 체적 (BID 수준, VISCODE=4 baseline)."""
    imaging_dir = os.path.join(metadata_dir, 'A4 Imaging data and docs')
    vmri = _load_csv(imaging_dir, IMAGING_CSV_FILES['vmri'])

    if vmri.empty:
        return pd.DataFrame()

    # VISCODE=4 (baseline MRI session)
    if 'VISCODE' in vmri.columns:
        vmri = vmri[vmri['VISCODE'] == 4].copy()

    if 'BID' not in vmri.columns:
        return pd.DataFrame()

    # 피험자별 첫 번째 행
    vmri = vmri.drop_duplicates(subset='BID', keep='first')
    vmri.set_index('BID', inplace=True)

    # BID와 VISCODE 외의 모든 컬럼 유지 (FreeSurfer regions)
    drop_cols = [c for c in ['VISCODE', 'Phase'] if c in vmri.columns]
    vmri.drop(columns=drop_cols, inplace=True, errors='ignore')

    # 컬럼명에 VMRI_ 접두사 추가
    vmri.columns = ['VMRI_' + c for c in vmri.columns]

    return vmri


def _build_tau_suvr(metadata_dir: str) -> pd.DataFrame:
    """TAUSUVR → Tau PET SUVR (서브셋, BID 수준)."""
    imaging_dir = os.path.join(metadata_dir, 'A4 Imaging data and docs')
    tausuvr = _load_csv(imaging_dir, IMAGING_CSV_FILES['tausuvr'])

    if tausuvr.empty:
        return pd.DataFrame()

    # ID 컬럼이 BID 역할
    bid_col = 'ID' if 'ID' in tausuvr.columns else 'BID'
    if bid_col not in tausuvr.columns:
        return pd.DataFrame()

    tausuvr = tausuvr.drop_duplicates(subset=bid_col, keep='first')
    if bid_col != 'BID':
        tausuvr = tausuvr.rename(columns={bid_col: 'BID'})
    tausuvr.set_index('BID', inplace=True)

    # 컬럼명에 TAU_ 접두사 추가
    tausuvr.columns = ['TAU_' + c for c in tausuvr.columns]

    return tausuvr


def _build_cognitive(metadata_dir: str) -> pd.DataFrame:
    """MMSE + CDR → 인지 평가 (BID 수준, 최초 방문 = 스크리닝).

    현재 공개 데이터 = pre-randomization만이므로 단일 행.
    """
    mmse = _load_csv(metadata_dir, CLINICAL_CSV_FILES['mmse'])
    cdr = _load_csv(metadata_dir, CLINICAL_CSV_FILES['cdr'])

    parts = []

    if not mmse.empty and 'BID' in mmse.columns:
        # 최초 방문 MMSE
        mmse_first = mmse.sort_values('VISCODE').drop_duplicates(subset='BID', keep='first')
        mmse_cols = ['BID']
        if 'MMSCORE' in mmse_first.columns:
            mmse_cols.append('MMSCORE')
        parts.append(mmse_first[mmse_cols].set_index('BID'))

    if not cdr.empty and 'BID' in cdr.columns:
        cdr_first = cdr.sort_values('VISCODE').drop_duplicates(subset='BID', keep='first')
        cdr_cols = ['BID']
        for c in ['CDGLOBAL', 'CDSOB']:
            if c in cdr_first.columns:
                cdr_cols.append(c)
        parts.append(cdr_first[cdr_cols].set_index('BID'))

    if not parts:
        return pd.DataFrame()

    result = parts[0]
    for p in parts[1:]:
        result = result.join(p, how='outer')

    return result


def _build_treatment(metadata_dir: str) -> pd.DataFrame:
    """dose.csv → 치료군 정보 (BID 수준, amyloidE만)."""
    dose = _load_csv(metadata_dir, TREATMENT_CSV)

    if dose.empty or 'BID' not in dose.columns:
        return pd.DataFrame()

    dose = dose.drop_duplicates(subset='BID', keep='first')
    dose.set_index('BID', inplace=True)

    # DOSELEVEL: 1600(mg) = solanezumab, NaN = placebo
    if 'DOSELEVEL' in dose.columns:
        dose['TREATMENT'] = np.where(
            dose['DOSELEVEL'].notna() & (dose['DOSELEVEL'] != ''),
            'solanezumab',
            'placebo',
        )
    cols = [c for c in ['DOSELEVEL', 'TREATMENT'] if c in dose.columns]
    return dose[cols] if cols else pd.DataFrame()


def _build_ptau217(clinical_dir: str) -> pd.DataFrame:
    """biomarker_pTau217.csv → wide-format pTau217 (BID 수준).

    VISCODE별 피벗:
    - A4: PTAU217_BL(6), PTAU217_WK12(9), PTAU217_WK240(66)
    - LEARN: PTAU217_SCR(1), PTAU217_WK72(24), PTAU217_WK240(66)

    값: ORRESRAW (수치, LLOQ 이하도 포함)
    LLOQ flag: PTAU217_{visit}_LLOQ (ORRES가 '<LLOQ'이면 True)
    """
    ext_dir = os.path.join(clinical_dir, 'External Data')
    ptau = _load_csv(ext_dir, BIOMARKER_CSV_FILES['ptau217'])

    if ptau.empty or 'BID' not in ptau.columns:
        return pd.DataFrame()

    # 필수 컬럼 확인
    for c in ['VISCODE', 'ORRESRAW', 'SUBSTUDY']:
        if c not in ptau.columns:
            logging.warning('pTau217 missing column: %s' % c)
            return pd.DataFrame()

    # LLOQ flag
    if 'ORRES' in ptau.columns:
        ptau['_LLOQ'] = ptau['ORRES'].astype(str).str.contains('<LLOQ', case=False, na=False)
    else:
        ptau['_LLOQ'] = False

    # SUBSTUDY별 피벗
    result_parts = []
    for substudy, visit_map in PTAU217_VISIT_MAP.items():
        sub_df = ptau[ptau['SUBSTUDY'].str.upper() == substudy].copy()
        if sub_df.empty:
            continue

        for viscode, col_name in visit_map.items():
            visit_rows = sub_df[sub_df['VISCODE'] == viscode].copy()
            if visit_rows.empty:
                continue

            # BID별 첫 번째 값
            visit_rows = visit_rows.drop_duplicates(subset='BID', keep='first')
            visit_data = visit_rows[['BID', 'ORRESRAW', '_LLOQ']].copy()
            visit_data = visit_data.rename(columns={
                'ORRESRAW': col_name,
                '_LLOQ': col_name + '_LLOQ',
            })
            visit_data.set_index('BID', inplace=True)
            result_parts.append(visit_data)

    if not result_parts:
        return pd.DataFrame()

    result = result_parts[0]
    for p in result_parts[1:]:
        result = result.join(p, how='outer')

    return result


def _build_demography_groups(metadata_dir: str) -> pd.DataFrame:
    """A4_demography.csv → Research Group 매핑 (amyloidNE 포함).

    demography.csv에는 모든 3 그룹이 포함되어 있으므로
    SUBJINFO에 없는 amyloidNE를 보충.
    """
    demog = _load_csv(metadata_dir, CLINICAL_CSV_FILES['demography'])

    if demog.empty:
        return pd.DataFrame()

    # Subject ID → BID
    bid_col = 'Subject ID' if 'Subject ID' in demog.columns else 'BID'
    if bid_col not in demog.columns:
        return pd.DataFrame()

    rg_col = 'Research Group' if 'Research Group' in demog.columns else None
    if rg_col is None:
        return pd.DataFrame()

    groups = demog[[bid_col, rg_col]].drop_duplicates(subset=bid_col, keep='first')
    groups = groups.rename(columns={bid_col: 'BID', rg_col: 'Research_Group_demog'})
    groups.set_index('BID', inplace=True)

    return groups


def build_clinical_table(metadata_dir: str = None,
                         clinical_dir: str = None,
                         include_screen_fail: bool = False) -> pd.DataFrame:
    """개별 A4 CSV → 통합 clinical DataFrame (BID 기준).

    Args:
        metadata_dir: metadata/ 경로 (기본: NFS_METADATA_BASE)
        clinical_dir: DEMO/Clinical/ 경로 (기본: NFS_CLINICAL_BASE)
        include_screen_fail: True이면 amyloidNE도 포함 (기본: 제외)

    Returns:
        DataFrame indexed by BID.
    """
    if metadata_dir is None:
        metadata_dir = NFS_METADATA_BASE
    if clinical_dir is None:
        clinical_dir = NFS_CLINICAL_BASE

    logging.info('Building clinical table from %s' % metadata_dir)

    # 1. 기본 demographics (PTDEMOG + SUBJINFO)
    demo = _build_demographics(metadata_dir)
    if demo.empty:
        logging.error('No demographics data — cannot build clinical table')
        return pd.DataFrame()
    logging.info('Demographics: %d BIDs' % len(demo))

    # 2. Amyloid status (PETVADATA)
    amy_status = _build_amyloid_status(metadata_dir)
    if not amy_status.empty:
        demo = demo.join(amy_status, how='left')
        logging.info('Amyloid status: %d BIDs' % len(amy_status))

    # 3. PET SUVR composite
    pet_suvr = _build_pet_suvr(metadata_dir)
    if not pet_suvr.empty:
        demo = demo.join(pet_suvr, how='left')
        logging.info('PET SUVR: %d BIDs' % len(pet_suvr))

    # 4. Cognitive scores (MMSE, CDR)
    cognitive = _build_cognitive(metadata_dir)
    if not cognitive.empty:
        demo = demo.join(cognitive, how='left')
        logging.info('Cognitive: %d BIDs' % len(cognitive))

    # 5. Treatment (dose.csv)
    treatment = _build_treatment(metadata_dir)
    if not treatment.empty:
        demo = demo.join(treatment, how='left')
        logging.info('Treatment: %d BIDs' % len(treatment))

    # 6. VMRI (NeuroQuant)
    vmri = _build_vmri(metadata_dir)
    if not vmri.empty:
        demo = demo.join(vmri, how='left')
        logging.info('VMRI: %d BIDs' % len(vmri))

    # 7. Tau PET SUVR (서브셋)
    tau_suvr = _build_tau_suvr(metadata_dir)
    if not tau_suvr.empty:
        demo = demo.join(tau_suvr, how='left')
        logging.info('Tau SUVR: %d BIDs' % len(tau_suvr))

    # 8. pTau217 혈액 바이오마커
    ptau217 = _build_ptau217(clinical_dir)
    if not ptau217.empty:
        demo = demo.join(ptau217, how='left')
        logging.info('pTau217: %d BIDs' % len(ptau217))

    # 9. amyloidNE Research Group 보충 (demography.csv)
    demog_groups = _build_demography_groups(metadata_dir)
    if not demog_groups.empty:
        # Research_Group이 비어있는 BID에 demography 값 채우기
        demo = demo.join(demog_groups, how='outer')
        if 'Research_Group_demog' in demo.columns:
            mask = demo['Research_Group'].isna() | (demo['Research_Group'] == '')
            demo.loc[mask, 'Research_Group'] = demo.loc[mask, 'Research_Group_demog']
            demo.drop(columns=['Research_Group_demog'], inplace=True)

    # 10. amyloidNE 제외 (기본값)
    if not include_screen_fail:
        before = len(demo)
        demo = demo[demo['Research_Group'] != 'amyloidNE']
        logging.info('Excluded amyloidNE: %d → %d BIDs' % (before, len(demo)))

    logging.info('Clinical table complete: %d BIDs, %d columns' % (len(demo), len(demo.columns)))
    return demo
