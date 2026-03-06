"""
config.py — A4/LEARN 전용 설정

NFS 경로, 모달리티 설정, JSON sidecar 필드 매핑, 임상 CSV 파일 매핑.
"""

import re

# =============================================================================
# Logging
# =============================================================================
LOG_FORMAT = '%(asctime)s | %(levelname)-5s | %(message)s'

# =============================================================================
# NFS 경로
# =============================================================================
NFS_NII_BASE = '/Volumes/nfs_storage/1_combined/A4/ORIG/NII'
NFS_METADATA_BASE = '/Volumes/nfs_storage/1_combined/A4/ORIG/metadata'
NFS_CLINICAL_BASE = '/Volumes/nfs_storage/1_combined/A4/ORIG/DEMO/Clinical'
OUTPUT_BASE = '/Volumes/nfs_storage/1_combined/A4/ORIG/DEMO'

# =============================================================================
# BID 정규식 (B + 1자리 비제로 + 8자리)
# =============================================================================
BID_RE = re.compile(r'^B[1-9]\d{8}$')

# =============================================================================
# 모달리티 설정
# =============================================================================
MODALITY_CONFIG = {
    'T1':        {'type': 'MR', 'folder': 'T1'},
    'FLAIR':     {'type': 'MR', 'folder': 'FLAIR'},
    'T2_SE':     {'type': 'MR', 'folder': 'T2_SE'},
    'T2_STAR':   {'type': 'MR', 'folder': 'T2_star'},
    'FBP':       {'type': 'PET', 'folder': 'FBP', 'tracer': 'Florbetapir'},
    'FTP':       {'type': 'PET', 'folder': 'FTP', 'tracer': 'Flortaucipir'},
    'FMRI_REST': {'type': 'MR', 'folder': 'fMRI_rest'},
    'B0CD':      {'type': 'MR', 'folder': 'b0CD'},
}

# NII 파일명 패턴: A4_{MR|PET}_{modality}_{BID}_{session}.nii.gz
NII_PATTERN = re.compile(
    r'^A4_(?:MR|PET)_\w+_B[1-9]\d{8}_\d{3}\.nii\.gz$'
)

# MERGED.csv에서 제외할 모달리티 (파생 데이터)
MERGE_EXCLUDE = ['DWI']

# =============================================================================
# JSON sidecar → 출력 컬럼 매핑 (MRI)
# =============================================================================
JSON_MR_FIELDS = {
    'EchoTime': 'TE',
    'RepetitionTime': 'TR',
    'InversionTime': 'TI',
    'FlipAngle': 'Flip Angle',
    'SliceThickness': 'Slice Thickness',
    'Manufacturer': 'Manufacturer',
    'ManufacturersModelName': 'Mfg Model',
    'MagneticFieldStrength': 'Field Strength',
    'SeriesDescription': 'description',
    'ProtocolName': 'Protocol Name',
}

# JSON sidecar → 출력 컬럼 매핑 (PET)
JSON_PET_FIELDS = {
    'Radiopharmaceutical': 'Tracer',
    'InjectedRadioactivity': 'Injected Dose',
    'FrameDuration': 'Frame Duration',
    'ReconstructionMethod': 'Recon Method',
    'Manufacturer': 'Manufacturer',
    'ManufacturersModelName': 'Mfg Model',
    'SliceThickness': 'Slice Thickness',
}

# =============================================================================
# 임상 CSV 파일 매핑 (metadata/ 기준)
# =============================================================================
CLINICAL_CSV_FILES = {
    'ptdemog':    'A4_PTDEMOG_PRV2_11Aug2025.csv',
    'subjinfo':   'A4_SUBJINFO_PRV2_11Aug2025.csv',
    'registry':   'A4_REGISTRY_PRV2_11Aug2025.csv',
    'mmse':       'A4_MMSE_PRV2_11Aug2025.csv',
    'cdr':        'A4_CDR_PRV2_11Aug2025.csv',
    'pacc':       'A4_SPPACC_PRV2_11Aug2025.csv',
    'demography': 'A4_demography.csv',
}

# 영상 관련 CSV (metadata/A4 Imaging data and docs/ 기준)
IMAGING_CSV_FILES = {
    'petsuvr':   'A4_PETSUVR_PRV2_11Aug2025.csv',
    'petvadata': 'A4_PETVADATA_PRV2_11Aug2025.csv',
    'vmri':      'A4_VMRI_PRV2_11Aug2025.csv',
    'tausuvr':   'TAUSUVR_11Aug2025.csv',
}

# 치료군 정보 (metadata/ 기준)
TREATMENT_CSV = 'dose.csv'

# =============================================================================
# 혈액 바이오마커 CSV (DEMO/Clinical/External Data/ 기준)
# =============================================================================
BIOMARKER_CSV_FILES = {
    'ptau217':      'biomarker_pTau217.csv',
    'roche_plasma': 'biomarker_Plasma_Roche_Results.csv',
    'ab_test':      'biomarker_AB_Test.csv',
}

# =============================================================================
# pTau217 VISCODE → 컬럼명 매핑
# =============================================================================
PTAU217_VISIT_MAP = {
    'A4': {
        6:   'PTAU217_BL',
        9:   'PTAU217_WK12',
        66:  'PTAU217_WK240',
        997: 'PTAU217_OLE',
        999: 'PTAU217_ET',
    },
    'LEARN': {
        1:   'PTAU217_SCR',
        24:  'PTAU217_WK72',
        66:  'PTAU217_WK240',
        999: 'PTAU217_ET',
    },
}


# =============================================================================
# APOE genotype 정규화 (ADNI 호환)
# =============================================================================
def format_apoe_genotype(genotype_str: str) -> str:
    """A4 APOEGN (e.g., 'E3/E4') → ADNI 형식 (e.g., 'e3/e4')."""
    if not genotype_str or not isinstance(genotype_str, str):
        return ''
    g = genotype_str.strip()
    # E3/E4 → e3/e4
    parts = g.replace('E', 'e').split('/')
    if len(parts) == 2:
        return '%s/%s' % (parts[0].strip(), parts[1].strip())
    return g


def map_ptgender(gender_val) -> str:
    """A4 PTGENDER (1=Male, 2=Female) → M/F."""
    try:
        g = int(gender_val)
    except (ValueError, TypeError):
        return ''
    if g == 1:
        return 'Male'
    elif g == 2:
        return 'Female'
    return ''
