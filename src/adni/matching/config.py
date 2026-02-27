"""
config.py — 상수, 경로, 모달리티 설정

params.py 대체 + 경로/모달리티별 regex 통합
"""

import numpy as np

# =============================================================================
# Logging
# =============================================================================
LOG_FORMAT = '%(asctime)s | %(levelname)-5s | %(message)s'

# =============================================================================
# ADNIMERGE Column Settings (params.py와 동일)
# =============================================================================
ADNIMERGE_MATCHING_TARGET_COLUMN = 'EXAMDATE_datetime'
ADNIMERGE_VISCODE_TARGET_COLUMN = 'EXAMDATE_bl_datetime'

# threshold 초과 시 비워야 할 컬럼 (방문별 데이터)
ADNIMERGE_NO_MATCHING_RESET_COLUMN = [
    'VISCODE', 'EXAMDATE', 'FDG', 'PIB', 'AV45', 'FBB', 'ABETA', 'TAU', 'PTAU', 'CDRSB', 'ADAS11',
    'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting',
    'LDELTOTAL', 'DIGITSCOR', 'TRABSCOR', 'FAQ', 'MOCA', 'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan',
    'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 'EcogSPPlan',
    'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal', 'FLDSTRENG', 'FSVERSION', 'IMAGEUID', 'Ventricles', 'Hippocampus',
    'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'DX', 'mPACCdigit', 'mPACCtrailsB', 'Month', 'M',
    'update_stamp'
]

# threshold 초과해도 유지할 컬럼 (피험자 고정 데이터)
ADNIMERGE_NO_MATCHING_KEEP_COLUMN = [
    'RID', 'COLPROT', 'ORIGPROT', 'PTID', 'SITE', 'DX_bl', 'AGE', 'PTGENDER', 'PTEDUCAT',
    'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'APOE4', 'EXAMDATE_bl',
    'CDRSB_bl', 'ADAS11_bl', 'ADAS13_bl', 'ADASQ4_bl', 'MMSE_bl', 'RAVLT_immediate_bl', 'RAVLT_learning_bl',
    'RAVLT_forgetting_bl', 'RAVLT_perc_forgetting_bl', 'LDELTOTAL_BL', 'DIGITSCOR_bl', 'TRABSCOR_bl', 'FAQ_bl',
    'mPACCdigit_bl', 'mPACCtrailsB_bl', 'FLDSTRENG_bl', 'FSVERSION_bl', 'IMAGEUID_bl', 'Ventricles_bl',
    'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl', 'ICV_bl', 'MOCA_bl',
    'EcogPtMem_bl', 'EcogPtLang_bl', 'EcogPtVisspat_bl', 'EcogPtPlan_bl', 'EcogPtOrgan_bl', 'EcogPtDivatt_bl',
    'EcogPtTotal_bl', 'EcogSPMem_bl', 'EcogSPLang_bl', 'EcogSPVisspat_bl', 'EcogSPPlan_bl', 'EcogSPOrgan_bl',
    'EcogSPDivatt_bl', 'EcogSPTotal_bl', 'ABETA_bl', 'TAU_bl', 'PTAU_bl', 'FDG_bl', 'PIB_bl', 'AV45_bl', 'FBB_bl',
    'Years_bl', 'Month_bl',
]

# =============================================================================
# VISCODE Mapping (params.py와 동일)
# =============================================================================
MONTH = 365 / 12
MONTHS = {0: 0, 3 * MONTH: 3, 6 * MONTH: 6, 9 * MONTH: 9, 12 * MONTH: 12}
for i in range(12, 360, 6):
    MONTHS[i * MONTH] = i
MONTHS = {key: 'm%03d' % item for key, item in MONTHS.items()}
MONTH_KEYS = np.array(list(MONTHS.keys()))

# =============================================================================
# DCM/NII 경로 설정
# =============================================================================
# NFS 기본 경로 (ADNI_New로 이전)
NFS_BASE = '/Volumes/nfs_storage/1_combined/ADNI_New/ORIG/DCM'

# 출력 기본 경로 (DCM 소스 경로와 분리)
OUTPUT_BASE = '/Volumes/nfs_storage/1_combined/ADNI_New/ORIG'

# DCM 소스별 경로 (source → 실제 디렉토리명)
# 2026-02: ADNI_New 구조 — T1/T2/PET/DTI/fMRI/MRI(ADNI4 혼합)
DCM_SOURCES = {
    'T1': 'T1',
    'T2': 'T2',
    'PET': 'PET',
    'DTI': 'DTI',
    'fMRI': 'fMRI',
    'MRI': 'MRI',
}

# =============================================================================
# 모달리티별 설정
# =============================================================================
# 모달리티 정의: 16개 (기존 13 + 신규 3: T2_3D, HIPPO, ASL)
# sources: DCM_SOURCES 키 리스트 (어떤 소스 폴더에서 가져올지)
# source_specific_regex: 소스별 include 패턴 (MRI/ 등 혼합 소스에서 모달리티 구분)
# regex: 모든 소스에 동일한 include 패턴 (source_specific_regex가 없을 때)
# exclude_regex: 모든 소스에 적용되는 제외 패턴
MODALITY_CONFIG = {
    'T1': {
        'sources': ['T1', 'MRI'],
        'source_specific_regex': {'T1': ['*'], 'MRI': ['*MPRAGE*', '*IR-FSPGR*']},
        'exclude_regex': ['*B1*Calibration*'],
        'threshold': 180,
        'file_type': 'dcm',
    },
    'AV45_8MM': {
        'sources': ['PET'],
        'regex': ['*AV45*Uniform_Resolution*'],
        'threshold': 180,
        'file_type': 'dcm',
    },
    'AV45_6MM': {
        'sources': ['PET'],
        'regex': ['*AV45*6mm*'],
        'threshold': 180,
        'file_type': 'dcm',
    },
    'AV1451_8MM': {
        'sources': ['PET'],
        'regex': ['*AV1451*Uniform_Resolution*'],
        'threshold': 180,
        'file_type': 'dcm',
    },
    'AV1451_6MM': {
        'sources': ['PET'],
        'regex': ['*AV1451*6mm*'],
        'threshold': 180,
        'file_type': 'dcm',
    },
    'FBB_6MM': {
        'sources': ['PET'],
        'regex': ['*FBB*6mm*'],
        'threshold': 180,
        'file_type': 'dcm',
    },
    'FLAIR': {
        'sources': ['T2', 'MRI'],
        'source_specific_regex': {'T2': ['*[Ff][Ll][Aa][Ii][Rr]*'], 'MRI': ['*FLAIR*']},
        'threshold': 180,
        'file_type': 'dcm',
    },
    'T2_FSE': {
        'sources': ['T2'],
        'regex': ['*[Ff][Ss][Ee]*'],
        'threshold': 180,
        'file_type': 'dcm',
    },
    'T2_TSE': {
        'sources': ['T2'],
        'regex': ['*[Tt][Ss][Ee]*'],
        'threshold': 180,
        'file_type': 'dcm',
    },
    'T2_STAR': {
        'sources': ['T2', 'MRI'],
        'source_specific_regex': {
            'T2': ['*[Ss][Tt][Aa][Rr]*', '*[Gg][Rr][Ee]*'],
            'MRI': ['*T2_[Ss][Tt][Aa][Rr]*', '*T2_GRE*', '*ME_T2_GRE*'],
        },
        'threshold': 180,
        'file_type': 'dcm',
    },
    'DTI': {
        'sources': ['DTI', 'MRI'],
        'source_specific_regex': {
            'DTI': ['*DTI*', '*[Dd][Ww][Ii]*'],
            'MRI': ['*DTI*'],
        },
        'exclude_regex': ['*MB*', '*ADC*', '*FA*', '*TRACEW*', '*Average_DC*',
                          '*Fractional_Aniso*', '*Isotropic*'],
        'threshold': 180,
        'file_type': 'dcm',
    },
    'DTI_MB': {
        'sources': ['DTI', 'MRI'],
        'source_specific_regex': {
            'DTI': ['*MB*DTI*', '*DTI*MB*'],
            'MRI': ['*MB_DTI*'],
        },
        'exclude_regex': ['*ADC*', '*FA*', '*TRACEW*'],
        'threshold': 180,
        'file_type': 'dcm',
    },
    'FMRI': {
        'sources': ['fMRI', 'MRI'],
        'source_specific_regex': {
            'fMRI': ['*fMRI*', '*rsfMRI*', '*fcMRI*', '*[Rr]esting*[Ss]tate*', '*BOLD*', '*[Rr]esting*'],
            'MRI': ['*rsfMRI*', '*fcMRI*'],
        },
        'exclude_regex': ['*relCBF*', '*MoCoSeries*', '*Perfusion*', '*pCASL*', '*PASL*', '*DTI*'],
        'threshold': 180,
        'file_type': 'dcm',
    },
    # === 신규 모달리티 (MRI/ ADNI4) ===
    'T2_3D': {
        'sources': ['MRI'],
        'regex': ['*T2_SPACE*', '*T2_[Cc][Uu][Bb][Ee]*', '*T2_[Vv][Ii][Ss][Tt][Aa]*',
                  '*SAG*T2*SPACE*', '*3D_T2_CUBE*', '*3D_T2_VISTA*'],
        'threshold': 180,
        'file_type': 'dcm',
    },
    'HIPPO': {
        'sources': ['MRI'],
        'regex': ['*HighResHippo*'],
        'threshold': 180,
        'file_type': 'dcm',
    },
    'ASL': {
        'sources': ['MRI'],
        'regex': ['*PASL*', '*pCASL*', '*pcasl*', '*3DpCASL*'],
        'exclude_regex': ['*WIP_SOURCE*'],
        'threshold': 180,
        'file_type': 'dcm',
    },
    # === 신규 PET tracers (ADNI4) ===
    'MK6240_6MM': {
        'sources': ['PET'],
        'regex': ['*MK6240*6mm*'],
        'threshold': 180,
        'file_type': 'dcm',
    },
    'NAV4694_6MM': {
        'sources': ['PET'],
        'regex': ['*NAV4694*6mm*'],
        'threshold': 180,
        'file_type': 'dcm',
    },
    'PI2620_6MM': {
        'sources': ['PET'],
        'regex': ['*PI2620*6mm*'],
        'threshold': 180,
        'file_type': 'dcm',
    },
}

# =============================================================================
# MRIQC 프로토콜 필드 매핑
# =============================================================================
# MRIQC 컬럼 → 출력 protocol 컬럼 매핑
MRIQC_PROTOCOL_FIELDS = {
    'SeriesDescription': 'description',
    'AcquisitionType': 'Acquisition Type',
    'AcquisitionPlane': 'Acquisition Plane',
    'Weighting': 'Weighting',
    'PulseSequence': 'Pulse Sequence',
    'SliceThickness': 'Slice Thickness',
    'PixelSpacingX': 'Pixel Spacing X',
    'PixelSpacingY': 'Pixel Spacing Y',
    'ReceiveCoilName': 'Coil',
    'SeriesType': 'Weighting',
    'MatrixX': 'Matrix X',
    'MatrixY': 'Matrix Y',
    'MatrixZ': 'Matrix Z',
    'ScannerManufacturer': 'Manufacturer',
    'ScannerModel': 'Mfg Model',
    'MagneticFieldStrength': 'Field Strength',
}

# pydicom으로 DCM에서 직접 추출할 필드 (DICOM tag → 출력 이름)
DCM_PROTOCOL_FIELDS = {
    (0x0018, 0x0081): 'TE',
    (0x0018, 0x0080): 'TR',
    (0x0018, 0x0082): 'TI',
    (0x0018, 0x1314): 'Flip Angle',
    (0x0018, 0x0020): 'Pulse Sequence',
    (0x0028, 0x0030): 'Pixel Spacing',
    (0x0028, 0x0010): 'Matrix Y',          # Rows
    (0x0028, 0x0011): 'Matrix X',          # Columns
    (0x0028, 0x0008): 'Matrix Z',          # NumberOfFrames
}

# =============================================================================
# APOE genotype 형식 변환
# =============================================================================
# APOERES GENOTYPE (e.g., "3/4") → ADNI.py 형식 (e.g., "e3/e4")
def format_apoe_genotype(genotype_str: str) -> str:
    if not genotype_str or not isinstance(genotype_str, str):
        return ''
    parts = genotype_str.strip().split('/')
    if len(parts) == 2:
        return 'e%s/e%s' % (parts[0].strip(), parts[1].strip())
    return genotype_str


# =============================================================================
# UCBerkeley PET Quantification Attach 설정
# =============================================================================
# 모달리티별 UCBerkeley 테이블 매핑
# - table: csv/tables/ 아래 파일명
# - date_col: UCBerkeley 날짜 컬럼
# - threshold: 날짜 매칭 허용 오차 (일)
# - tracer_filter: TRACER 컬럼 필터 (AMY: FBP=AV45, FBB; TAU: FTP=AV1451)
UCBERKELEY_ATTACH_CONFIG = {
    'AV45_6MM': {
        'table': 'UCBERKELEY_AMY_6MM.csv',
        'date_col': 'SCANDATE',
        'threshold': 30,
        'tracer_filter': ['FBP'],       # Florbetapir = AV45
        'column_prefix': 'UCBERKELEY_AMY/',
    },
    'FBB_6MM': {
        'table': 'UCBERKELEY_AMY_6MM.csv',
        'date_col': 'SCANDATE',
        'threshold': 30,
        'tracer_filter': ['FBB'],       # Florbetaben
        'column_prefix': 'UCBERKELEY_AMY/',
    },
    'AV1451_6MM': {
        'table': 'UCBERKELEY_TAU_6MM.csv',
        'date_col': 'SCANDATE',
        'threshold': 30,
        'tracer_filter': ['FTP'],       # Flortaucipir = AV1451
        'column_prefix': 'UCBERKELEY_TAU/',
    },
    'MK6240_6MM': {
        'table': 'UCBERKELEY_TAU_6MM.csv',
        'date_col': 'SCANDATE',
        'threshold': 30,
        'tracer_filter': ['MK6240'],
        'column_prefix': 'UCBERKELEY_TAU_MK6240/',
    },
    'NAV4694_6MM': {
        'table': 'UCBERKELEY_AMY_6MM.csv',
        'date_col': 'SCANDATE',
        'threshold': 30,
        'tracer_filter': ['NAV'],
        'column_prefix': 'UCBERKELEY_AMY_NAV/',
    },
    'PI2620_6MM': {
        'table': 'UCBERKELEY_TAU_6MM.csv',
        'date_col': 'SCANDATE',
        'threshold': 30,
        'tracer_filter': ['PI2620'],
        'column_prefix': 'UCBERKELEY_TAU_PI2620/',
    },
}
