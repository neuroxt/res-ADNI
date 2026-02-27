import numpy as np
import os

FORMAT = '%(asctime)s | ' + os.getlogin() + ' | %(levelname)-5s | %(message)s'

ADNIMERGE_MATCHING_TARGET_COLUMN = 'EXAMDATE_datetime'
ADNIMERGE_VISCODE_TARGET_COLUMN = 'EXAMDATE_bl_datetime'
ADNIMERGE_NO_MATCHING_RESET_COLUMN = [
    'VISCODE', 'EXAMDATE', 'FDG', 'PIB', 'AV45', 'FBB', 'ABETA', 'TAU', 'PTAU', 'CDRSB', 'ADAS11',
    'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting',
    'LDELTOTAL', 'DIGITSCOR', 'TRABSCOR', 'FAQ', 'MOCA', 'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan',
    'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 'EcogSPPlan',
    'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal', 'FLDSTRENG', 'FSVERSION', 'IMAGEUID', 'Ventricles', 'Hippocampus',
    'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'DX', 'mPACCdigit', 'mPACCtrailsB', 'Month', 'M',
    'update_stamp'
]
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
MONTH = 365 / 12
MONTHS = {0: 0, 3 * MONTH: 3, 6 * MONTH: 6, 9 * MONTH: 9, 12 * MONTH: 12}
for i in range(12, 360, 6):
    MONTHS[i * MONTH] = i
MONTHS = {key: 'm%03d' % item for key, item in MONTHS.items()}
MONTH_KEYS = np.array(list(MONTHS.keys()))
T1_MATCHING_CSV = 't1_matching.csv'
MATCHING_CSV = 'multimodal_matching.csv'
