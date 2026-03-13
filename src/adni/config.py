"""
config.py — ADNI 공통 NFS 경로 설정

환경변수 ADNI_NFS_ROOT로 오버라이드 가능:
    export ADNI_NFS_ROOT=/mnt/nfs/1_combined
"""

import os

NFS_ROOT = os.environ.get(
    'ADNI_NFS_ROOT',
    '/Volumes/nfs_storage-1/1_combined'
)

# ADNI 파생 경로
ADNI_ORIG = os.path.join(NFS_ROOT, 'ADNI_New', 'ORIG')
ADNI_DEMO = os.path.join(ADNI_ORIG, 'DEMO')
ADNI_TABLES = os.path.join(ADNI_DEMO, 'tables')
ADNI_DCM = os.path.join(ADNI_ORIG, 'DCM')

# NACC 파생 경로 (향후)
NACC_ORIG = os.path.join(NFS_ROOT, 'NACC_NEW', 'ORIG')
