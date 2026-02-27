#!/usr/bin/env python3
"""
cli.py — 전체 매칭 파이프라인 실행

기존 10_run_matching_merged.py + ADNI.py를 대체하는 XML-free 파이프라인.

사용법:
    python -m adni.matching [OPTIONS]

    --modality T1,AV45,...    실행할 모달리티 (쉼표 구분, 기본: 전체)
    --merge-only              MERGED.csv만 생성
    --inventory-only          DCM inventory만 생성
    --n-jobs N                병렬 작업 수 (기본: 8)
    --output-dir DIR          출력 디렉토리
    --overwrite               기존 결과 덮어쓰기
    --skip-inventory          DCM inventory 생성 건너뛰기 (기존 inventory 사용)
"""

import os
import sys
import argparse
import logging

# 프로젝트 루트 (src/adni/matching/cli.py → matching → adni → src → project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, PROJECT_ROOT)

from adni.matching.config import NFS_BASE, OUTPUT_BASE, MODALITY_CONFIG, UCBERKELEY_ATTACH_CONFIG
from adni.matching.utils import setup_logger
from adni.matching.inventory import build_inventory, save_inventory, load_inventory
from adni.matching.matching import match_modality, attach_ucberkeley
from adni.matching.merge import unique_csv_merge
from glob import glob

# =============================================================================
# Default Configuration
# =============================================================================

# 출력 디렉토리
DEFAULT_OUTPUT_DIR = os.path.join(OUTPUT_BASE, 'DEMO', 'ADNI_matching_v4')

# CSV 경로 (csv/ 디렉토리 기준)
DEFAULT_ADNIMERGE_CSV = os.path.join(PROJECT_ROOT, 'csv', 'ADNIMERGE_260213.csv')
DEFAULT_MRIQC_CSV = os.path.join(PROJECT_ROOT, 'csv', 'tables', 'MRIQC.csv')
DEFAULT_APOERES_CSV = os.path.join(PROJECT_ROOT, 'csv', 'tables', 'APOERES.csv')
DEFAULT_BIRTH_DATES_CSV = os.path.join(PROJECT_ROOT, 'csv', 'birth_dates.csv')

# UCBerkeley 테이블 디렉토리
DEFAULT_UCB_TABLES_DIR = os.path.join(PROJECT_ROOT, 'csv', 'tables')

# MERGED.csv에서 제외할 모달리티 (파생 데이터)
MERGE_EXCLUDE = ['ADC', 'FA', 'TRACEW']

# 기본 threshold (일)
DEFAULT_THRESHOLD = 180


def run_inventory(output_dir: str, nfs_base: str = None, force_rebuild: bool = False,
                  scan_workers: int = 8) -> dict:
    """DCM inventory 생성 또는 로드"""
    inventory_path = os.path.join(output_dir, 'dcm_inventory.json')

    if not force_rebuild and os.path.isfile(inventory_path):
        logging.info('Loading existing inventory from %s' % inventory_path)
        return load_inventory(inventory_path)

    logging.info('Building DCM inventory (v2, optimized)...')
    inventory = build_inventory(nfs_base=nfs_base, ptid_workers=scan_workers)
    save_inventory(inventory, inventory_path)
    return inventory


def run_matching(modalities: list, output_dir: str, n_jobs: int,
                 overwrite: bool, dcm_inventory: dict,
                 adnimerge_csv: str, mriqc_csv: str,
                 apoeres_csv: str, birth_dates_csv: str):
    """모달리티별 매칭 실행 (인벤토리 기반)"""
    log_path = os.path.join(output_dir, 'matching.log')

    if dcm_inventory is None:
        logging.error('DCM inventory is required for matching. Run inventory first.')
        return

    for mod in modalities:
        mod_upper = mod.upper()
        if mod_upper not in MODALITY_CONFIG:
            logging.warning('Unknown modality: %s, skipping' % mod)
            continue

        config = MODALITY_CONFIG[mod_upper]
        threshold = config.get('threshold', DEFAULT_THRESHOLD)

        logging.info('Starting %s matching (inventory-based)' % mod_upper)

        match_modality(
            adnimerge_csv=adnimerge_csv,
            output_directory=output_dir,
            threshold=threshold,
            modality=mod_upper,
            dcm_inventory=dcm_inventory,
            n_jobs=n_jobs,
            overwrite=overwrite,
            mriqc_csv=mriqc_csv,
            apoeres_csv=apoeres_csv,
            birth_dates_csv=birth_dates_csv,
            log_path=log_path,
        )


def _resolve_ucb_table(ucb_tables_dir: str, table_name: str) -> str:
    """Find UCBerkeley table (exact match or latest date-stamped)."""
    exact = os.path.join(ucb_tables_dir, table_name)
    if os.path.isfile(exact):
        return exact
    base = table_name.replace('.csv', '')
    pattern = os.path.join(ucb_tables_dir, base + '_*.csv')
    matches = sorted(glob(pattern))
    return matches[-1] if matches else exact


def run_ucberkeley_attach(modalities: list, output_dir: str,
                          ucb_tables_dir: str):
    """매칭 완료된 모달리티에 UCBerkeley PET quantification 조인"""
    logging.info('=' * 60)
    logging.info('UCBerkeley PET quantification attach')
    logging.info('=' * 60)

    for mod in modalities:
        mod_upper = mod.upper()
        if mod_upper not in UCBERKELEY_ATTACH_CONFIG:
            continue

        ucb_cfg = UCBERKELEY_ATTACH_CONFIG[mod_upper]
        matching_csv = os.path.join(output_dir, '%s_unique.csv' % mod_upper)
        ucberkeley_csv = _resolve_ucb_table(ucb_tables_dir, ucb_cfg['table'])

        attach_ucberkeley(
            matching_csv=matching_csv,
            ucberkeley_csv=ucberkeley_csv,
            modality=mod_upper,
            date_threshold=ucb_cfg.get('threshold', 30),
            ucb_date_col=ucb_cfg.get('date_col', 'SCANDATE'),
            tracer_filter=ucb_cfg.get('tracer_filter', None),
            column_prefix=ucb_cfg.get('column_prefix', ''),
        )


def main():
    parser = argparse.ArgumentParser(
        description='ADNI XML-free matching pipeline (v4)')
    parser.add_argument('--modality', type=str, default=None,
                        help='Modalities to process (comma-separated, e.g., T1,AV45,DTI)')
    parser.add_argument('--merge-only', action='store_true',
                        help='Only generate MERGED.csv from existing *_unique.csv')
    parser.add_argument('--inventory-only', action='store_true',
                        help='Only generate DCM inventory')
    parser.add_argument('--nfs-base', type=str, default=None,
                        help='NFS base path for DCM sources (default: config.NFS_BASE)')
    parser.add_argument('--skip-inventory', action='store_true', default=False,
                        help='Skip inventory build (use existing)')
    parser.add_argument('--build-inventory', action='store_true',
                        help='Force rebuild DCM inventory')
    parser.add_argument('--n-jobs', type=int, default=8,
                        help='Number of parallel jobs (default: 8)')
    parser.add_argument('--scan-workers', type=int, default=8,
                        help='PTID-level scan threads per source (default: 8, total max = 6 sources x N)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing results')
    parser.add_argument('--adnimerge', type=str, default=DEFAULT_ADNIMERGE_CSV,
                        help='Path to ADNIMERGE CSV')
    parser.add_argument('--mriqc', type=str, default=DEFAULT_MRIQC_CSV,
                        help='Path to MRIQC CSV')
    parser.add_argument('--apoeres', type=str, default=DEFAULT_APOERES_CSV,
                        help='Path to APOERES CSV')
    parser.add_argument('--birth-dates', type=str, default=DEFAULT_BIRTH_DATES_CSV,
                        help='Path to birth_dates CSV')
    parser.add_argument('--ucb-tables-dir', type=str, default=DEFAULT_UCB_TABLES_DIR,
                        help='Directory containing UCBerkeley CSV tables')
    parser.add_argument('--skip-ucberkeley', action='store_true',
                        help='Skip UCBerkeley PET quantification attach')
    args = parser.parse_args()

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'matching.log')
    setup_logger(log_path)

    logging.info('=' * 60)
    logging.info('ADNI XML-free Matching Pipeline v4')
    logging.info('=' * 60)
    logging.info('Output: %s' % args.output_dir)
    logging.info('ADNIMERGE: %s' % args.adnimerge)
    logging.info('MRIQC: %s' % args.mriqc)
    logging.info('APOERES: %s' % args.apoeres)
    logging.info('birth_dates: %s' % args.birth_dates)
    logging.info('n_jobs: %d' % args.n_jobs)

    # 사전 검사
    if not os.path.isfile(args.adnimerge):
        logging.error('ADNIMERGE CSV not found: %s' % args.adnimerge)
        logging.error('Run: python -m adni.extraction --build-adnimerge')
        sys.exit(1)

    # Merge-only 모드
    if args.merge_only:
        unique_csv_merge(args.output_dir, exclude_modalities=MERGE_EXCLUDE)
        return

    # Inventory (항상 필요 — 인벤토리 기반 매칭)
    dcm_inventory = None
    if args.inventory_only:
        run_inventory(args.output_dir, nfs_base=args.nfs_base, force_rebuild=True,
                      scan_workers=args.scan_workers)
        return

    if args.build_inventory:
        dcm_inventory = run_inventory(args.output_dir, nfs_base=args.nfs_base, force_rebuild=True,
                                      scan_workers=args.scan_workers)
    elif args.skip_inventory:
        inventory_path = os.path.join(args.output_dir, 'dcm_inventory.json')
        if os.path.isfile(inventory_path):
            dcm_inventory = load_inventory(inventory_path)
            logging.info('Loaded existing inventory: %d series' % dcm_inventory.get('metadata', {}).get('total_series', 0))
        else:
            logging.error('No existing inventory found at %s. Run without --skip-inventory.' % inventory_path)
            sys.exit(1)
    else:
        dcm_inventory = run_inventory(args.output_dir, nfs_base=args.nfs_base,
                                      scan_workers=args.scan_workers)

    # 모달리티 결정
    if args.modality:
        modalities = [m.strip() for m in args.modality.split(',')]
    else:
        modalities = list(MODALITY_CONFIG.keys())

    logging.info('Modalities: %s' % ', '.join(modalities))

    # 매칭 실행
    run_matching(
        modalities=modalities,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        overwrite=args.overwrite,
        dcm_inventory=dcm_inventory,
        adnimerge_csv=args.adnimerge,
        mriqc_csv=args.mriqc,
        apoeres_csv=args.apoeres,
        birth_dates_csv=args.birth_dates,
    )

    # UCBerkeley PET quantification attach (매칭 후, 머지 전)
    if not args.skip_ucberkeley:
        run_ucberkeley_attach(
            modalities=modalities,
            output_dir=args.output_dir,
            ucb_tables_dir=args.ucb_tables_dir,
        )

    # MERGED.csv 생성
    unique_csv_merge(args.output_dir, exclude_modalities=MERGE_EXCLUDE)

    logging.info('=' * 60)
    logging.info('Pipeline complete')
    logging.info('=' * 60)


if __name__ == '__main__':
    main()
