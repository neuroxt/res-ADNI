"""
cli.py — A4/LEARN 파이프라인 CLI

Usage:
    python -m a4 [OPTIONS]

Examples:
    python -m a4                          # 전체 파이프라인
    python -m a4 --inventory-only         # NII inventory만 생성
    python -m a4 --clinical-only          # 임상 테이블만 생성
    python -m a4 --merge-only             # 기존 *_unique.csv → MERGED.csv
    python -m a4 --modality T1,FBP        # 특정 모달리티만
    python -m a4 --include-screen-fail    # amyloidNE 포함
"""

import os
import sys
import argparse
import logging

from .config import (
    NFS_NII_BASE, NFS_METADATA_BASE, NFS_CLINICAL_BASE, OUTPUT_BASE,
    MODALITY_CONFIG, MERGE_EXCLUDE, LOG_FORMAT,
)
from .inventory import build_inventory, save_inventory, load_inventory
from .clinical import build_clinical_table
from .pipeline import run_pipeline, unique_csv_merge

DEFAULT_OUTPUT_DIR = os.path.join(OUTPUT_BASE, 'A4_matching_v1')
INVENTORY_FILENAME = 'nii_inventory.json'


def setup_logger(log_path: str, level=logging.DEBUG):
    """로거 설정 (파일 + 콘솔)."""
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT)

    # 기존 핸들러 중복 방지
    has_fh = any(
        isinstance(h, logging.FileHandler) and
        getattr(h, 'baseFilename', None) == os.path.abspath(log_path)
        for h in logger.handlers
    )
    has_sh = any(
        isinstance(h, logging.StreamHandler) and
        not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )

    if not has_fh:
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if not has_sh:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)


def run_inventory_step(output_dir: str, nfs_base: str = None,
                       force_rebuild: bool = False) -> dict:
    """인벤토리 빌드 또는 로드."""
    inv_path = os.path.join(output_dir, INVENTORY_FILENAME)

    if os.path.isfile(inv_path) and not force_rebuild:
        logging.info('Loading existing inventory: %s' % inv_path)
        return load_inventory(inv_path)

    inventory = build_inventory(nfs_base=nfs_base)
    save_inventory(inventory, inv_path)
    return inventory


def run_clinical_step(metadata_dir: str = None, clinical_dir: str = None,
                      include_screen_fail: bool = False,
                      output_dir: str = None) -> 'pd.DataFrame':
    """임상 테이블 빌드 + CSV 저장."""
    import pandas as pd

    clinical = build_clinical_table(
        metadata_dir=metadata_dir,
        clinical_dir=clinical_dir,
        include_screen_fail=include_screen_fail,
    )

    if not clinical.empty and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'clinical_table.csv')
        clinical.to_csv(csv_path)
        logging.info('Clinical table saved to %s' % csv_path)

    return clinical


def parse_args(argv=None):
    """CLI 인수 파싱."""
    parser = argparse.ArgumentParser(
        description='A4/LEARN NII inventory + clinical data → MERGED.csv pipeline',
    )

    # 모드 선택
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--inventory-only', action='store_true',
                      help='NII inventory만 생성')
    mode.add_argument('--clinical-only', action='store_true',
                      help='임상 테이블만 생성')
    mode.add_argument('--merge-only', action='store_true',
                      help='기존 *_unique.csv에서 MERGED.csv만 재생성')

    # 경로
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                        help='출력 디렉토리 (기본: %s)' % DEFAULT_OUTPUT_DIR)
    parser.add_argument('--nii-base', default=NFS_NII_BASE,
                        help='NII 루트 경로 (기본: %s)' % NFS_NII_BASE)
    parser.add_argument('--metadata-dir', default=NFS_METADATA_BASE,
                        help='metadata 디렉토리 (기본: %s)' % NFS_METADATA_BASE)
    parser.add_argument('--clinical-dir', default=NFS_CLINICAL_BASE,
                        help='Clinical 디렉토리 (기본: %s)' % NFS_CLINICAL_BASE)

    # 옵션
    parser.add_argument('--modality', type=str, default=None,
                        help='처리할 모달리티 (쉼표 구분, e.g., T1,FBP)')
    parser.add_argument('--overwrite', action='store_true',
                        help='기존 결과 덮어쓰기')
    parser.add_argument('--build-inventory', action='store_true',
                        help='인벤토리 강제 재생성')
    parser.add_argument('--include-screen-fail', action='store_true',
                        help='amyloidNE (스크리닝 탈락) 포함 (기본: 제외)')
    parser.add_argument('--group', type=str, default=None,
                        help='특정 Research Group만 처리 (쉼표 구분)')

    return parser.parse_args(argv)


def main(argv=None):
    """CLI 메인 진입점."""
    args = parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'a4_pipeline.log')
    setup_logger(log_path)

    logging.info('=' * 60)
    logging.info('A4/LEARN Pipeline Start')
    logging.info('Output: %s' % args.output_dir)
    logging.info('=' * 60)

    # 모달리티 목록 파싱
    modalities = None
    if args.modality:
        modalities = [m.strip().upper() for m in args.modality.split(',')]
        # config에 없는 모달리티 경고
        for m in modalities:
            if m not in MODALITY_CONFIG:
                logging.warning('Unknown modality: %s (not in MODALITY_CONFIG)' % m)

    # --merge-only
    if args.merge_only:
        unique_csv_merge(args.output_dir, exclude_modalities=MERGE_EXCLUDE)
        return

    # --inventory-only
    if args.inventory_only:
        run_inventory_step(args.output_dir, nfs_base=args.nii_base,
                           force_rebuild=args.build_inventory)
        return

    # --clinical-only
    if args.clinical_only:
        run_clinical_step(
            metadata_dir=args.metadata_dir,
            clinical_dir=args.clinical_dir,
            include_screen_fail=args.include_screen_fail,
            output_dir=args.output_dir,
        )
        return

    # 전체 파이프라인
    # 1. Inventory
    inventory = run_inventory_step(
        args.output_dir, nfs_base=args.nii_base,
        force_rebuild=args.build_inventory,
    )

    # 2. Clinical table
    clinical = run_clinical_step(
        metadata_dir=args.metadata_dir,
        clinical_dir=args.clinical_dir,
        include_screen_fail=args.include_screen_fail,
        output_dir=args.output_dir,
    )

    if clinical.empty:
        logging.error('Clinical table is empty — aborting pipeline')
        sys.exit(1)

    # 3. Pipeline (inventory + clinical → modality CSVs → MERGED.csv)
    run_pipeline(
        inventory=inventory,
        clinical=clinical,
        output_dir=args.output_dir,
        modalities=modalities,
        overwrite=args.overwrite,
    )

    logging.info('Pipeline complete')


if __name__ == '__main__':
    main()
