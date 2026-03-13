"""
cli.py — adni.extraction CLI 진입점

사용법:
    python -m adni.extraction [OPTIONS]

    --convert-all        217개 .rda -> CSV 전체 변환
    --build-adnimerge    ADNIMERGE_{DATE}.csv 구축
    --build-ucberkeley   UCBERKELEY PET CSVs 구축 (FDG, AMY, TAU, TAUPVC)
    --build-birth-dates  birth_dates.csv 생성 (PTDEMOG PTDOB → est_birth_date)
    --all                전부 실행 (기본)
    --rda-dir DIR        .rda 소스 디렉토리 (기본: vendor/ADNIMERGE2/data)
    --output-dir DIR     출력 디렉토리 (기본: csv/)
    --date DATE          출력 날짜 문자열 YYMMDD (기본: 오늘)
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# 프로젝트 루트 (src/adni/extraction/cli.py → extraction → adni → src → project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

from adni.config import ADNI_DEMO, ADNI_TABLES

DEFAULT_RDA_DIR = os.path.join(PROJECT_ROOT, 'vendor', 'ADNIMERGE2', 'data')
DEFAULT_OUTPUT_DIR = ADNI_DEMO
DEFAULT_TABLES_DIR = ADNI_TABLES


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def main():
    parser = argparse.ArgumentParser(
        description='ADNIMERGE2 .rda -> CSV conversion (Python replacement for R scripts)')

    parser.add_argument('--convert-all', action='store_true',
                        help='Convert all 217 .rda files to CSV')
    parser.add_argument('--build-adnimerge', action='store_true',
                        help='Build ADNIMERGE_{DATE}.csv')
    parser.add_argument('--build-ucberkeley', action='store_true',
                        help='Build UCBERKELEY PET CSVs (FDG, AMY, TAU, TAUPVC)')
    parser.add_argument('--build-birth-dates', action='store_true',
                        help='Build birth_dates.csv from PTDEMOG PTDOB')
    parser.add_argument('--all', action='store_true',
                        help='Run all steps (default if no flags given)')
    parser.add_argument('--rda-dir', type=str, default=DEFAULT_RDA_DIR,
                        help='Path to .rda source directory (default: ADNIMERGE2/data)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory (default: csv/)')
    parser.add_argument('--date', type=str, default=None,
                        help='Date string YYMMDD for output filenames (default: today)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose logging')

    args = parser.parse_args()
    setup_logging(args.verbose)

    # If no specific action requested, run all
    run_all = args.all or not (args.convert_all or args.build_adnimerge
                                or args.build_ucberkeley or args.build_birth_dates)

    date_str = args.date or datetime.now().strftime('%y%m%d')
    tables_dir = os.path.join(args.output_dir, 'tables')

    logging.info('=' * 60)
    logging.info('adni.extraction — .rda -> CSV Conversion')
    logging.info('=' * 60)
    logging.info('RDA dir: %s', args.rda_dir)
    logging.info('Output dir: %s', args.output_dir)
    logging.info('Date: %s', date_str)

    if not os.path.isdir(args.rda_dir):
        logging.error('RDA directory not found: %s', args.rda_dir)
        sys.exit(1)

    # Step 1: Convert all .rda -> individual CSVs
    if run_all or args.convert_all:
        from adni.extraction.rda_converter import convert_all_rda, print_report
        logging.info('')
        logging.info('--- Converting all .rda files ---')
        results = convert_all_rda(args.rda_dir, tables_dir)
        print_report(results)

    # Step 2: Build ADNIMERGE CSV
    if run_all or args.build_adnimerge:
        from adni.extraction.build_adnimerge import build_adnimerge
        logging.info('')
        logging.info('--- Building ADNIMERGE CSV ---')
        build_adnimerge(args.rda_dir, args.output_dir, date_str)

    # Step 3: Build UCBERKELEY PET CSVs
    if run_all or args.build_ucberkeley:
        from adni.extraction.build_adnimerge import build_all_ucberkeley
        logging.info('')
        logging.info('--- Building UCBERKELEY PET CSVs ---')
        build_all_ucberkeley(args.rda_dir, args.output_dir, date_str)

    # Step 4: Build birth_dates.csv
    if run_all or args.build_birth_dates:
        from adni.extraction.build_adnimerge import build_birth_dates
        logging.info('')
        logging.info('--- Building birth_dates.csv ---')
        build_birth_dates(args.rda_dir, args.output_dir)

    logging.info('')
    logging.info('=' * 60)
    logging.info('All done!')
    logging.info('=' * 60)


if __name__ == '__main__':
    main()
