"""
inventory.py — NII 인벤토리 빌더

NFS NII 폴더를 스캔하여 BID/session/modality 구조의 인벤토리 JSON 생성.
ADNI inventory.py 패턴 차용하되 NIfTI + JSON sidecar 기반으로 대폭 단순화.
"""

import os
import json
import logging

from .config import (
    NFS_NII_BASE, BID_RE, MODALITY_CONFIG, NII_PATTERN,
    JSON_MR_FIELDS, JSON_PET_FIELDS,
)


def _read_json_sidecar(json_path: str) -> dict:
    """JSON sidecar 파일 읽기. 실패 시 빈 dict 반환."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.debug('Failed to read JSON sidecar %s: %s' % (json_path, e))
        return {}


def _extract_json_fields(json_data: dict, modality_type: str) -> dict:
    """JSON sidecar에서 모달리티 타입에 따른 필드 추출."""
    field_map = JSON_MR_FIELDS if modality_type == 'MR' else JSON_PET_FIELDS
    result = {}
    for json_key, out_name in field_map.items():
        val = json_data.get(json_key, '')
        if val != '':
            result[out_name] = str(val)
        else:
            result[out_name] = ''
    return result


def _find_primary_nii(modality_dir: str) -> tuple:
    """모달리티 폴더에서 primary .nii.gz 파일과 JSON sidecar 찾기.

    Returns:
        (nii_path, json_path) — 없으면 ('', '')
    """
    nii_path = ''
    json_path = ''
    try:
        for entry in os.scandir(modality_dir):
            name = entry.name
            if name.endswith('.nii.gz') and NII_PATTERN.match(name):
                nii_path = entry.path
                # 대응 JSON sidecar
                json_candidate = entry.path.replace('.nii.gz', '.json')
                if os.path.isfile(json_candidate):
                    json_path = json_candidate
                break
    except OSError as e:
        logging.debug('Failed to scan %s: %s' % (modality_dir, e))

    # NII_PATTERN에 매칭 안 되면 아무 .nii.gz라도 찾기
    if not nii_path:
        try:
            for entry in os.scandir(modality_dir):
                if entry.name.endswith('.nii.gz'):
                    nii_path = entry.path
                    json_candidate = entry.path.replace('.nii.gz', '.json')
                    if os.path.isfile(json_candidate):
                        json_path = json_candidate
                    break
        except OSError:
            pass

    return nii_path, json_path


def _scan_bid(bid_dir: str, bid: str, modality_folders: dict) -> dict:
    """단일 BID 폴더 스캔 → {session: {modality: record}}."""
    sessions = {}
    try:
        for session_entry in sorted(os.scandir(bid_dir), key=lambda e: e.name):
            if not session_entry.is_dir():
                continue
            session = session_entry.name  # e.g., '002', '004'
            session_data = {}

            try:
                for mod_entry in os.scandir(session_entry.path):
                    if not mod_entry.is_dir():
                        continue
                    folder_name = mod_entry.name  # e.g., 'T1', 'FBP'

                    # folder_name → modality key 매핑
                    mod_key = modality_folders.get(folder_name)
                    if mod_key is None:
                        continue

                    nii_path, json_path = _find_primary_nii(mod_entry.path)
                    if not nii_path:
                        continue

                    mod_config = MODALITY_CONFIG[mod_key]
                    json_meta = {}
                    if json_path:
                        raw_json = _read_json_sidecar(json_path)
                        json_meta = _extract_json_fields(raw_json, mod_config['type'])

                    record = {
                        'session': session,
                        'nii_path': nii_path,
                        'json_path': json_path,
                        'json_meta': json_meta,
                    }
                    session_data[mod_key] = record

            except OSError as e:
                logging.debug('Failed to scan session %s/%s: %s' % (bid, session, e))

            if session_data:
                sessions[session] = session_data

    except OSError as e:
        logging.debug('Failed to scan BID %s: %s' % (bid, e))

    return sessions


def build_inventory(nfs_base: str = None) -> dict:
    """NII 폴더 스캔 → 인벤토리 dict.

    Args:
        nfs_base: NII 루트 경로 (기본: NFS_NII_BASE)

    Returns:
        inventory dict with 'metadata', 'by_modality', 'by_bid_session' keys.
    """
    if nfs_base is None:
        nfs_base = NFS_NII_BASE

    logging.info('Building NII inventory from %s' % nfs_base)

    # folder_name → modality key 역매핑
    modality_folders = {}
    for mod_key, cfg in MODALITY_CONFIG.items():
        modality_folders[cfg['folder']] = mod_key

    by_modality = {mod: {} for mod in MODALITY_CONFIG}
    by_bid_session = {}
    total_files = 0

    # 1단계: BID 목록 수집
    try:
        bid_entries = sorted(
            [e for e in os.scandir(nfs_base) if e.is_dir() and BID_RE.match(e.name)],
            key=lambda e: e.name,
        )
    except OSError as e:
        logging.error('Cannot scan NII base %s: %s' % (nfs_base, e))
        return {'metadata': {}, 'by_modality': {}, 'by_bid_session': {}}

    logging.info('Found %d BID directories' % len(bid_entries))

    # 2단계: 각 BID 스캔
    for bid_entry in bid_entries:
        bid = bid_entry.name
        sessions = _scan_bid(bid_entry.path, bid, modality_folders)

        if not sessions:
            continue

        bid_session_map = {}
        for session, mod_records in sessions.items():
            mods_in_session = []
            for mod_key, record in mod_records.items():
                # by_modality 채우기
                if bid not in by_modality[mod_key]:
                    by_modality[mod_key][bid] = []
                by_modality[mod_key][bid].append(record)
                mods_in_session.append(mod_key)
                total_files += 1

            bid_session_map[session] = sorted(mods_in_session)

        by_bid_session[bid] = bid_session_map

    # 메타데이터 집계
    modality_counts = {}
    for mod, bid_dict in by_modality.items():
        n_bids = len(bid_dict)
        n_files = sum(len(recs) for recs in bid_dict.values())
        if n_bids > 0:
            modality_counts[mod] = {'bids': n_bids, 'files': n_files}

    metadata = {
        'version': 1,
        'nfs_base': nfs_base,
        'total_files': total_files,
        'total_bids': len(by_bid_session),
        'modality_counts': modality_counts,
    }

    inventory = {
        'metadata': metadata,
        'by_modality': by_modality,
        'by_bid_session': by_bid_session,
    }

    logging.info('Inventory complete: %d BIDs, %d files across %d modalities' % (
        len(by_bid_session), total_files, len(modality_counts),
    ))
    for mod, counts in sorted(modality_counts.items()):
        logging.info('  %s: %d BIDs, %d files' % (mod, counts['bids'], counts['files']))

    return inventory


def save_inventory(inventory: dict, output_path: str):
    """인벤토리를 JSON 파일로 저장."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(inventory, f, indent=2)
    logging.info('Inventory saved to %s' % output_path)


def load_inventory(input_path: str) -> dict:
    """JSON 인벤토리 파일 로드."""
    with open(input_path, 'r') as f:
        inventory = json.load(f)
    logging.info('Inventory loaded from %s (%d BIDs)' % (
        input_path, inventory.get('metadata', {}).get('total_bids', 0),
    ))
    return inventory
