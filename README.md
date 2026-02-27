# res-cohort-matching

뇌영상 코호트 데이터 추출 및 DICOM 매칭 통합 파이프라인.

코호트별 원본 데이터(R 패키지, CSV 등)를 Python으로 추출하고,
NFS 서버의 DICOM 파일과 매칭하여 통합 CSV를 생성한다.

---

## 지원 코호트

| 코호트 | 상태 | 설명 |
|--------|------|------|
| **ADNI** (GO/1/2/3/4) | Active | Alzheimer's Disease Neuroimaging Initiative |
| **NACC** | Planned | National Alzheimer's Coordinating Center |

---

## 프로젝트 구조

```
res-cohort-matching/
├── src/
│   └── adni/                   # ADNI 코호트 파이프라인
│       ├── extraction/         #   Part 1: .rda → ADNIMERGE CSV 추출
│       └── matching/           #   Part 2: DICOM 매칭 파이프라인
│           └── reference/      #     레퍼런스 코드 (수정 금지)
│
├── vendor/                     # 원본 데이터 패키지 (수정 금지)
│   └── ADNIMERGE2/             #   LONI ADNIMERGE2 R 패키지
│
├── scripts/                    # 검증 및 유틸리티 스크립트
├── csv/                        # 추출된 CSV 출력
└── pyproject.toml
```

---

## Quick Start

```bash
# 설치
pip install -e .

# ADNI: .rda → CSV 추출 + ADNIMERGE 구축
adni-extract --all

# ADNI: DICOM 매칭 파이프라인
adni-match --skip-inventory --modality T1,AV45
```

---

## 코호트별 문서

- **ADNI**: [`src/adni/README.md`](src/adni/README.md) - 데이터 흐름, CLI 옵션, 모달리티 목록
- **ADNI Extraction**: [`src/adni/extraction/README.md`](src/adni/extraction/README.md) - ADNIMERGE 구축 상세

---

## 의존성

- Python >= 3.9
- numpy, pandas, pydicom, pyreadr, joblib
