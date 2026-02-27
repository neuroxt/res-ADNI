# ADNI_match

ADNI (Alzheimer's Disease Neuroimaging Initiative) GO/1/2/3/4 전체 코호트의
임상 데이터 추출 및 DICOM 영상 매칭 파이프라인.

LONI에서 제공하는 ADNIMERGE2 R 패키지의 `.rda` 데이터를 Python으로 추출하고,
NFS 서버의 DICOM 파일과 매칭하여 통합 CSV(`MERGED.csv`)를 생성한다.

---

## 프로젝트 구조

```
res-ADNI/
├── adni/                # Python 패키지
│   ├── extraction/      #   Part 1: .rda → ADNIMERGE CSV 추출
│   └── matching/        #   Part 2: DICOM 매칭 파이프라인
│       └── reference/   #     레퍼런스 코드 (XML 기반, 수정 금지)
├── vendor/ADNIMERGE2/   # LONI ADNIMERGE2 R 패키지 (원본, 수정 금지)
└── scripts/             # 검증 및 유틸리티 스크립트
```

---

## 데이터 흐름

```
vendor/ADNIMERGE2/data/*.rda (217개)
    │
    ▼  adni.extraction (rda_converter)
csv/tables/*.csv (217개 1:1 변환, MRIQC.csv·APOERES.csv 포함)
    │
    ▼  adni.extraction (build_adnimerge)
ADNIMERGE_{DATE}.csv (23,479행 × 132열)
    │
    ▼  adni.matching
DCM 인벤토리 + ADNIMERGE 매칭
    │
    ▼
{MOD}_all.csv → {MOD}_unique.csv → MERGED.csv (13,042행 × 782열)
```

---

## 1. ADNIMERGE2 (LONI R 패키지)

LONI ATRI Biostatistics에서 제공하는 R 데이터 패키지.
217개 `.rda` 파일에 ADNI 전체 코호트의 임상, 인지검사, 바이오마커, MRI 용적, PET 데이터가 포함되어 있다.

- **출처**: https://atri-biostats.github.io/ADNIMERGE2
- **버전**: 0.1.1 (2026-01-05)
- **주의**: 이 디렉토리는 원본 그대로 유지하며 수정하지 않는다.

---

## 2. adni.extraction (ADNIMERGE CSV 추출)

ADNIMERGE2 R 패키지의 빌드 로직을 Python/pandas로 1:1 이식한 패키지.
`.rda` 원본에서 직접 ADNIMERGE CSV를 생성한다.

### 모듈 구성

| 파일 | 역할 |
|------|------|
| `build_adnimerge.py` | 12단계 빌드 프로세스로 ADNIMERGE CSV 생성 |
| `rda_converter.py` | .rda → CSV 일괄 변환 (217개 테이블) |
| `compare_ref.py` | 레퍼런스 CSV 대비 비교 검증 |
| `cli.py` | CLI 인터페이스 |

### 12단계 빌드 프로세스

| 단계 | 내용 | 주요 소스 |
|------|------|-----------|
| 1 | 코어 데이터셋 로드 | ADSL, REGISTRY, DXSUM, PTDEMOG 등 22개 |
| 2 | 베이스 프레임 생성 | REGISTRY (VISCODE 표준화) |
| 3 | 인구통계 + EXAMDATE_bl | PTDEMOG, APOERES, ADSL |
| 4 | 진단 (DX, DX_bl) | DXSUM + ARM.rda (EMCI/LMCI/SMC 유지) |
| 5 | 인지검사 점수 | ADAS, MMSE, CDR, MOCA, ECOG 등 |
| 6 | CSF 바이오마커 | UPENNBIOMK_MASTER + ROCHE_ELECSYS |
| 7 | MRI 용적 | UCSFFSX51 (primary) + UCSDVOL (Ventricles) |
| 8 | PET 데이터 | UCBERKELEYFDG_8mm, AMY_6MM, TAU_6MM |
| 9 | 혈장 바이오마커 | UPenn, C2N, Fujirebio, Quanterix, UGOT |
| 10 | 파생 변수 | Baseline coalesce, Month, SITE, mPACC |
| 11 | 132개 컬럼 스키마 선택 | — |
| 12 | CSV 저장 | ADNIMERGE_{DATE}.csv |

### 실행

```bash
# 전체 실행 (rda 변환 + ADNIMERGE 빌드)
python -m adni.extraction

# 개별 단계
python -m adni.extraction --build-adnimerge
python -m adni.extraction --convert-all

# 옵션
python -m adni.extraction --rda-dir /path/to/ADNIMERGE2/data --output-dir /path/to/csv --date 260213 -v
```

### 출력

| 파일 | 설명 |
|------|------|
| `ADNIMERGE_{DATE}.csv` | 23,479행 × 132열, 4,498 피험자 |
| `tables/*.csv` | 217개 .rda 1:1 CSV 변환 (MRIQC.csv, APOERES.csv 등 포함) |

상세 설명: [`adni/extraction/README.md`](adni/extraction/README.md)

---

## 3. adni.matching (DICOM 매칭 파이프라인)

XML 메타데이터를 완전히 제거하고, 경로 파싱 + ADNIMERGE + MRIQC + pydicom으로
DICOM 영상을 임상 데이터와 매칭하는 v4 파이프라인.

### 모듈 구성

| 파일 | 역할 |
|------|------|
| `config.py` | 모달리티별 설정, 경로, 상수 |
| `inventory.py` | DCM 디렉토리 스캔 → 모달리티별 인벤토리 |
| `matcher.py` | 이미지-ADNIMERGE 매칭 (핵심 로직) |
| `merge.py` | `*_unique.csv` → `MERGED.csv` 병합 |
| `cli.py` | CLI 오케스트레이션 |
| `utils.py` | 로깅, 경로 추출, DICOM 유틸 |

### 매칭 로직

1. DCM 인벤토리에서 모달리티별 피험자/시리즈 수집
2. 경로에서 촬영일(AQUDATE), ImageUID, SeriesUID 추출
3. ADNIMERGE에서 가장 가까운 방문(EXAMDATE) 매칭
4. `VISCODE_FIX = 촬영일 - EXAMDATE_bl` → 표준 방문 시점 매핑 (m000, m006, m012...)
5. MRIQC에서 프로토콜 정보, DCM에서 TE/TR/TI 추출
6. `_all.csv` (전체) → `_unique.csv` (PTID×VISCODE 중복 제거) → `MERGED.csv`

### 지원 모달리티

**매칭 완료 (12개)**: T1, AV45_8MM/6MM, AV1451_8MM/6MM, FBB_6MM, FLAIR, T2_FSE/TSE/STAR, T2_3D, MK6240_6MM

**미실행**: DTI, DTI_MB, FMRI, HIPPO, ASL, NAV4694_6MM, PI2620_6MM

### 실행

```bash
# 전체 파이프라인
python -m adni.matching --adnimerge /path/to/ADNIMERGE.csv --output /path/to/output

# 특정 모달리티만
python -m adni.matching --modality T1,AV45_6MM

# 병합만 (이미 매칭된 CSV가 있을 때)
python -m adni.matching --merge-only --output /path/to/output
```

### 출력

| 파일 | 설명 |
|------|------|
| `{MOD}_all.csv` | 모달리티별 전체 매칭 결과 (중복 포함) |
| `{MOD}_unique.csv` | PTID×VISCODE 중복 제거 (error 행 제외) |
| `MERGED.csv` | 전체 모달리티 통합 (13,042행 × 782열, 3,278 피험자) |

### `_all.csv` vs `_unique.csv`

- **`_all.csv`**: 매칭된 모든 결과. 동일 피험자-방문에 여러 스캔이 있으면 중복 행 포함, VISCODE 매핑 실패(`error`) 행 포함.
- **`_unique.csv`**: PTID × VISCODE_FIX 조합당 1행만 유지 (중복 시 마지막 항목, error 행 제외). MERGED.csv는 이 파일들을 병합하여 생성.

### 레퍼런스 코드 (`reference/`)

`ADNI.py`와 `params.py`는 XML 기반의 기존 매칭 코드로, 로직 참조용으로만 보관한다. 수정하지 않는다.

---

## 4. scripts (유틸리티)

| 파일 | 역할 |
|------|------|
| `compare_merged.py` | 생성된 MERGED.csv를 레퍼런스와 비교 검증 |
| `remap_proc_viscode.py` | N4/VA/FastSurfer 전처리 결과를 VISCODE 기반으로 재구성 |
| `reorganize_proc_t1.py` | PROC/T1 디렉토리를 PTID/VISCODE_FIX 구조로 재배치 |

---

## 5. Validation

v4 파이프라인(XML 제거, Python 이식) 결과를 기존 XML 기반 매칭 결과(ref)와 비교 검증하였다.

### 5.1. ADNI1/GO/2/3 매칭 검증

| Metric | Ref | New |
|--------|-----|-----|
| 행 수 | 11,710 | 11,744 |
| 피험자 | 2,631 | 2,632 |
| 공통 행 (PTID+VISCODE) | 11,692 | (ref 커버리지 99.8%) |

**모달리티별 ImageUID 일치율 (공통 행 기준)**

| 모달리티 | 비교 건수 | 일치 | 일치율 |
|----------|-----------|------|--------|
| T1 | 11,021 | 11,018 | **100.0%** |
| AV45_8MM | 3,154 | 3,154 | **100.0%** |
| AV45_6MM | 2,860 | 2,860 | **100.0%** |
| AV1451_8MM | 1,570 | 1,570 | **100.0%** |
| AV1451_6MM | 1,532 | 1,532 | **100.0%** |
| FBB_6MM | 586 | 586 | **100.0%** |
| FLAIR | 3,359 | 3,358 | **100.0%** |
| T2_FSE | 2,094 | 2,084 | 99.5% |
| T2_TSE | 1,705 | 1,703 | 99.9% |
| T2_STAR | 6,576 | 6,571 | 99.9% |

촬영일(AQUDATE)은 전 모달리티 **100.0%** 일치.

### 5.2. ADNI4 매칭 검증

VISCODE 체계가 다르므로 (ref: `4_sc/4_init/4_m12`, new: `m000/m003/m012`) PTID+AQUDATE 기준으로 비교.

| 모달리티 | 공통 건수 | 일치율 | 비고 |
|----------|-----------|--------|------|
| AV45_6MM | 398 | **100.0%** | — |
| AV1451_6MM | 549 | **100.0%** | — |
| FBB_6MM | 305 | 99.7% | — |
| MK6240_6MM | 108 | **100.0%** | — |
| T2_3D | 891 | **100.0%** | — |
| FLAIR | 949 | 96.9% | VISCODE 차이 영향 |
| T1 | 944 | 89.2% | VISCODE 차이 + 다중 시리즈 선택 |
| T2_STAR | 824 | 26.6% | 세션당 최대 10 시리즈, VISCODE 차이 증폭 |

**Ref-only PTID 229명 원인**: 224명은 NFS에 DCM 파일 자체가 없음 (ref는 XML 메타데이터 기반이라 물리적 파일 없이도 행 생성 가능). 파이프라인 버그가 아닌 **데이터 가용성 차이**.

### 5.3. ADNI4 VISCODE Error 분석

ADNI3/4에서 MRI 촬영일과 ADNIMERGE 방문일 간 차이가 threshold(180일)를 초과하는 케이스 발생.

| 프로토콜 | Error 스캔 | 전체 스캔 | Error율 |
|----------|-----------|-----------|---------|
| ADNI1 | 2 | 9,115 | 0.02% |
| ADNIGO | 0 | 1,193 | 0.00% |
| ADNI2 | 2 | 9,313 | 0.02% |
| **ADNI3** | **16** | **2,650** | **0.6%** |
| **ADNI4** | **8** | **1,149** | **0.7%** |

**원인**: Screening MRI가 enrollment 전에 촬영되었으나, ADNIMERGE에는 enrollment 이후 방문만 기록. ADNI4 error 8건 모두 촬영일이 ADNIMERGE 첫 방문보다 186~261일 전.

**AQUDATE-EXAMDATE gap 분포 (정상 매칭 스캔)**:

| 프로토콜 | Median gap | >90일 비율 |
|----------|-----------|-----------|
| ADNI1 | 3일 | 0.5% |
| ADNI2 | 0일 | 0.8% |
| ADNI3 | 8일 | 4.3% |
| ADNI4 | 19일 | 8.2% |

ADNI4로 갈수록 gap 증가 — 원격/분산 촬영 증가, screening-enrollment 간 지연.

### 5.4. ADNIMERGE 임상 변수 비교

LONI 제공 레퍼런스 CSV(`ADNIMERGE_240821.csv`, 16,421행)와 Python 빌드(`ADNIMERGE_260213.csv`, 23,479행) 비교.

#### (A) 핵심 임상 변수 (99~100% 일치)

| 변수 | 비교 건수 | 일치율 |
|------|----------|--------|
| SITE, PTEDUCAT, PTETHCAT, PTMARRY | 16,411 | **100%** |
| RAVLT_immediate, RAVLT_learning, DIGITSCOR | 3,800~16,410 | **100%** |
| CDRSB, FAQ, TRABSCOR, LDELTOTAL | 9,441~11,745 | **99.9%+** |
| DX (방문별 진단) | 11,457 | **99.9%** |
| MMSE | 11,468 | **98.5%** |
| MOCA | 2,447 | **90.0%** |

MMSE/MOCA 불일치는 .rda 소스 데이터 업데이트(2024→2026)에 의한 것.

#### (B) DX_bl — ARM.rda 연동 (91.3% 일치)

- ADNI1/GO/2: ARM.rda enrollment group 기준 (EMCI/LMCI/SMC/CN/AD 유지)
- ADNI3/4: ADSL.DX + DXSUM.DIAGNOSIS 기준 (CN/MCI/AD만)
- 잔여 불일치 8.7%는 ADNI3의 ARM.rda 미포함에 기인

#### (C) 데이터 소스 변경 (의도된 차이)

| 변수 | 일치율 | Ref 소스 | New 소스 | 비고 |
|------|--------|----------|----------|------|
| Hippocampus | ~93% | UCSFFSX (FS 4.x) | UCSFFSX51 (FS 5.1) | r=0.998 |
| ICV | ~93% | UCSFFSX | UCSFFSX51 | r=0.997 |
| WholeBrain, Entorhinal 등 | ~0% | UCSFFSX | UCSFFSX51 | atlas/ROI 정의 변경 |
| Ventricles | ~0% | UCSFFSX | UCSDVOL | 다른 파이프라인, r=0.995 |
| AV45 (amyloid PET) | ~24% | 8mm SUVR | AMY_6MM SUMMARY_SUVR | r=0.998, 해상도 변경 |
| FDG (FDG PET) | ~6% | UCBERKELEYFDG (구) | UCBERKELEYFDG_8mm MetaROI | r=0.847, 파이프라인 세대교체 |
| ABETA/TAU/PTAU (CSF) | ~7-10% | Luminex 단독 | MASTER + Elecsys 병합 | 플랫폼 교체 |
| ADAS11/13 | ~69% | 정수 | 소수점 (Q4 sub-scoring) | r=0.999, round 시 99.7% |
| AGE | ~60% | ADSL AGE | ADSL AGE | r≈1.000, 소수점 반올림 차이 |

MRI 용적 소스 선택 근거: UCSFFSX51(4,896행, 1,067명)이 coverage 최대. UCSDVOL(3,128행, 811명)은 Ventricles(lateral ventricle)만 사용 — FreeSurfer에는 lateral ventricle 컬럼이 없기 때문.

CSF 소스 선택 근거: ADNI3 이후 Elecsys로 전환되어 Luminex 단독 사용 시 최신 데이터 누락. ADNIMERGE2 패키지 설계에 따라 MASTER + ELECSYS 병합 사용. ([Shaw et al. 2019](https://www.nature.com/articles/s41598-019-54204-z))

### 5.5. 검증 요약

| 범주 | 결과 |
|------|------|
| **영상 매칭 (ADNI1/GO/2/3)** | T1/PET **100%**, T2 99.5~99.9% |
| **영상 매칭 (ADNI4)** | PET/T2_3D **100%**, T1 89.2% (VISCODE 체계 차이) |
| **핵심 임상 변수** | 인구통계 **100%**, 인지검사 **98.5~100%** |
| **DX_bl** | **91.3%** (ARM.rda 연동, ADNI3 ARM 부재 8.7%) |
| **MRI 용적** | Hippocampus/ICV **~93%** (UCSFFSX51 전환) |
| **바이오마커** | 의도된 소스 변경 (Elecsys, 6mm PET 등) |

---

## 의존성

```
pip install -r requirements.txt
```

| 패키지 | 용도 |
|--------|------|
| numpy | 수치 연산 |
| pandas | 데이터 처리 |
| pydicom | DICOM 메타데이터 추출 |
| pyreadr | .rda 파일 읽기 |
| joblib | 병렬 처리 |
