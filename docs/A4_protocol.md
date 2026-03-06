# A4/LEARN Study Protocol & Data Reference

## 1. Study Overview

### A4 (Anti-Amyloid Treatment in Asymptomatic Alzheimer's)
- **Phase**: 3, randomized, double-blind, placebo-controlled
- **Drug**: Solanezumab (anti-amyloid monoclonal antibody, Eli Lilly)
- **Population**: Cognitively normal older adults (65-85y) with elevated brain amyloid
- **Timeline**: 2014-2023 (enrollment 2014-2017, primary endpoint 240 weeks)
- **Sites**: 67 sites across US, Canada, Australia, Japan
- **Registry**: NCT02008357

### LEARN (Longitudinal Evaluation of Amyloid Risk and Neurodegeneration)
- **Purpose**: Parallel observational study for amyloid-negative controls
- **Sponsor**: Alzheimer's Association + GHR Foundation (separate from A4)
- **Protocol**: Identical imaging/cognitive assessment as A4
- **Identification**: SUBJINFO.LRNFLGSNM = 'Y'

### Primary Outcome
Solanezumab did **not** significantly slow cognitive decline or reduce amyloid plaques in cognitively normal amyloid-positive individuals (JAMA Neurology 2020).

---

## 2. Cohort Structure

```
~6,763 screened
    |
  Cognitive assessment (CDR=0, MMSE 25-30, no MCI)
    |
  Amyloid PET (18F-Florbetapir, SUVr >= 1.15 = positive)
    |
  +-- Positive (1,323) -----> A4 Trial (randomized) --> amyloidE
  |                            solanezumab vs placebo
  |
  +-- Negative (3,163+)
        |
        +-- LEARN enrolled (567) --> LEARN amyloidNE
        |    Same imaging protocol, longitudinal observation
        |
        +-- Not enrolled (2,596) --> amyloidNE
             PET screening only, no MRI, no follow-up
```

| Research Group | N | Description | Imaging |
|---|---|---|---|
| **amyloidE** | 1,323 | Amyloid+ → A4 randomized | PET + full MRI + tau PET subset (392) |
| **LEARN amyloidNE** | 567 | Amyloid- → LEARN observational | PET + full MRI + tau PET subset (55) |
| **amyloidNE** | 2,596 | Amyloid- → screen fail | **PET only** (no MRI, no follow-up) |

---

## 3. Imaging Protocol

### MRI (3T only)
| Modality | NII Folder | Description | De-identification |
|---------|-----------|-------------|-------------------|
| T1 | T1/ | MPRAGE structural | mri_reface |
| FLAIR | FLAIR/ | Fluid-attenuated inversion recovery | mri_reface |
| T2_SE | T2_SE/ | T2 Spin Echo | mri_reface |
| T2_star | T2_star/ | T2* GRE (SWI) | — |
| fMRI_rest | fMRI_rest/ | Resting-state BOLD | — |
| b0CD | b0CD/ | B0/DWI raw diffusion | — |
| DWI | DWI/ | ADC/trace derived (**excluded from MERGE**) | — |

**Volumetrics**: NeuroQuant (CorTechs Labs) — 53 FreeSurfer regions, VISCODE=4 only.

### PET
| Tracer | NII Folder | Target | N (subjects) | Eligibility |
|--------|-----------|--------|---------------|-------------|
| Florbetapir (AV45/FBP) | FBP/ | Amyloid-β | ~4,479 | All 3 groups |
| Flortaucipir (AV1451/FTP) | FTP/ | Tau | ~447 | Subset (392 amyloidE + 55 LEARN) |

**Amyloid eligibility**: Composite SUVr ≥ 1.15 (cerebellar reference).

---

## 4. Visit/Session Structure

### A4 (amyloidE) — Key Sessions
| Session (NII) | Meaning | Imaging |
|---|---|---|
| 002 | SCV2: Amyloid PET screening | FBP |
| 004 | SCV4: Baseline MRI + full eval | T1, FLAIR, T2_SE, T2_star, fMRI, DWI, FTP |
| 009 | Wk 12 follow-up | MRI |
| 027 | Wk 52 (~12mo) | MRI |
| 048 | Wk 96 (~24mo) | MRI |
| 066 | Wk 240: primary endpoint | MRI |
| 072 | Wk 288: open-label extension | MRI |
| 084 | Wk 336 | MRI |
| 999 | Early termination | variable |

### LEARN (amyloidNE) — Key Sessions
| Session (NII) | Meaning | Imaging |
|---|---|---|
| 002 | SCV2: Amyloid PET screening (negative) | FBP |
| **006** | Baseline MRI (**differs from A4's 004!**) | T1, FLAIR, T2_SE, T2_star, fMRI, DWI, FTP |
| 024 | Wk 72 | MRI subset |
| 048 | Wk 96 | MRI subset |
| 066 | Wk 240 (end of observation) | MRI |
| 999 | Special/ET | variable |

**Note**: LEARN baseline MRI = session **006** (vs A4 = **004**). Extra 2 weeks for negative PET confirmation + LEARN enrollment.

---

## 5. Blood Biomarkers

### pTau217 (Primary)
- **Assay**: Lilly Clinical Diagnostics Lab, ECL immunoassay (MSD Sector S Imager 600 MM)
- **Unit**: U/mL
- **Results**: ORRESRAW (numeric, includes <LLOQ values) + ORRES (text, `<LLOQ` notation)
- **14.5% below LLOQ** → use ORRESRAW + LLOQ flag column
- **Visits**: A4: BL(VISCODE=6), Wk12(9), Wk240(66) | LEARN: SCR(1), Wk72(24), Wk240(66)
- **N**: 5,813 samples, 1,653 BIDs (A4: 1,165 + LEARN: 488)
- **Role**: PD biomarker (treatment effect) + amyloid positivity predictor
- **Location**: `DEMO/Clinical/External Data/biomarker_pTau217.csv`

### Additional Panels (Screening only)
| File | Content | N |
|------|---------|---|
| biomarker_Plasma_Roche_Results.csv | Roche ECLIA: GFAP, Aβ40/42, ApoE4, p-tau181, NF-L | 13,418 |
| biomarker_AB_Test.csv | Araclon ELISA Aβ plasma ratio | 4,588 |

**pTau217 VISCODE ≠ NII session code!** pTau217 uses integer VISCODE (6, 9, 24, 66), NII uses zero-padded 3-digit (002, 004, 006). Mapping required for joins.

---

## 6. Clinical Data Files

### Core Tables (metadata/)
| File | Content | Key Columns |
|------|---------|-------------|
| A4_PTDEMOG | Demographics (6,945 BIDs) | BID, PTGENDER(1=M/2=F), PTAGE, PTEDUCAT |
| A4_SUBJINFO | APOE genotype (78.9% complete) | BID, APOEGN(E3/E3 format), AGEYR, LRNFLGSNM(Y/N) |
| A4_REGISTRY | Visit schedule | BID, VISCODE(1-5), EXAMDAY(relative days) |
| A4_demography | Imaging inventory (14,333 rows) | Subject ID, Research Group, Visit, Study Date, Description |
| A4_MMSE | MMSE scores | BID, VISCODE, MMSCORE |
| A4_CDR | CDR scores | BID, VISCODE, CDGLOBAL, CDSOB |
| A4_SPPACC | PACC composite (34,009 rows) | BID, PACCQSNUM, PACCRN |
| dose.csv | Treatment arm (1,165 amyloidE) | BID, DOSELEVEL(1600mg/NA=placebo) |

### Imaging Tables (metadata/A4 Imaging data and docs/)
| File | Content | Key Columns |
|------|---------|-------------|
| A4_PETSUVR | Amyloid PET SUVR (35,935 rows) | BID, brain_region, suvr_cer, centiloid |
| A4_PETVADATA | Amyloid eligibility (4,491 BIDs) | BID, PMODSUVR, SCORE(positive/negative) |
| A4_VMRI | MRI volumes (1,271 BIDs, VISCODE=4) | BID, 53 FreeSurfer regions (NeuroQuant) |
| TAUSUVR | Tau PET SUVR (447 BIDs) | ID(=BID), 250+ FreeSurfer regions |

### NFS Documentation (PDFs)
Located in `metadata/A4 Imaging data and docs/`:
- Quick_Guide_to_A4_imaging_data_v1.1.pdf
- A4_Pre-rand_Data_Primer.pdf / FAQ.pdf
- A4_VMRI_Volumetric_MRI_Methods.pdf (NeuroQuant)
- A4_PET_Processing_Details_20221019.pdf (tau PET, Stanford pipeline)
- A4_PETSUVR / PETVADATA methods PDFs

---

## 7. Data Layout (NFS)

### NII Structure
```
/Volumes/nfs_storage/1_combined/A4/ORIG/NII/
    {BID}/                          # e.g., B10081264
        {session}/                  # e.g., 002, 004, 006
            {modality}/             # e.g., T1, FBP, FLAIR
                A4_{MR|PET}_{modality}_{BID}_{session}.nii.gz
                A4_{MR|PET}_{modality}_{BID}_{session}.json  # sidecar
```

### Clinical & Metadata
```
/Volumes/nfs_storage/1_combined/A4/ORIG/
    metadata/                       # Core clinical CSVs
        A4 Imaging data and docs/   # Imaging CSVs + PDFs
    DEMO/
        Clinical/
            External Data/          # Blood biomarker CSVs
        A4_matching_v1/             # Pipeline output
```

### Key Directories (Ignored)
- `NII_/`, `NII2/` — duplicates of NII/, scheduled for deletion. Use NII/ only.

---

## 8. ADNI vs A4 Key Differences

| | ADNI | A4 |
|---|---|---|
| Image format | DICOM (.dcm) | **NIfTI (.nii.gz + .json sidecar)** |
| Metadata source | pydicom extraction | **JSON parsing** |
| Folder structure | SOURCE/PTID/protocol/datetime/I{UID}/ | **BID/session/modality/A4_*.nii.gz** |
| Modality classification | regex pattern matching | **folder name = modality** |
| Matching strategy | EXAMDATE date matching (±180d) | **session_code direct join** |
| Dates | Absolute (YYYY-MM-DD) | **De-identified (year only, EXAMDAY=relative days)** |
| Subject ID | PTID (XXX_S_XXXX) | **BID (B[1-9][0-9]{8})** |
| Clinical data | ADNIMERGE.rda (unified) | **51+ individual CSVs** |
| Cohorts | ADNI 1/GO/2/3/4 | **A4(amyloidE) + LEARN(amyloidNE)** |

---

## 9. References

### NFS PDFs
- Quick_Guide_to_A4_imaging_data_v1.1.pdf
- A4_Pre-rand_Data_Primer.pdf
- A4_VMRI_Volumetric_MRI_Methods.pdf
- A4_PET_Processing_Details_20221019.pdf

### Key Publications
- Sperling RA, et al. *The A4 Study: Stopping AD Before Symptoms Begin?* Sci Transl Med. 2014.
- Sperling RA, et al. *Amyloid-related imaging abnormalities in amyloid-modifying therapeutic trials.* Ann Neurol. 2011.
- Donohue MC, et al. *The Preclinical Alzheimer Cognitive Composite.* JAMA Neurol. 2014.
