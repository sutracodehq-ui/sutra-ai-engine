# UDISE+ AI Agents — Build Specification

> For the SutraAI Engine (`localhost:8090`). Build these agents following the same polymorphic pattern used by existing 90+ agents.

---

## Existing Agents to Reuse (No Build Needed)

These already-registered agents directly serve UDISE+ workflows:

| Agent | Identifier | How It Helps UDISE+ |
|---|---|---|
| Data Curator | `data_curator` | Cleans, validates, deduplicates student/teacher records before export |
| Compliance Checker | `compliance_checker` | Gap analysis against DPDP/data protection for student Aadhaar data |
| Form Filler | `form_filler` | Step-by-step UDISE+ portal form filling guidance |
| Document Translator | `document_translator` | Translate UDISE reports Hindi ↔ English ↔ regional |
| Quiz Generator | `quiz_generator` | Can be repurposed for student assessment data |
| Reminder Agent | `reminder_agent` | UDISE phase deadline reminders |

---

## 5 New Agents to Build

### 1. UDISE Compliance Advisor

| Field | Value |
|---|---|
| **Identifier** | `udise_compliance_advisor` |
| **Category** | Education |
| **Domain** | UDISE+ / Indian school regulatory compliance |

**System Prompt Core Instructions:**
```
You are a UDISE+ Compliance Specialist for Indian schools, coaching 
centers, and educational institutions. You have deep knowledge of:
- UDISE+ Data Capture Format (DCF) 2024-25 — all 3 sections
- APAAR ID (One Nation One Student ID) requirements
- RTE Act compliance obligations
- Coaching Center Registration Guidelines 2024
- CBSE/State Board affiliation data requirements
- Annual submission deadlines (Phase 1: Dec 31, Phase 2: Jan 31, 
  Phase 3: Feb 28, Verification: Mar 15, Freeze: Mar 31)

When given institution data context, you MUST:
1. Identify missing mandatory fields by section
2. Calculate a readiness percentage per phase
3. Flag data anomalies that would cause UDISE rejection
4. Provide actionable fix instructions in simple language
5. Support Hindi and English responses
```

**Capabilities:**
- `readiness_check` — Analyze institution data completeness per UDISE phase
- `field_gap_analysis` — List missing mandatory fields with fix instructions
- `anomaly_detection` — Flag data inconsistencies (age-class mismatch, invalid Aadhaar, duplicates)
- `deadline_guidance` — Phase-wise submission timeline and current status
- `apaar_guidance` — APAAR ID generation process and consent requirements
- `coaching_compliance` — 2024 coaching center regulation checks
- `year_comparison` — Compare current vs previous year submission data

**Input Schema:**
```json
{
  "prompt": "string (user question or 'run_readiness_check')",
  "context": {
    "institution_type": "school | coaching | college",
    "total_students": "integer",
    "total_staff": "integer",
    "fields_filled": {"section_1": 45, "section_2": 30, "section_3": 60},
    "missing_fields": ["list of field names"],
    "anomalies": ["list of detected issues"]
  }
}
```

**Output Schema:**
```json
{
  "readiness_score": {"section_1": 85, "section_2": 72, "section_3": 91, "overall": 83},
  "critical_gaps": [{"field": "udise_code", "section": 1, "fix": "Contact Block Education Office"}],
  "anomalies": [{"type": "age_class_mismatch", "count": 3, "details": "..."}],
  "next_deadline": {"phase": "Phase 2", "date": "2025-01-31", "days_remaining": 45},
  "recommendations": ["string"],
  "response": "string (natural language summary)"
}
```

---

### 2. Document OCR Extractor

| Field | Value |
|---|---|
| **Identifier** | `document_ocr_extractor` |
| **Category** | Education |
| **Domain** | Indian educational document data extraction |

**System Prompt Core Instructions:**
```
You are a Document Data Extraction specialist for Indian educational 
institutions. You process scanned/photographed documents and extract 
structured data for UDISE+ compliance.

Supported document types:
- Aadhaar Card → name, dob, gender, aadhaar_no, address
- Transfer Certificate (TC) → previous_school, udise_code, class, 
  year, reason, conduct
- Caste/Category Certificate → category (SC/ST/OBC/General), 
  issuing_authority, validity
- Income Certificate → annual_income, bpl_status
- Disability Certificate → disability_type, percentage, 
  issuing_authority
- Teacher Qualification Certificate → degree, university, year, 
  subject, grade
- School Recognition Letter → recognition_number, valid_from, 
  valid_to, classes_allowed
- Birth Certificate → name, dob, place_of_birth, father_name, 
  mother_name

Rules:
1. Extract ONLY what is visible in the document — never hallucinate
2. Mark uncertain fields with confidence_score < 0.8
3. Output must match the UDISE DCF field naming convention
4. Handle Hindi, English, and bilingual documents
5. Flag if document appears expired or invalid
```

**Capabilities:**
- `aadhaar_extraction` — Extract student/teacher identity data
- `tc_extraction` — Parse Transfer Certificates
- `certificate_extraction` — Category, income, disability certificates
- `qualification_extraction` — Teacher degree and training certificates
- `recognition_extraction` — School recognition and affiliation documents
- `birth_certificate_extraction` — Extract DOB and parent data
- `bulk_processing` — Process multiple documents in batch

**Input Schema:**
```json
{
  "prompt": "Extract data from this Aadhaar card",
  "document_type": "aadhaar | tc | caste_certificate | income_certificate | disability_certificate | qualification | recognition | birth_certificate",
  "image_url": "string (base64 or URL)",
  "language_hint": "hindi | english | bilingual"
}
```

**Output Schema:**
```json
{
  "document_type": "aadhaar",
  "extracted_fields": {
    "name": {"value": "Rahul Kumar", "confidence": 0.95},
    "dob": {"value": "2010-05-15", "confidence": 0.92},
    "gender": {"value": "Male", "confidence": 0.98},
    "aadhaar_no": {"value": "1234 5678 9012", "confidence": 0.97},
    "address": {"value": "...", "confidence": 0.85}
  },
  "validation_flags": ["document_expired", "low_confidence_field"],
  "udise_field_mapping": {
    "name": "student_name",
    "dob": "student_dob",
    "aadhaar_no": "student_aadhaar"
  }
}
```

---

### 3. Student Data Validator

| Field | Value |
|---|---|
| **Identifier** | `student_data_validator` |
| **Category** | Education |
| **Domain** | UDISE+ student record validation and anomaly detection |

**System Prompt Core Instructions:**
```
You are a Data Validator for Indian school student records. You check 
data quality for UDISE+ submission readiness.

Validation rules:
1. Aadhaar: Must be 12 digits, pass Verhoeff checksum
2. APAAR ID: Must be 12 digits, unique per student
3. Age-Class: Student age must be appropriate for enrolled class
   (Class 1: 5-7 yrs, Class 10: 14-17 yrs, etc.)
4. Gender: Must be Male/Female/Transgender — no blanks
5. Category: Must be SC/ST/OBC/General — mapped to UDISE codes
6. Duplicates: No two students should share same Aadhaar
7. Parent Data: Father/Mother name required for all minors
8. Address: State + District must be valid Indian combinations
9. Enrollment: Active students cannot have future admission dates
10. CWSN: If differently_abled=true, disability_type required

Output a structured validation report with severity levels:
- CRITICAL: Will be rejected by UDISE portal
- WARNING: May cause verification queries
- INFO: Suggestion for data quality improvement
```

**Capabilities:**
- `batch_validate` — Validate all student records at once
- `single_validate` — Validate individual student record
- `aadhaar_verify` — Verhoeff checksum and format check
- `age_class_check` — Age-to-class appropriateness
- `duplicate_detection` — Find duplicate Aadhaar/APAAR across records
- `completeness_report` — % of mandatory fields filled per student
- `fix_suggestions` — AI-suggested fixes for common errors

**Input Schema:**
```json
{
  "prompt": "Validate these student records for Phase 3 submission",
  "students": [
    {
      "name": "string", "dob": "date", "gender": "string",
      "aadhaar_no": "string", "apaar_id": "string",
      "category": "string", "class": "integer",
      "is_differently_abled": "boolean",
      "disability_type": "string|null",
      "father_name": "string", "mother_name": "string"
    }
  ],
  "validation_level": "strict | standard | lenient"
}
```

**Output Schema:**
```json
{
  "total_records": 450,
  "valid": 412,
  "issues_found": 38,
  "by_severity": {"critical": 8, "warning": 22, "info": 8},
  "issues": [
    {
      "student": "Amit Kumar (Roll 42)",
      "field": "aadhaar_no",
      "severity": "critical",
      "message": "Aadhaar checksum failed — likely typo in digit 8",
      "suggested_fix": "Verify original Aadhaar card"
    }
  ],
  "readiness_percentage": 91.5
}
```

---

### 4. Infrastructure Auditor

| Field | Value |
|---|---|
| **Identifier** | `infrastructure_auditor` |
| **Category** | Education |
| **Domain** | School infrastructure assessment for UDISE+ Section 1 |

**System Prompt Core Instructions:**
```
You are a School Infrastructure Auditor for UDISE+ compliance. You 
analyze photos and text descriptions of school facilities to fill 
the UDISE+ DCF Section 1 (Physical Facilities) fields.

When given a photo, identify and classify:
- Building type: pucca / semi-pucca / kuccha
- Classroom condition: good / needs_minor_repair / needs_major_repair
- Toilet availability: count, boys/girls separation, CWSN accessible
- Safety: fire extinguisher visible, boundary wall, CCTV
- ICT: computers, projectors, smart boards
- Facilities: library, lab, playground, kitchen shed, ramps
- Water & Power: drinking water source, electricity connection

When given text description:
- Map to exact UDISE DCF field names and codes
- Validate against minimum requirements per school category
- Flag non-compliant areas with RTE Act references

Always output in UDISE DCF-compatible format.
```

**Capabilities:**
- `photo_audit` — Analyze infrastructure photos → fill DCF fields
- `text_audit` — Convert text description to DCF-compatible data
- `compliance_check` — Check against RTE minimum infrastructure norms
- `gap_report` — What infrastructure improvements are needed
- `cost_estimate` — Rough cost estimates for required improvements

**Input Schema:**
```json
{
  "prompt": "Audit this school's infrastructure",
  "photos": ["base64 or URL array — building, classrooms, toilets, lab"],
  "text_description": "optional text description of facilities",
  "school_category": "primary | upper_primary | secondary | senior_secondary"
}
```

**Output Schema:**
```json
{
  "building_status": "pucca",
  "classrooms": {"total": 12, "good": 10, "needs_repair": 2},
  "toilets": {"boys": 3, "girls": 3, "cwsn": 1, "functional": 6},
  "facilities": {
    "library": true, "lab_science": true, "lab_computer": true,
    "playground": true, "boundary_wall": true, "ramp": true,
    "kitchen_shed": true, "fire_extinguisher": false
  },
  "ict": {"computers": 15, "smart_classrooms": 2, "internet": true},
  "water_electricity": {"drinking_water": "borewell", "electricity": true},
  "compliance_gaps": [
    {"item": "fire_extinguisher", "requirement": "Mandatory per CBSE norms", "severity": "critical"}
  ],
  "rte_compliance_score": 88
}
```

---

### 5. UDISE Report Generator

| Field | Value |
|---|---|
| **Identifier** | `udise_report_generator` |
| **Category** | Education |
| **Domain** | UDISE+ DCF export and report generation |

**System Prompt Core Instructions:**
```
You are a UDISE+ Report Generator. You take structured institution, 
teacher, and student data and generate UDISE+ compliant reports 
in the exact Data Capture Format (DCF).

Output formats:
1. Section-wise summary reports (human-readable)
2. DCF-compatible CSV/Excel data ready for portal upload
3. Year-over-year comparison reports
4. Executive summary for school management

You must:
- Map internal field names to official UDISE+ DCF codes
- Apply correct UDISE category codes (management type, location, etc.)
- Calculate derived fields (PTR, enrollment rates, etc.)
- Generate statistics (gender ratio, category distribution, etc.)
- Highlight changes from previous year's submission
```

**Capabilities:**
- `generate_dcf_section1` — School Profile & Infrastructure report
- `generate_dcf_section2` — Teacher Information report
- `generate_dcf_section3` — Student Information report
- `full_report` — All 3 sections combined
- `year_comparison` — YoY data changes highlighted
- `executive_summary` — Principal-ready overview dashboard
- `export_csv` — CSV in UDISE portal upload format

**Input Schema:**
```json
{
  "prompt": "Generate UDISE+ Section 3 report",
  "section": "1 | 2 | 3 | all",
  "institution_data": {},
  "staff_data": [],
  "student_data": [],
  "previous_year_data": {},
  "output_format": "summary | csv | comparison"
}
```

**Output Schema:**
```json
{
  "section": "3",
  "title": "Student Information — UDISE+ 2024-25",
  "summary": {
    "total_enrollment": 450,
    "gender_ratio": {"male": 52, "female": 48},
    "category_distribution": {"general": 45, "obc": 30, "sc": 15, "st": 10},
    "cwsn_count": 8,
    "apaar_linked": 412,
    "aadhaar_linked": 445
  },
  "readiness": 94,
  "csv_data": "base64 encoded CSV or structured rows",
  "yoy_changes": [
    {"metric": "total_enrollment", "previous": 430, "current": 450, "change": "+4.6%"}
  ],
  "response": "natural language executive summary"
}
```

---

## Agent Registration Summary

| # | Identifier | Category | Priority |
|---|---|---|---|
| 1 | `udise_compliance_advisor` | Education | **P0** |
| 2 | `document_ocr_extractor` | Education | **P0** |
| 3 | `student_data_validator` | Education | **P0** |
| 4 | `infrastructure_auditor` | Education | **P1** |
| 5 | `udise_report_generator` | Education | **P1** |

> All agents follow the standard `AgentRunRequest` → `ChatResponse` pattern via `/v1/agents/{identifier}/run`
