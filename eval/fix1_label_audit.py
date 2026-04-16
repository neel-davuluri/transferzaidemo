"""
Fix 1 — Label error audit for sequential UCSC course pairs.

Flags eval pairs where the CCC description's sequence position
likely disagrees with the labeled UCSC target's sequence position.

Output: logs/ucsc_label_audit.csv
        (ccc_course, ccc_description, current_label, current_label_title,
         suggested_correction, confidence, reason)

Does NOT auto-correct. Human review required.
"""

import sys, re
sys.path.insert(0, ".")
from pathlib import Path
from paths import CCC_UCSC_CLEAN, UCSC_CATALOG
from sklearn.model_selection import train_test_split
import pandas as pd

# ── Reproduce eval split ─────────────────────────────────────
df = pd.read_csv(CCC_UCSC_CLEAN)
df.columns = df.columns.str.strip()

def ucsc_dept(c):
    m = re.match(r"([A-Z]+)\s+", str(c).strip())
    return m.group(1) if m else "UNK"

df["dept"] = df["UCSC Course Code"].apply(ucsc_dept)
dc = df["dept"].value_counts()
df["strat"] = df["dept"].apply(lambda d: "RARE" if dc[d] < 2 else d)
_, eval_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["strat"])
print(f"Eval set: {len(eval_df)} rows")

# ── Load UCSC catalog for titles ─────────────────────────────
ucsc_cat = pd.read_csv(UCSC_CATALOG).dropna(subset=["course_code"])
ucsc_titles = {str(r["course_code"]).strip(): str(r.get("course_title", ""))
               for _, r in ucsc_cat.iterrows()}
ucsc_descs  = {str(r["course_code"]).strip(): str(r.get("course_description", ""))
               for _, r in ucsc_cat.iterrows()}


# ── Sequence position extractors ─────────────────────────────

def extract_desc_position(text):
    """
    Extract sequence position from CCC course description.
    Returns (position: int 1-5, method: str, match: str) or None.
    """
    text = str(text)
    tl   = text.lower()

    patterns = [
        # Ordinal words at start or as qualifier
        (1, r"\b(?:first|introductory|intro|beginning|elementary|1st)\b(?:\s+(?:course|quarter|semester|part|in the series|of the series))?"),
        (2, r"\b(?:second|intermediate|2nd)\b(?:\s+(?:course|quarter|semester|part|in the series|of the series))?"),
        (3, r"\b(?:third|3rd|advanced|final)\b(?:\s+(?:course|quarter|semester|part|in the series|of the series))?"),
        (4, r"\b(?:fourth|4th)\b(?:\s+(?:course|quarter|semester|part))"),
        # Roman numerals as part of sequence
        (1, r"\bpart\s+(?:1|i)\b"),
        (2, r"\bpart\s+(?:2|ii)\b"),
        (3, r"\bpart\s+(?:3|iii)\b"),
        # "continuation of" → implies ≥2
        (2, r"\bcontinuation\s+of\b"),
        # Explicit quarter/semester number
        (1, r"\bfirst\s+(?:quarter|semester|course)\b"),
        (2, r"\bsecond\s+(?:quarter|semester|course)\b"),
        (3, r"\bthird\s+(?:quarter|semester|course)\b"),
        (3, r"\bfinal\s+(?:quarter|semester|course)\b"),
        # "year-long series" qualifiers
        (1, r"\b(?:first|1st)\s+(?:of|in)\s+(?:a|the)\s+(?:two|three|four)"),
        (2, r"\b(?:second|2nd)\s+(?:of|in)\s+(?:a|the)\s+(?:two|three|four)"),
        (3, r"\b(?:third|3rd|last|final)\s+(?:of|in)\s+(?:a|the)\s+(?:two|three|four)"),
    ]

    for pos, pattern in patterns:
        m = re.search(pattern, tl)
        if m:
            return (pos, pattern, m.group(0))
    return None


def extract_code_position(code):
    """
    Extract sequence position from a UCSC course code suffix.
    CHEM 3A→1, 3B→2, 3C→3 / PHYS 6A→1, 6B→2, 6C→3 / MATH 19A→1, 19B→2
    Returns int 1-5 or None.
    """
    code = str(code).strip()
    # Letter suffix after number: 3A, 3B, 3C, 19A, 19B, 5A, 5C etc.
    m = re.search(r"\d+([A-D])(?:L)?$", code)
    if m:
        return {"A": 1, "B": 2, "C": 3, "D": 4}.get(m.group(1))
    # Roman numerals appended: unlikely in UCSC but handle
    m = re.search(r"\s+(I{1,3}|IV|V)$", code)
    if m:
        return {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}.get(m.group(1))
    return None


def same_sequence_family(code1, code2):
    """True if both codes share the same dept + base number (e.g. CHEM 3A/3B/3C)."""
    def base(c):
        c = str(c).strip()
        m = re.match(r"([A-Z]+)\s+(\d+)[A-DL]?$", c)
        return (m.group(1), m.group(2)) if m else None
    b1, b2 = base(code1), base(code2)
    return b1 is not None and b1 == b2


def find_sequence_sibling(current_code, target_pos, ucsc_titles):
    """
    Find the UCSC course with same dept+number base but at target_pos.
    Returns code or None.
    """
    dept_base = re.match(r"([A-Z]+)\s+(\d+)", str(current_code).strip())
    if not dept_base:
        return None
    dept, num = dept_base.group(1), dept_base.group(2)
    suffix = {1: "A", 2: "B", 3: "C", 4: "D"}.get(target_pos)
    if not suffix:
        return None
    candidate = f"{dept} {num}{suffix}"
    return candidate if candidate in ucsc_titles else None


# ── Audit ─────────────────────────────────────────────────────
SEQUENCE_DEPTS = {"CHEM", "PHYS", "MATH", "AM", "BIOL", "BIOE", "BIOC", "CSE", "ECE", "CMPM"}

flags = []
seq_indicators = [
    r"\bsecond\b", r"\bthird\b", r"\bfinal quarter\b", r"\bfinal course\b",
    r"\bcontinuation\b", r"\bpart\s+[23iiiii]\b", r"\b[23]rd\b",
    r"\bfirst\b", r"\bintroductory\b", r"\bpart 1\b", r"\bpart i\b",
]
SEQ_PATTERN = re.compile("|".join(seq_indicators), re.IGNORECASE)

for _, row in eval_df.iterrows():
    ccc_desc  = str(row.get("CCC Description", ""))
    ccc_course = str(row.get("CCC Course", ""))
    label     = str(row.get("UCSC Course Code", "")).strip()
    label_title = ucsc_titles.get(label, "")

    # Only audit sequence-sensitive departments
    label_dept = ucsc_dept(label)
    if label_dept not in SEQUENCE_DEPTS:
        continue

    # Only if description contains sequence indicator
    if not SEQ_PATTERN.search(ccc_desc):
        continue

    desc_pos = extract_desc_position(ccc_desc)
    code_pos = extract_code_position(label)

    if desc_pos is None or code_pos is None:
        continue

    desc_position, method, match_text = desc_pos

    if desc_position == code_pos:
        continue  # alignment looks correct

    # Mismatch — flag it
    suggested = find_sequence_sibling(label, desc_position, ucsc_titles)
    suggested_title = ucsc_titles.get(suggested, "") if suggested else ""

    # Confidence
    if abs(desc_position - code_pos) >= 2:
        conf = "high"
    elif "continuation" in method or "final" in match_text or "third" in match_text:
        conf = "high"
    else:
        conf = "medium"

    flags.append({
        "ccc_course":          ccc_course,
        "ccc_description":     ccc_desc[:400],
        "current_label":       label,
        "current_label_title": label_title,
        "current_label_pos":   code_pos,
        "desc_inferred_pos":   desc_position,
        "desc_match_text":     match_text,
        "suggested_correction":suggested or "—",
        "suggested_title":     suggested_title,
        "confidence":          conf,
        "reason":              f"Desc says pos={desc_position} ('{match_text}'), label is pos={code_pos} ({label})",
    })

# ── Output ────────────────────────────────────────────────────
flags_df = pd.DataFrame(flags)
out_path = "logs/ucsc_label_audit.csv"
flags_df.to_csv(out_path, index=False)

print(f"\nFlagged {len(flags_df)} potential label errors")
print(f"Saved to {out_path}")
print()

if len(flags_df) > 0:
    print(f"{'Conf':<8} {'Current':<12} {'Suggested':<12} {'Desc match':}")
    print("-" * 70)
    for _, r in flags_df.sort_values("confidence", ascending=False).iterrows():
        print(f"  {r['confidence']:<6} {r['current_label']:<12} {r['suggested_correction']:<12} '{r['desc_match_text']}'")
        print(f"         CCC: {r['ccc_course']}")
        print(f"         {r['reason']}")
        print()
