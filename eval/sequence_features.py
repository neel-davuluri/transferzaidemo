"""
Sequence position extraction for course text augmentation.

Extracts sequence position signals and appends natural-language tokens
to course text so BGE and CE can distinguish CHEM 3A vs 3B vs 3C etc.

Usage:
    from eval.sequence_features import seq_token_from_code, seq_token_from_text, augment_text

Unit tests at bottom (run directly: python eval/sequence_features.py).
"""

import re

# ── Code-based extraction (UCSC target side) ─────────────────

_CODE_SUFFIX = re.compile(r"([A-Z]+)\s+\d+([A-D])L?$")

_SUFFIX_LABELS = {
    "A": "first in sequence",
    "B": "second in sequence",
    "C": "third in sequence",
    "D": "fourth in sequence",
}

def seq_token_from_code(course_code):
    """
    Extract sequence token from a course code letter suffix.

    CHEM 3A → 'first in sequence'
    PHYS 6C → 'third in sequence'
    MATH 23B → 'second in sequence'
    CHEM 1N  → None  (N is not a sequence letter)
    STAT 5   → None  (no suffix)
    """
    m = _CODE_SUFFIX.match(str(course_code).strip())
    if not m:
        return None
    suffix = m.group(2)
    return _SUFFIX_LABELS.get(suffix)


# ── Description-based extraction (CCC query side) ────────────

_DESC_PATTERNS = [
    # Explicit ordinals
    (1, re.compile(r"\bfirst\s+(?:in\s+(?:a|the)\s+)?(?:course|quarter|semester|part|of the series|in the series)\b", re.I)),
    (2, re.compile(r"\bsecond\s+(?:in\s+(?:a|the)\s+)?(?:course|quarter|semester|part|of the series|in the series)\b", re.I)),
    (3, re.compile(r"\bthird\s+(?:in\s+(?:a|the)\s+)?(?:course|quarter|semester|part|of the series|in the series)\b", re.I)),
    (3, re.compile(r"\bfinal\s+(?:course|quarter|semester|part)\b", re.I)),
    # Roman numerals in description body
    (1, re.compile(r"\bpart\s+(?:1|i)\b", re.I)),
    (2, re.compile(r"\bpart\s+(?:2|ii)\b", re.I)),
    (3, re.compile(r"\bpart\s+(?:3|iii)\b", re.I)),
    # "continuation of" strongly implies ≥2
    (2, re.compile(r"\bcontinuation\s+of\b", re.I)),
]

_TITLE_PATTERNS = [
    # Roman numerals or digits in course TITLE (e.g. "Calculus II", "Chemistry 3")
    (1, re.compile(r"\b(?:i|1st|one)\s*$", re.I)),
    (2, re.compile(r"\b(?:ii|2nd|two)\s*$", re.I)),
    (3, re.compile(r"\b(?:iii|3rd|three)\s*$", re.I)),
    (4, re.compile(r"\b(?:iv|4th|four)\s*$", re.I)),
    (1, re.compile(r"\b[1I]\b\s*$")),
    (2, re.compile(r"\b(?:2|II)\b\s*$")),
    (3, re.compile(r"\b(?:3|III)\b\s*$")),
]

_LABEL_MAP = {
    1: "first in sequence",
    2: "second in sequence",
    3: "third in sequence",
    4: "fourth in sequence",
}

def seq_token_from_text(title, description=""):
    """
    Infer sequence position from course title and/or description text.

    Priority: title patterns > description patterns.
    Returns natural-language token or None.

    Examples:
        "Calculus II", ""             → 'second in sequence'
        "General Chemistry", "This is the third quarter..." → 'third in sequence'
        "Introduction to Physics", "" → None
    """
    # Title takes priority (more reliable signal)
    for pos, pat in _TITLE_PATTERNS:
        if pat.search(title.strip()):
            return _LABEL_MAP[pos]

    # Fall back to description
    for pos, pat in _DESC_PATTERNS:
        if pat.search(description):
            return _LABEL_MAP[pos]

    return None


# ── Text augmentation ─────────────────────────────────────────

def augment_text(base_text, token):
    """Append sequence token to text if not None."""
    if not token:
        return base_text
    return f"{base_text} {token}".strip()


# ══════════════════════════════════════════════════════════════
# UNIT TESTS
# ══════════════════════════════════════════════════════════════

def run_tests():
    tests = [
        # (description, expected, label)

        # Code-based
        ("seq_token_from_code('CHEM 3A')",
         seq_token_from_code("CHEM 3A"),
         "first in sequence"),

        ("seq_token_from_code('PHYS 6C')",
         seq_token_from_code("PHYS 6C"),
         "third in sequence"),

        ("seq_token_from_code('MATH 23B')",
         seq_token_from_code("MATH 23B"),
         "second in sequence"),

        ("seq_token_from_code('STAT 5') → None",
         seq_token_from_code("STAT 5"),
         None),

        ("seq_token_from_code('CHEM 4AL') → first (A before L)",
         seq_token_from_code("CHEM 4AL"),
         "first in sequence"),

        ("seq_token_from_code('CHEM 1N') → None (N not sequence)",
         seq_token_from_code("CHEM 1N"),
         None),

        # Text-based (title)
        ("seq_token_from_text('Calculus II')",
         seq_token_from_text("Calculus II"),
         "second in sequence"),

        ("seq_token_from_text('General Chemistry I')",
         seq_token_from_text("General Chemistry I"),
         "first in sequence"),

        ("seq_token_from_text('Organic Chemistry III')",
         seq_token_from_text("Organic Chemistry III"),
         "third in sequence"),

        # Text-based (description)
        ("seq_token_from_text('Physics B', 'This is the second course in the series')",
         seq_token_from_text("Physics B", "This is the second course in the series"),
         "second in sequence"),

        ("seq_token_from_text('Chemistry', 'This is the third and final quarter...')",
         seq_token_from_text("Chemistry", "This is the third and final quarter of the sequence"),
         "third in sequence"),

        ("seq_token_from_text('Chemistry', 'A continuation of the previous course')",
         seq_token_from_text("Chemistry", "A continuation of the previous course"),
         "second in sequence"),

        ("seq_token_from_text('Introduction to Acting', '') → None",
         seq_token_from_text("Introduction to Acting", "Introduces students to basic acting."),
         None),
    ]

    passed = failed = 0
    for label, got, expected in tests:
        ok = got == expected
        status = "PASS" if ok else "FAIL"
        if not ok:
            print(f"  {status}  {label}")
            print(f"         expected={expected!r}  got={got!r}")
            failed += 1
        else:
            print(f"  {status}  {label}")
            passed += 1

    print(f"\n{passed}/{passed+failed} tests passed")
    return failed == 0


# ── Lab course token ─────────────────────────────────────────

_LAB_TITLE = re.compile(r"\b(lab|laboratory)\b", re.I)
_LAB_CODE  = re.compile(r"[A-Z]+\s+\d+L$")

def lab_token_from_code_and_desc(course_code, title="", description=""):
    """
    Returns 'laboratory course' if the course is a genuine lab section.

    Requires BOTH:
      - course code ends in L  (e.g. ECE 101L, PHYS 6L)
      - title OR description contains 'lab' or 'laboratory'

    The double condition avoids false positives like MUSC 105L
    ('The Music and Life of Prince') which has an L suffix but is not a lab.
    """
    if not _LAB_CODE.match(str(course_code).strip()):
        return None
    if _LAB_TITLE.search(title) or _LAB_TITLE.search(description):
        return "laboratory course"
    return None


def lab_token_from_query(title, description=""):
    """
    Returns 'laboratory course' if the CCC query is a lab course.
    Matches titles like 'Circuit Analysis Laboratory' or 'Chemistry Lab'.
    """
    if _LAB_TITLE.search(title):
        return "laboratory course"
    return None


if __name__ == "__main__":
    print("Running sequence feature unit tests...\n")
    ok = run_tests()

    print("\nRunning lab token tests...\n")
    lab_tests = [
        ("ECE 101L, title has lab",
         lab_token_from_code_and_desc("ECE 101L", "Introduction to Electronic Circuits Laboratory", ""),
         "laboratory course"),
        ("PHYS 6L, desc has laboratory",
         lab_token_from_code_and_desc("PHYS 6L", "Introductory Physics Lab", "One lab session per week."),
         "laboratory course"),
        ("MUSC 105L, no lab in title/desc",
         lab_token_from_code_and_desc("MUSC 105L", "The Music and Life of Prince", "Explores Prince's music."),
         None),
        ("ECE 101, no L suffix",
         lab_token_from_code_and_desc("ECE 101", "Introduction to Electronic Circuits", ""),
         None),
        ("CCC query with 'Laboratory' in title",
         lab_token_from_query("Circuit Analysis Laboratory"),
         "laboratory course"),
        ("CCC query without lab",
         lab_token_from_query("General Chemistry"),
         None),
    ]
    for label, got, expected in lab_tests:
        status = "PASS" if got == expected else "FAIL"
        print(f"  {status}  {label}")
        if got != expected:
            print(f"         expected={expected!r}  got={got!r}")

    import sys; sys.exit(0 if ok else 1)
