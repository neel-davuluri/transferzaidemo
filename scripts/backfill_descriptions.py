import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from paths import CCC_UCSC_MERGED, CCC_UCSC_CLEAN

"""
Backfill missing CCC descriptions in ccc_ucsc_merged.csv.

Strategy:
1. For each missing row, try all 5 schools using various prefix/slug mappings.
2. De Anza: build reverse lookup (normalize D-prefix codes → short form).
3. Write filtered CSV with only rows that have BOTH descriptions.
"""

import csv
import re
import time
import requests
from bs4 import BeautifulSoup
from collections import defaultdict

import build_ccc_ucsc_dataset as b

HEADERS = b.HEADERS
session = b.session

# Foothill: space→underscore slug map for multi-word prefixes
FOOTHILL_PREFIX_TO_SLUG = {
    "C S":    "c_s",
    "L A":    "l_a",
    "D A":    "d_a",
    "D H":    "d_h",
    "R T":    "r_t",
    "V T":    "v_t",
    # Prefix → different slug
    "ACCT":   "actg",
    "ACCTG":  "actg",
    "ACTG":   "actg",
    "BUSAC":  "actg",
    "ARTS":   "art",
    "ART":    "art",
    "BUSI":   "busi",
    "BUS":    "busi",
    "THTR":   "thtr",
    "THEA":   "thtr",
    "TH ART": "thtr",
    "DRAMA":  "thtr",
    "TA":     "thtr",
    "MUS":    "mus",
    "MUSI":   "mus",
    "MDIA":   "mdia",
    "GID":    "gid",
    "ANTHR":  "anth",
    "ANTHRO": "anth",
    "FRNCH":  None,   # Foothill doesn't have French
    "FREN":   None,
    "GERM":   None,
    "GRMN":   None,
    "GRMAN":  None,
    "GERMAN": None,
    "ITAL":   None,
    "POLI":   "poli",
    "POLSC":  "poli",
    "POLS":   "pols",
    "PS":     "poli",
    "POL":    "poli",
    "SOC":    "soc",
    "SOCIO":  "soc",
    "SOCIOL": "soc",
    "PSYC":   "psyc",
    "PSYCH":  "psyc",
    "ENGL":   "engl",
    "EWRT":   "engl",
    "ENGR":   "engr",
    "ENGIN":  "engr",
    "JAPN":   "japn",
    "JAPAN":  "japn",
    "ASTR":   "astr",
    "ASTRO":  "astr",
    "GEOG":   "geog",
    "BIOL":   "biol",
    "BIO":    "biol",
    "HLTH":   "hlth",
    "HS":     "hlth",
    "EDUC":   "educ",
    "EDAC":   "edac",
    "WMN":    "wmn",
    "WS":     "wmn",
    "WGS":    "wmn",
    "WMST":   "wmn",
    "ETHN":   "ethn",
    "ES":     None,   # Foothill uses ETHN not ES
}


def _load_foothill_prefix_ex(prefix):
    """Load Foothill courses for a prefix, handling slug overrides and underscores."""
    # Check override map (None means Foothill doesn't have this dept)
    if prefix.upper() in FOOTHILL_PREFIX_TO_SLUG:
        slug = FOOTHILL_PREFIX_TO_SLUG[prefix.upper()]
        if slug is None:
            return {}
        url = f"https://catalog.foothill.edu/courses-az/{slug}/"
    else:
        # Default: lowercase, spaces→underscore
        slug = prefix.lower().replace(" ", "_")
        url = f"https://catalog.foothill.edu/courses-az/{slug}/"

    if url in _foothill_ex_cache:
        return _foothill_ex_cache[url]

    r = session.get(url, headers=b._FOOTHILL_HEADERS, timeout=20)
    courses = {}
    if r and r.status_code == 200:
        soup = BeautifulSoup(r.text, "html.parser")
        for block in soup.find_all("div", class_="courseblock"):
            code_el  = block.find(class_="detail-code")
            title_el = block.find(class_="detail-title")
            extra    = block.find(class_="courseblockextra")
            if code_el and extra:
                code = re.sub(r"\s*[•·]\s*$", "", code_el.get_text(strip=True)).strip()
                desc = extra.get_text(separator=" ", strip=True)
                desc = re.sub(r"\s*(Prerequisite|Advisory|Formerly|Transfer Credit|UC|CSU).*", "", desc, flags=re.S).strip()
                title = title_el.get_text(strip=True) if title_el else ""
                if code:
                    courses[code] = {"title": title, "desc": desc if len(desc) > 20 else "Description not found"}
    _foothill_ex_cache[url] = courses
    time.sleep(b.CATALOG_DELAY)
    return courses


_foothill_ex_cache = {}

# ── De Anza reverse lookup ────────────────────────────────────

def normalize_deanza_code(catalog_code):
    """'CHEM D001A' -> 'CHEM 1A',  'CHEM D01AH' -> 'CHEM 1AH'"""
    c = catalog_code.rstrip(".")
    m = re.match(r"^([A-Z][A-Z0-9 ]*?)\s+D(\w+)$", c.strip())
    if not m:
        return None
    prefix, num = m.group(1).strip(), m.group(2)
    short = re.sub(r"^0+(\d)", r"\1", num)
    return f"{prefix} {short}"


# Maps: normalized_short_code -> (dept_slug, catalog_slug)
_deanza_reverse = {}   # e.g. "CHEM 1A" -> "chemd001a"
_deanza_loaded_depts = set()


def _ensure_deanza_dept_loaded(dept_slug):
    if dept_slug in _deanza_loaded_depts:
        return
    dept_map = b._load_deanza_dept(dept_slug)   # {data_code: slug}
    for catalog_code, course_slug in dept_map.items():
        short = normalize_deanza_code(catalog_code)
        if short:
            _deanza_reverse[short.upper()] = course_slug
    _deanza_loaded_depts.add(dept_slug)


# ASSIST prefix → De Anza catalog dept slug (expanded from auto-map + manual)
ASSIST_TO_DEANZA_DEPT = {
    "MATH":    "math-mathematics-courses",
    "STAT":    "stat-statistics-courses",
    "CHEM":    "chem-chemistry-courses",
    "PHYS":    "phys-physics-courses",
    "PHYSCS":  "phys-physics-courses",   # De Anza uses PHYSCS in ASSIST
    "BIOL":    "biol-biology-courses",
    "BIO":     "biol-biology-courses",
    "C S":     "cis-computer-sceince-and-information-systems-courses",
    "CS":      "cis-computer-sceince-and-information-systems-courses",
    "CIS":     "cis-computer-sceince-and-information-systems-courses",
    "ICS":     "ics-intercultural-studies-courses",
    "ECON":    "econ-economics-courses",
    "HIST":    "hist-history-courses",
    "POLI":    "poli-political-science-courses",
    "POL":     "poli-political-science-courses",
    "PSYC":    "psyc-psychology-courses",
    "PSYCH":   "psyc-psychology-courses",
    "SOC":     "soc-sociology-courses",
    "SOCIO":   "soc-sociology-courses",
    "ANTH":    "anth-anthropology-courses",
    "ANTHR":   "anth-anthropology-courses",
    "PHIL":    "phil-philosophy-courses",
    "PHILO":   "phil-philosophy-courses",
    "ENGL":    "engl-english-courses",
    "EWRT":    "ewrt-english-writing-courses",
    "SPAN":    "span-spanish-courses",
    "FREN":    "fren-french-courses",
    "GERM":    "germ-german-courses",
    "ITAL":    "ital-italian-courses",
    "JAPN":    "japn-japanese-courses",
    "JAPAN":   "japn-japanese-courses",
    "MAND":    "mand-mandarin-courses",
    "CHNESE":  "mand-mandarin-courses",
    "GEOL":    "geol-geology-courses",
    "DANC":    "danc-dance-courses",
    "DANCE":   "danc-dance-courses",
    "THEA":    "thea-theatre-arts-courses",
    "TH ART":  "thea-theatre-arts-courses",
    "MUS":     "musi-music-courses",
    "MUSI":    "musi-music-courses",
    "ART":     "arts-art-courses",
    "ARTS":    "arts-art-courses",
    "ARTDM":   "artdm-art-digital-media-courses",
    "PHTG":    "phtg-photography-courses",
    "PHOTO":   "phtg-photography-courses",
    "ACCT":    "accounting-courses",
    "ACCTG":   "accounting-courses",
    "ACTG":    "accounting-courses",
    "BUS":     "bus-business-courses",
    "ECE":     "cd-child-development-courses",
    "C D":     "cd-child-development-courses",
    "ETHN":    "es-environmental-studies-courses",  # might be wrong
    "ES":      "es-environmental-studies-courses",
    "CETH":    "ceth-comparative-ethnic-studies-courses",
    "CHLX":    "chlx-chicanx-latinx-studies-courses",
    "ENGR":    "engr-engineering-courses",
    "ENGIN":   "engr-engineering-courses",
    "GID":     "artdm-art-digital-media-courses",
    "LING":    "ling-linguistics-courses",
    "WGS":     "wmst-womens-studies-courses",
    "WMST":    "wmst-womens-studies-courses",
    "WS":      "wmst-womens-studies-courses",
    "WMN":     "wmst-womens-studies-courses",
    "EDUC":    "educ-education-courses",
    "EDAC":    "edac-educational-access-courses",
    "INTL":    "intl-international-studies-courses",
    "HS":      "hlth-health-courses",
    "ASTRO":   "astr-astronomy-courses",
    "ASTR":    "astr-astronomy-courses",
    "FTVE":    "ftv-film-and-television-production-courses",
    "FILM":    "ftv-film-and-television-production-courses",
    "ETECH":   "dmt-design-and-manufacturing-technologies-courses",
    "AP":      "admj-administration-of-justice-courses",
    "ADMJ":    "admj-administration-of-justice-courses",
    "PS":      "poli-political-science-courses",
    "GEOG":    "geo-geography-courses",
}

# ASSIST uses different prefix for De Anza courses in some cases
# e.g. ASSIST shows "PHYSCS 21" but De Anza catalog has "PHYS D051A" (different number too!)
# Build a title-based fallback for these edge cases
ASSIST_CODE_OVERRIDES = {
    # ASSIST code (upper) -> De Anza catalog slug
    # These are cases where number mapping fails
}


def fetch_deanza_by_assist_code(assist_code):
    """Try to get De Anza description using ASSIST code like 'CHEM 1A', 'PHYSCS 21'."""
    code_upper = assist_code.upper().strip()
    prefix = extract_prefix(code_upper)
    if not prefix:
        return None
    num_part = code_upper[len(prefix):].strip().split()[0] if code_upper[len(prefix):].strip() else ""
    num = num_part

    dept_slug = ASSIST_TO_DEANZA_DEPT.get(prefix)
    if not dept_slug:
        return None

    _ensure_deanza_dept_loaded(dept_slug)

    # Try exact match against reverse lookup
    desc = _try_deanza_slug_from_reverse(code_upper, prefix, num)
    if desc:
        return desc

    # Also try alternate prefix mapping (e.g. PHYSCS -> PHYS)
    alt_prefixes = {
        "PHYSCS": "PHYS", "C S": "CIS", "CS": "CIS",
        "POL": "POLI", "TH ART": "THEA",
    }
    if prefix in alt_prefixes:
        alt = alt_prefixes[prefix]
        alt_code = f"{alt} {num}"
        desc = _try_deanza_slug_from_reverse(alt_code.upper(), alt, num)
        if desc:
            return desc

    return None


def _try_deanza_slug_from_reverse(code_upper, prefix, num):
    course_slug = _deanza_reverse.get(code_upper)
    if not course_slug:
        # Try fuzzy: prefix + num without spaces
        slug_guess = b._elumen_slug(f"{prefix} {num}")
        # Check if any reverse entry ends with this slug suffix
        for k, v in _deanza_reverse.items():
            if k.startswith(prefix) and v == slug_guess:
                course_slug = slug_guess
                break
        if not course_slug:
            # Final guess: direct slug
            course_slug = slug_guess

    cfg = b.ELUMEN_SCHOOLS[113]
    url = f"https://api-prod.elumenapp.com/catalog/sites/publish/content/{cfg['catalog']},course,{course_slug}?tenant={cfg['tenant']}"
    r = b.requests_get(url)
    if not r:
        return None
    desc = b._parse_elumen_desc(r.text, "deanza")
    time.sleep(b.CATALOG_DELAY)
    return desc if desc and len(desc) > 20 else None


# ── Multi-school fallback ─────────────────────────────────────
# Try each school in order until we get a description

def extract_prefix(code):
    """Extract course prefix, handling multi-word prefixes like 'C S' and 'TH ART'."""
    upper = code.upper().strip()
    # Known multi-word prefixes
    for pref in ["TH ART", "POL SC", "C S", "C D", "L A", "D A", "D H", "R T", "V T"]:
        if upper.startswith(pref + " "):
            return pref
    m = re.match(r"^([A-Z][A-Z0-9]*)\s", upper)
    return m.group(1) if m else ""


def try_all_schools(ccc_code):
    """Try DVC, De Anza, Cabrillo, Foothill, SMC in order."""
    prefix = extract_prefix(ccc_code)

    # 1. De Anza (most likely for the problem prefixes)
    desc = fetch_deanza_by_assist_code(ccc_code)
    if desc:
        return desc

    # 2. DVC
    cfg = b.ELUMEN_SCHOOLS[114]
    desc = b._fetch_elumen_course(cfg, ccc_code)
    if desc:
        time.sleep(b.CATALOG_DELAY)
        return desc

    # 3. Cabrillo
    desc = b._fetch_cabrillo_course(ccc_code)
    if desc:
        return desc

    # 4. Foothill (with extended slug handling)
    if prefix:
        dept_courses = _load_foothill_prefix_ex(prefix)
        info = dept_courses.get(ccc_code.upper()) or dept_courses.get(ccc_code)
        if info and info["desc"] != "Description not found":
            return info["desc"]

    # 5. SMC (with extended prefix map)
    if prefix:
        smc_map = dict(b._SMC_PREFIX_MAP)
        smc_map.update({
            "PHYSCS": "Physics",
            "C S":    "Computer Science",
            "ICS":    "Computer Information Systems",
            "ARTS":   "Art",
            "CHNESE": "Chinese",
            "MAND":   "Chinese",
            "TH ART": "Theatre Arts",
            "THEA":   "Theatre Arts",
            "PHTG":   "Photography",
            "PHOTO":  "Photography",
            "ACTG":   "Accounting",
            "BUSAC":  "Accounting",
            "POL":    "Political Science",
            "POL SC": "Political Science",
            "PS":     "Political Science",
            "SOCIO":  "Sociology",
            "SOCIOL": "Sociology",
            "ANTHR":  "Anthropology",
            "ANTHRO": "Anthropology",
            "PHILO":  "Philosophy",
            "PHILOS": "Philosophy",
            "FRNCH":  "French",
            "FRENCH": "French",
            "GRMN":   "German",
            "GRMAN":  "German",
            "GERMAN": "German",
            "JAPAN":  "Japanese",
            "ASTRO":  "Astronomy",
            "GID":    "Graphic Design",
            "ARTG":   "Graphic Design",
            "EWRT":   "English – Composition",
            "ENGL":   "English – Composition",
            "ENGR":   "Engineering",
            "ENGIN":  "Engineering",
            "BIO":    "Biological Sciences",
            "BIOSC":  "Biological Sciences",
            "BIOL":   "Biological Sciences",
            "ETECH":  "Engineering",
            "HS":     "Health Education",
            "HLTH":   "Health Education",
            "EDUC":   "Education",
            "EDAC":   "Education",
            "INTL":   "Global Studies",
            "CETH":   "Ethnic Studies",
            "CHLX":   "Ethnic Studies",
            "MSIA":   "Media Studies",
            "MDIA":   "Media Studies",
            "MEDIA":  "Media Studies",
            "FTVE":   "Film Studies",
            "AP":     "Administration of Justice",
            "ADMJ":   "Administration of Justice",
        })
        subject = smc_map.get(prefix.upper())
        if subject:
            subj_courses = b._load_smc_subject(subject)
            info = subj_courses.get(ccc_code.upper()) or subj_courses.get(ccc_code)
            if info and info["desc"] != "Description not found":
                return info["desc"]

    return None


# ── Main ──────────────────────────────────────────────────────

def main():
    rows = []
    with open(str(CCC_UCSC_MERGED)) as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    missing_before = sum(1 for r in rows if r["CCC Description"] == "Description not found")
    print(f"Loaded {total} rows, {missing_before} missing CCC descriptions")

    # Collect unique missing codes
    missing_codes = {}
    for r in rows:
        if r["CCC Description"] == "Description not found":
            prefix = extract_prefix(r["CCC Course"])
            rest = r["CCC Course"][len(prefix):].strip()
            num = rest.split()[0] if rest else ""
            if prefix and num:
                code = f"{prefix} {num}"
                if code not in missing_codes:
                    missing_codes[code] = None

    print(f"Unique missing codes to look up: {len(missing_codes)}")

    # Fetch descriptions for each unique code
    n_found = 0
    for i, code in enumerate(sorted(missing_codes.keys())):
        desc = try_all_schools(code)
        missing_codes[code] = desc
        status = "OK" if desc else "MISS"
        if (i + 1) % 20 == 0 or (i + 1) == len(missing_codes):
            print(f"  [{i+1}/{len(missing_codes)}] [{status}] {code}")
        elif desc:
            print(f"  [{status}] {code}")

    n_found = sum(1 for v in missing_codes.values() if v)
    print(f"\nBackfill: found {n_found}/{len(missing_codes)} missing descriptions")

    # Apply backfill to rows
    for r in rows:
        if r["CCC Description"] == "Description not found":
            prefix = extract_prefix(r["CCC Course"])
            rest = r["CCC Course"][len(prefix):].strip()
            num = rest.split()[0] if rest else ""
            if prefix and num:
                code = f"{prefix} {num}"
                if missing_codes.get(code):
                    r["CCC Description"] = missing_codes[code]

    # Write full updated CSV
    with open(str(CCC_UCSC_MERGED), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", "CCC Course", "CCC Description",
                         "UCSC Course Code", "UCSC Course Title", "UCSC Description"])
        for i, row in enumerate(rows):
            writer.writerow([i, row["CCC Course"], row["CCC Description"],
                             row["UCSC Course Code"], row["UCSC Course Title"],
                             row["UCSC Description"]])

    # Write clean filtered CSV (both descriptions present)
    clean_rows = [r for r in rows
                  if r["CCC Description"] != "Description not found"
                  and r["UCSC Description"] != "Description not found"]

    with open(str(CCC_UCSC_CLEAN), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", "CCC Course", "CCC Description",
                         "UCSC Course Code", "UCSC Course Title", "UCSC Description"])
        for i, row in enumerate(clean_rows):
            writer.writerow([i, row["CCC Course"], row["CCC Description"],
                             row["UCSC Course Code"], row["UCSC Course Title"],
                             row["UCSC Description"]])

    missing_after = sum(1 for r in rows if r["CCC Description"] == "Description not found")
    print(f"\nFinal stats:")
    print(f"  Total rows:           {total}")
    print(f"  CCC desc missing:     {missing_after} ({missing_after/total*100:.1f}%)")
    ucsc_miss = sum(1 for r in rows if r["UCSC Description"] == "Description not found")
    print(f"  UCSC desc missing:    {ucsc_miss} ({ucsc_miss/total*100:.1f}%)")
    print(f"  Both present (clean): {len(clean_rows)}")
    print(f"\nSaved: ccc_ucsc_merged.csv ({total} rows) + ccc_ucsc_clean.csv ({len(clean_rows)} rows)")


if __name__ == "__main__":
    main()
