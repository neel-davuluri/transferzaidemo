import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from paths import VT_MERGED

"""
Build vccs_vt_merged.csv — VCCS→VT transfer equivalency dataset.

Steps:
  1. Parse VT equivalency table (filter X-pattern courses)
  2. Scrape VCCS course descriptions from courses.vccs.edu
  3. Scrape VT course descriptions from catalog.vt.edu
  4. Assemble CSV matching vccs_wm_merged.csv schema
  5. Validate and print diagnostics
"""

import re
import time
import random
import csv
import requests
from bs4 import BeautifulSoup
from collections import defaultdict

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
DELAY = 0.5  # seconds between requests

session = requests.Session()
session.headers.update(HEADERS)


def get(url, retries=3):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == retries - 1:
                print(f"  FAILED {url}: {e}")
                return None
            time.sleep(1 + attempt)
    return None


# ══════════════════════════════════════════════════════════════
# STEP 1 — Parse VT equivalency table
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Parsing VT equivalency table")
print("=" * 60)

VT_TABLE_URL = "https://transferguide.registrar.vt.edu/VCCS-Equivalencies/VCCS-Equivalencies-2025.html"
r = get(VT_TABLE_URL)
soup = BeautifulSoup(r.text, "html.parser")
rows = soup.find_all("table")[0].find_all("tr")[1:]  # skip header row

# X-pattern: any code containing an X character (e.g. BUS 1XXX, MATH 1XXX, VT XXXX)
HAS_X = re.compile(r'X', re.IGNORECASE)

# Parse a course code string that may have " + " combos
# e.g. "BIOL 1105 + 1115"  or  "BIO 101 + BIO 102"
# Returns list of "DEPT NNNN" strings
def split_codes(raw, default_dept=None):
    """Split a possibly-combined course code string into individual codes."""
    parts = [p.strip() for p in raw.split("+")]
    codes = []
    last_dept = default_dept
    for part in parts:
        part = part.strip()
        # Could be "DEPT NNNN" or just "NNNN" (dept carried from previous)
        m = re.match(r"([A-Z]{2,6})\s+(\d{3,4})", part)
        if m:
            last_dept = m.group(1)
            codes.append(f"{m.group(1)} {m.group(2)}")
        else:
            # Try just a number, carry dept
            m2 = re.match(r"(\d{3,4})", part)
            if m2 and last_dept:
                codes.append(f"{last_dept} {m2.group(1)}")
    return codes


equivalencies = []  # list of dicts

for row in rows:
    cells = [td.get_text(strip=True) for td in row.find_all("td")]
    if len(cells) < 5:
        continue

    vccs_code_raw  = cells[0].strip()
    vccs_title_raw = cells[1].strip()
    vt_code_raw    = cells[3].strip()
    vt_title_raw   = cells[4].strip()

    # Skip rows where VT code contains X (electives, no-credit)
    if HAS_X.search(vt_code_raw):
        continue
    if not vt_code_raw or vt_code_raw.upper() in ("NO VIRGINIA TECH CREDIT", ""):
        continue

    # Parse multi-course combos
    vccs_codes = split_codes(vccs_code_raw)
    vt_codes   = split_codes(vt_code_raw)

    if not vccs_codes or not vt_codes:
        continue

    # Validate: no X in any individual code after splitting
    if any(HAS_X.search(c) for c in vccs_codes + vt_codes):
        continue

    equivalencies.append({
        "vccs_codes":   vccs_codes,
        "vccs_title":   vccs_title_raw,
        "vt_codes":     vt_codes,
        "vt_title":     vt_title_raw,
        "vccs_raw":     vccs_code_raw,
        "vt_raw":       vt_code_raw,
    })

print(f"Valid equivalency pairs (no-X): {len(equivalencies)}")
print(f"Multi-VCCS-course rows: {sum(1 for e in equivalencies if len(e['vccs_codes']) > 1)}")
print(f"Multi-VT-course rows:   {sum(1 for e in equivalencies if len(e['vt_codes']) > 1)}")

# Unique VCCS and VT codes
all_vccs_codes = set()
all_vt_codes   = set()
for e in equivalencies:
    for c in e["vccs_codes"]: all_vccs_codes.add(c)
    for c in e["vt_codes"]:   all_vt_codes.add(c)

print(f"Unique VCCS codes to scrape: {len(all_vccs_codes)}")
print(f"Unique VT codes to scrape:   {len(all_vt_codes)}")


# ══════════════════════════════════════════════════════════════
# STEP 2 — Scrape VCCS course descriptions
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Scraping VCCS course descriptions")
print("=" * 60)

vccs_desc_cache = {}

def scrape_vccs_desc(code):
    """Fetch description for a VCCS course code like 'MTH 263'."""
    # URL format: /courses/MTH263 (no space)
    url_code = code.replace(" ", "")
    url = f"https://courses.vccs.edu/courses/{url_code}"
    r = get(url)
    if r is None:
        return "Description not found"
    soup = BeautifulSoup(r.text, "html.parser")
    el = soup.find("div", class_="coursedesc")
    if el:
        text = el.get_text(separator=" ", strip=True)
        if text:
            return text
    return "Description not found"


sorted_vccs = sorted(all_vccs_codes)
for i, code in enumerate(sorted_vccs):
    desc = scrape_vccs_desc(code)
    vccs_desc_cache[code] = desc
    if (i + 1) % 20 == 0 or (i + 1) == len(sorted_vccs):
        found = sum(1 for v in vccs_desc_cache.values() if v != "Description not found")
        print(f"  {i+1}/{len(sorted_vccs)} scraped  ({found} found)")
    time.sleep(DELAY)

found = sum(1 for v in vccs_desc_cache.values() if v != "Description not found")
missing = len(vccs_desc_cache) - found
print(f"VCCS descriptions: {found} found, {missing} missing ({missing/len(vccs_desc_cache)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════
# STEP 3 — Scrape VT course descriptions
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Scraping VT course descriptions")
print("=" * 60)

# Group VT codes by department prefix to batch-fetch by dept page
vt_by_dept = defaultdict(list)
for code in all_vt_codes:
    m = re.match(r"([A-Z]+)\s+(\d+)", code)
    if m:
        vt_by_dept[m.group(1)].append(code)

print(f"VT departments to fetch: {sorted(vt_by_dept.keys())}")

vt_desc_cache  = {}
vt_title_cache = {}

def scrape_vt_dept(dept):
    """Fetch all courses for a VT department from the catalog."""
    url = f"https://catalog.vt.edu/undergraduate/course-descriptions/{dept.lower()}/"
    r = get(url)
    if r is None:
        return {}
    soup = BeautifulSoup(r.text, "html.parser")
    courses = {}
    for block in soup.find_all("div", class_="courseblock"):
        code_tag  = block.find("span", class_="detail-code")
        title_tag = block.find("span", class_="detail-title")
        desc_tag  = block.find("div", class_="courseblockextra")

        if not code_tag:
            continue
        code  = code_tag.get_text(strip=True)
        title = title_tag.get_text(strip=True).lstrip("- ").strip() if title_tag else ""
        desc  = desc_tag.get_text(separator=" ", strip=True) if desc_tag else ""

        if code:
            courses[code] = {"title": title, "desc": desc or "Description not found"}
    return courses


fetched_depts = set()
for i, (dept, codes) in enumerate(sorted(vt_by_dept.items())):
    if dept in fetched_depts:
        continue
    print(f"  [{i+1}/{len(vt_by_dept)}] Fetching VT dept: {dept} ({len(codes)} codes needed)")
    dept_courses = scrape_vt_dept(dept)

    if not dept_courses:
        # Dept page not found — mark all as not found
        for code in codes:
            vt_desc_cache[code]  = "Description not found"
            vt_title_cache[code] = ""
        print(f"    No courses found for {dept}")
    else:
        found_in_dept = 0
        for code in codes:
            if code in dept_courses:
                vt_desc_cache[code]  = dept_courses[code]["desc"]
                vt_title_cache[code] = dept_courses[code]["title"]
                found_in_dept += 1
            else:
                vt_desc_cache[code]  = "Description not found"
                vt_title_cache[code] = ""
        print(f"    {found_in_dept}/{len(codes)} codes found on dept page ({len(dept_courses)} total courses on page)")

    fetched_depts.add(dept)
    time.sleep(DELAY)

found_vt  = sum(1 for v in vt_desc_cache.values() if v != "Description not found")
missing_vt = len(vt_desc_cache) - found_vt
print(f"\nVT descriptions: {found_vt} found, {missing_vt} missing ({missing_vt/max(len(vt_desc_cache),1)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════
# STEP 4 — Assemble CSV
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Assembling CSV")
print("=" * 60)

# Build course string like "MTH 263 CALCULUS I"
def make_vccs_course_string(codes, title):
    """Build VCCS Course column value. For combos, join with ' TAKEN WITH '."""
    # For a simple single code: "MTH 263 CALCULUS I"
    # For multi: "BIO 101 GENERAL BIOLOGY I TAKEN WITH BIO 102 GENERAL BIOLOGY II"
    # But we only have one title for the whole row from the table.
    # For multi-VCCS, embed the codes with the shared title.
    if len(codes) == 1:
        return f"{codes[0]} {title}"
    else:
        return " TAKEN WITH ".join(f"{c} {title}" for c in codes)


output_rows = []
for eq in equivalencies:
    vccs_codes = eq["vccs_codes"]
    vt_codes   = eq["vt_codes"]

    # VCCS Course column
    vccs_course_str = make_vccs_course_string(vccs_codes, eq["vccs_title"])

    # VCCS Description — join multi-course descriptions with " | "
    vccs_descs = [vccs_desc_cache.get(c, "Description not found") for c in vccs_codes]
    vccs_desc_str = " | ".join(vccs_descs) if len(vccs_descs) > 1 else vccs_descs[0]

    # VT Course Code — join multi-course codes with " + "
    vt_code_str = " + ".join(vt_codes)

    # VT Course Title — from equivalency table (already combined in original)
    vt_title_str = eq["vt_title"]

    # VT Description — join multi-course descriptions with " | "
    vt_descs = [vt_desc_cache.get(c, "Description not found") for c in vt_codes]
    vt_desc_str = " | ".join(vt_descs) if len(vt_descs) > 1 else vt_descs[0]

    output_rows.append({
        "VCCS Course":    vccs_course_str,
        "VCCS Description": vccs_desc_str,
        "VT Course Code": vt_code_str,
        "VT Course Title": vt_title_str,
        "VT Description": vt_desc_str,
    })

print(f"Total output rows: {len(output_rows)}")


# ══════════════════════════════════════════════════════════════
# STEP 5 — Validate
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Validation")
print("=" * 60)

total = len(output_rows)
vccs_missing = sum(1 for r in output_rows if "Description not found" in r["VCCS Description"])
vt_missing   = sum(1 for r in output_rows if "Description not found" in r["VT Description"])
x_in_vccs    = sum(1 for r in output_rows if HAS_X.search(r["VCCS Course"].split()[0] + " " + r["VCCS Course"].split()[1]))
x_in_vt      = sum(1 for r in output_rows if HAS_X.search(r["VT Course Code"]))

print(f"Total rows:                  {total}")
print(f"VCCS desc missing:           {vccs_missing} ({vccs_missing/total*100:.1f}%) {'⚠ FLAG' if vccs_missing/total > 0.20 else '✓'}")
print(f"VT desc missing:             {vt_missing} ({vt_missing/total*100:.1f}%) {'⚠ FLAG' if vt_missing/total > 0.20 else '✓'}")
print(f"X in VCCS Course Code:       {x_in_vccs} {'⚠ FLAG' if x_in_vccs > 0 else '✓'}")
print(f"X in VT Course Code:         {x_in_vt} {'⚠ FLAG' if x_in_vt > 0 else '✓'}")

print(f"\n5 random rows for spot-check:")
import random as rnd
rnd.seed(42)
sample = rnd.sample(output_rows, min(5, total))
for i, r in enumerate(sample):
    print(f"\n  [{i+1}]")
    print(f"    VCCS Course:  {r['VCCS Course']}")
    print(f"    VCCS Desc:    {r['VCCS Description'][:120]}...")
    print(f"    VT Code:      {r['VT Course Code']}")
    print(f"    VT Title:     {r['VT Course Title']}")
    print(f"    VT Desc:      {r['VT Description'][:120]}...")


# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════
out_path = VT_MERGED
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["", "VCCS Course", "VCCS Description", "VT Course Code", "VT Course Title", "VT Description"])
    for i, row in enumerate(output_rows):
        writer.writerow([
            i,
            row["VCCS Course"],
            row["VCCS Description"],
            row["VT Course Code"],
            row["VT Course Title"],
            row["VT Description"],
        ])

print(f"\nSaved {total} rows to {out_path}")
