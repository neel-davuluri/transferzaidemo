import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from paths import VT_CATALOG

"""
Scrape the full Virginia Tech 2025-26 undergraduate course catalog.

Output: vt_courses_2025.csv
Schema: course_code, course_title, course_description
        (identical to wm_courses_2025.csv)

Source: https://catalog.vt.edu/undergraduate/course-descriptions/{dept}/
Dept list: extracted from catalog.vt.edu/sitemap.xml
"""

import csv
import re
import time
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}
DELAY = 0.4  # seconds between requests

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
            time.sleep(1.5 * (attempt + 1))
    return None


# ══════════════════════════════════════════════════════════════
# STEP 1 — Get all department slugs from sitemap
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Discovering departments from sitemap")
print("=" * 60)

# Warm up session with a known-good page first (sitemap needs a live session)
warmup = get("https://catalog.vt.edu/undergraduate/course-descriptions/math/")
if warmup is None:
    raise RuntimeError("Cannot reach catalog.vt.edu — check network.")

time.sleep(0.5)

sitemap = get("https://catalog.vt.edu/sitemap.xml")
if sitemap is None or len(sitemap.content) == 0:
    raise RuntimeError("Sitemap empty — try running again (session warm-up sometimes needed).")

text = sitemap.content.decode("utf-8", errors="replace")
depts = sorted(set(re.findall(r"/undergraduate/course-descriptions/([a-z]+)/", text)))

print(f"Departments found in sitemap: {len(depts)}")
print(depts)


# ══════════════════════════════════════════════════════════════
# STEP 2 — Scrape each department page
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Scraping course descriptions")
print("=" * 60)

BASE_URL = "https://catalog.vt.edu/undergraduate/course-descriptions/{dept}/"

courses = []       # list of (course_code, course_title, course_description)
dept_stats = {}    # dept -> count

for i, dept in enumerate(depts):
    url = BASE_URL.format(dept=dept)
    r = get(url)
    if r is None:
        print(f"  [{i+1:>3}/{len(depts)}] {dept.upper():<8}  FAILED — skipping")
        dept_stats[dept] = 0
        time.sleep(DELAY)
        continue

    soup = BeautifulSoup(r.text, "html.parser")
    blocks = soup.find_all("div", class_="courseblock")

    dept_count = 0
    for block in blocks:
        code_el  = block.find("span", class_="detail-code")
        title_el = block.find("span", class_="detail-title")
        desc_el  = block.find("div", class_="courseblockextra")

        if not code_el:
            continue

        code  = code_el.get_text(strip=True)
        title = title_el.get_text(strip=True) if title_el else ""
        desc  = desc_el.get_text(separator=" ", strip=True) if desc_el else ""

        # Normalize: title comes as "- Course Name" — strip leading dash
        title = re.sub(r"^\s*-\s*", "", title).strip()

        # Skip grad-only courses (5000+) — the undergrad page rarely has them,
        # but filter defensively
        m = re.search(r"\d+", code)
        if m and int(m.group()) >= 5000:
            continue

        courses.append({
            "course_code":        code,
            "course_title":       title,
            "course_description": desc if desc else "Description not found",
        })
        dept_count += 1

    dept_stats[dept] = dept_count
    print(f"  [{i+1:>3}/{len(depts)}] {dept.upper():<8}  {dept_count:>4} courses")
    time.sleep(DELAY)

print(f"\nTotal courses scraped: {len(courses)}")


# ══════════════════════════════════════════════════════════════
# STEP 3 — Validate
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Validation")
print("=" * 60)

total = len(courses)
missing_desc = sum(1 for c in courses if c["course_description"] == "Description not found")
missing_title = sum(1 for c in courses if not c["course_title"])
dupes = total - len({c["course_code"] for c in courses})

print(f"Total courses:        {total}")
print(f"Missing description:  {missing_desc} ({missing_desc/max(total,1)*100:.1f}%)")
print(f"Missing title:        {missing_title}")
print(f"Duplicate codes:      {dupes}")

# Top 5 largest departments
top_depts = sorted(dept_stats.items(), key=lambda x: -x[1])[:5]
print(f"\nTop 5 departments by course count:")
for dept, cnt in top_depts:
    print(f"  {dept.upper():<8} {cnt}")

# Zero-course departments (likely 404 or no undergrad offerings)
empty_depts = [d for d, c in dept_stats.items() if c == 0]
if empty_depts:
    print(f"\nEmpty departments ({len(empty_depts)}): {empty_depts}")

# Sample
import random
random.seed(42)
sample = random.sample(courses, min(5, total))
print(f"\n5 random courses:")
for c in sample:
    print(f"  {c['course_code']:<14} {c['course_title'][:50]:<50}  "
          f"desc={len(c['course_description'])} chars")


# ══════════════════════════════════════════════════════════════
# STEP 4 — Save
# ══════════════════════════════════════════════════════════════
out_path = VT_CATALOG
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["course_code", "course_title", "course_description"])
    writer.writeheader()
    writer.writerows(courses)

print(f"\nSaved {total} courses to {out_path}")
print("Done.")
