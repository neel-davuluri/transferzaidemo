import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from paths import UCSC_CATALOG

"""
Scrape the full UC Santa Cruz 2025-26 undergraduate course catalog.

Output: ucsc_courses_2025.csv
Schema: course_code, course_title, course_description
        (identical to wm_courses_2025.csv and vt_courses_2025.csv)

Source: https://catalog.ucsc.edu/en/current/general-catalog/courses/{dept-slug}/
Dept list: scraped from courses index page
"""

import csv
import re
import time
import requests
from bs4 import BeautifulSoup

BASE = "https://catalog.ucsc.edu"
COURSES_INDEX = f"{BASE}/en/current/general-catalog/courses/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
DELAY = 0.4

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
# STEP 1 — Discover department pages
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Discovering departments from courses index")
print("=" * 60)

r = get(COURSES_INDEX)
if r is None:
    raise RuntimeError("Cannot reach catalog.ucsc.edu")

soup = BeautifulSoup(r.text, "html.parser")
dept_paths = sorted(set(
    l["href"] for l in soup.find_all("a", href=True)
    if "/general-catalog/courses/" in l["href"] and l["href"].count("/") == 5
))

# Skip the catch-all graduate section — grad courses are 200+ and also filtered below
dept_paths = [p for p in dept_paths if not p.endswith("/grad-graduate")]

print(f"Department pages found: {len(dept_paths)}")


# ══════════════════════════════════════════════════════════════
# STEP 2 — Scrape each department page
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Scraping course descriptions")
print("=" * 60)

courses = []
dept_stats = {}

for i, path in enumerate(dept_paths):
    dept_slug = path.split("/")[-1]
    url = f"{BASE}{path}/"
    r = get(url)

    if r is None:
        print(f"  [{i+1:>3}/{len(dept_paths)}] {dept_slug:<45}  FAILED")
        dept_stats[dept_slug] = 0
        time.sleep(DELAY)
        continue

    soup = BeautifulSoup(r.text, "html.parser")
    courselist = soup.find("div", class_="courselist")

    if not courselist:
        print(f"  [{i+1:>3}/{len(dept_paths)}] {dept_slug:<45}  0 courses (no courselist)")
        dept_stats[dept_slug] = 0
        time.sleep(DELAY)
        continue

    # Each course starts with an h2.course-name; description is the first div.desc after it
    dept_count = 0
    headings = courselist.find_all("h2", class_="course-name")

    for h2 in headings:
        a = h2.find("a")
        if not a:
            continue

        # Code is in <span>, title is the remaining text of the <a>
        span = a.find("span")
        code = span.get_text(strip=True) if span else ""
        if span:
            span.extract()
        title = a.get_text(strip=True).lstrip("- ").strip()

        if not code:
            continue

        # Filter out graduate courses (numbered 200+)
        m = re.search(r"\d+", code)
        if m and int(m.group()) >= 200:
            continue

        # Description: first non-empty div.desc immediately after the h2
        desc = ""
        for sibling in h2.next_siblings:
            if sibling.name == "h2":
                break
            if sibling.name == "div" and "desc" in (sibling.get("class") or []):
                text = sibling.get_text(separator=" ", strip=True)
                if text:
                    desc = text
                    break

        courses.append({
            "course_code":        code,
            "course_title":       title,
            "course_description": desc if desc else "Description not found",
        })
        dept_count += 1

    dept_stats[dept_slug] = dept_count
    print(f"  [{i+1:>3}/{len(dept_paths)}] {dept_slug:<45}  {dept_count:>4} courses")
    time.sleep(DELAY)

print(f"\nTotal courses scraped: {len(courses)}")


# ══════════════════════════════════════════════════════════════
# STEP 3 — Validate
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Validation")
print("=" * 60)

# Deduplicate on course_code (some courses appear under multiple division headings)
seen = set()
deduped = []
for c in courses:
    if c["course_code"] not in seen:
        seen.add(c["course_code"])
        deduped.append(c)
courses = deduped

total = len(courses)
missing_desc = sum(1 for c in courses if c["course_description"] == "Description not found")
missing_title = sum(1 for c in courses if not c["course_title"])
dupes = 0  # already resolved

print(f"Total courses:        {total}")
print(f"Missing description:  {missing_desc} ({missing_desc/max(total,1)*100:.1f}%)")
print(f"Missing title:        {missing_title}")
print(f"Duplicate codes:      {dupes}")

top_depts = sorted(dept_stats.items(), key=lambda x: -x[1])[:5]
print(f"\nTop 5 departments by course count:")
for dept, cnt in top_depts:
    print(f"  {dept:<45} {cnt}")

empty = [d for d, c in dept_stats.items() if c == 0]
if empty:
    print(f"\nEmpty departments ({len(empty)}): {[d.split('-')[0].upper() for d in empty]}")

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
out_path = UCSC_CATALOG
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["course_code", "course_title", "course_description"])
    writer.writeheader()
    writer.writerows(courses)

print(f"\nSaved {total} courses to {out_path}")
print("Done.")
