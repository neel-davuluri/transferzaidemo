import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from paths import CCC_UCSC_MERGED

"""
Build ccc_ucsc_merged.csv — CCC→UCSC transfer equivalency dataset.

Steps:
  1. Load ASSIST.org via Playwright to get session + agreements list
  2. For each agreement, NAVIGATE to the results page and intercept the
     articulation API response (avoids direct API calls → no rate limiting)
  3. Scrape UCSC course descriptions from catalog.ucsc.edu
  4. Scrape CCC course descriptions from accessible school catalogs
  5. Assemble CSV matching vccs_wm_merged.csv schema
  6. Validate and print diagnostics

Rate-limit strategy: navigate to each agreement page and let Angular make the
API call naturally via page.on('response', ...) interception. No fetch() calls
into the articulation endpoint — those trigger 429 after ~50 calls.
"""

import asyncio
import re
import time
import csv
import json
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
from playwright.async_api import async_playwright

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
CATALOG_DELAY = 0.35  # seconds between catalog scrape calls

# ASSIST.org IDs
UCSC_ID = 132
YEAR_ID = 74  # 2023-2024

# Key Bay Area / Central Coast CCCs
TARGET_CCC_IDS = {
    41:   "Cabrillo College",
    113:  "De Anza College",
    114:  "Diablo Valley College",
    51:   "Foothill College",
    137:  "Santa Monica College",
}

session = requests.Session()
session.headers.update(HEADERS)


def requests_get(url, retries=3):
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
# STEP 1-2  — ASSIST.org scraping via page navigation
# ══════════════════════════════════════════════════════════════

async def scrape_assist():
    print("=" * 60)
    print("STEP 1-2: Scraping ASSIST.org (page-navigation approach)")
    print("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Capture dict for institutions response only
        captured = {}

        async def on_institutions(response):
            if "assist.org/api/institutions" in response.url and response.status == 200:
                try:
                    captured["institutions"] = await response.json()
                except Exception:
                    pass

        page.on("response", on_institutions)

        # ── Prime the session by loading the ASSIST home page ──
        print("Loading ASSIST.org to establish session...")
        await page.goto("https://assist.org", wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(1000)

        # ── Get institutions via direct API (only ~1 call, always works) ──
        async def assist_get_small(endpoint):
            """Light API call — only used for institutions + agreements lists."""
            result = await page.evaluate(f"""
                async () => {{
                    const xToken = document.cookie
                        .split('; ')
                        .find(c => c.startsWith('X-XSRF-TOKEN='))
                        ?.split('=')[1] || '';
                    const r = await fetch('https://assist.org{endpoint}', {{
                        headers: {{
                            'Accept': 'application/json, text/plain, */*',
                            'Content-Type': 'application/json',
                            'X-XSRF-TOKEN': decodeURIComponent(xToken)
                        }}
                    }});
                    if (!r.ok) return null;
                    return await r.json();
                }}
            """)
            return result

        institutions = captured.get("institutions")
        if not institutions:
            print("Fetching institutions list...")
            institutions = await assist_get_small("/api/institutions")
        if not institutions:
            print("ERROR: Could not load institutions")
            await browser.close()
            return []

        cccs = [i for i in institutions if i.get("isCommunityCollege")]
        target_cccs = [c for c in cccs if c["id"] in TARGET_CCC_IDS]
        found_ids = {c["id"] for c in target_cccs}
        missing = set(TARGET_CCC_IDS.keys()) - found_ids
        if missing:
            print(f"Warning: CCC IDs not found in institutions: {missing}")
        print(f"Using {len(target_cccs)} target CCCs: {[c['names'][0]['name'] for c in target_cccs]}")

        # ── Fetch agreements list for each CCC (small number of calls) ──
        all_agreements = []
        print(f"\nFetching agreement lists...")
        for i, ccc in enumerate(target_cccs):
            ccc_id = ccc["id"]
            ccc_name = ccc["names"][0]["name"]
            endpoint = (
                f"/api/agreements?receivingInstitutionId={UCSC_ID}"
                f"&sendingInstitutionId={ccc_id}"
                f"&academicYearId={YEAR_ID}&categoryCode=major"
            )
            data = await assist_get_small(endpoint)
            count = 0
            if data and "reports" in data:
                for report in data["reports"]:
                    all_agreements.append({
                        "ccc_id": ccc_id,
                        "ccc_name": ccc_name,
                        "label": report["label"],
                        "key": report["key"],
                    })
                    count += 1
            print(f"  [{i+1}/{len(target_cccs)}] {ccc_name}: {count} agreements")
            await asyncio.sleep(0.3)

        print(f"\nTotal agreements to navigate: {len(all_agreements)}")

        # ── Articulation data via page navigation (rate-limit proof) ──
        def parse_articulations(data, agr):
            pairs = []
            if not data or not isinstance(data, dict) or not data.get("result"):
                return pairs
            try:
                articulations = json.loads(data["result"].get("articulations", "[]"))
            except Exception:
                return pairs
            for art in articulations:
                articulation = art.get("articulation", {})
                ucsc_course = articulation.get("course", {})
                if not ucsc_course:
                    continue
                ucsc_code = f"{ucsc_course.get('prefix','')} {ucsc_course.get('courseNumber','')}".strip()
                ucsc_title = ucsc_course.get("courseTitle", "")
                sending = articulation.get("sendingArticulation", {})
                if sending.get("noArticulationReason"):
                    continue
                for group in sending.get("items", []):
                    for ccc_course in group.get("items", []):
                        ccc_code = f"{ccc_course.get('prefix','')} {ccc_course.get('courseNumber','')}".strip()
                        ccc_title = ccc_course.get("courseTitle", "")
                        if ccc_code and ucsc_code:
                            pairs.append({
                                "ccc_code": ccc_code,
                                "ccc_title": ccc_title,
                                "ccc_name": agr["ccc_name"],
                                "ccc_id": agr["ccc_id"],
                                "ucsc_code": ucsc_code,
                                "ucsc_title": ucsc_title,
                            })
            return pairs

        course_pairs = []
        print(f"\nNavigating to each agreement page (intercepting API responses)...")

        for i, agr in enumerate(all_agreements):
            key = agr["key"]
            ccc_id = agr["ccc_id"]

            nav_url = (
                f"https://assist.org/transfer/results?year={YEAR_ID}"
                f"&institution={ccc_id}&agreement={UCSC_ID}"
                f"&agreementType=to&view=agreement&viewBy=major"
                f"&viewSendingAgreements=false&viewByKey={key}&includeCourseFamily=true"
            )

            # Use response interception via page.on('response') + short fixed wait
            # to avoid hanging for the full timeout on NO DATA pages
            captured_response = {}

            def make_handler(k, store):
                def handler(response):
                    if f"/api/articulation/Agreements?Key={k}" in response.url and response.status == 200:
                        store["response"] = response
                return handler

            handler = make_handler(key, captured_response)
            page.on("response", handler)

            data = None
            try:
                await page.goto(nav_url, wait_until="domcontentloaded", timeout=30000)
                # Wait up to 6 seconds for the articulation API response
                for _ in range(12):   # 12 × 0.5s = 6s max
                    if "response" in captured_response:
                        break
                    await asyncio.sleep(0.5)
                if "response" in captured_response:
                    data = await captured_response["response"].json()
            except Exception as e:
                pass
            finally:
                page.remove_listener("response", handler)

            new_pairs = parse_articulations(data, agr)
            course_pairs.extend(new_pairs)

            status = f"{len(new_pairs)} pairs" if data else "NO DATA"
            if (i + 1) % 25 == 0 or (i + 1) == len(all_agreements) or not data:
                print(f"  [{i+1}/{len(all_agreements)}] {agr['ccc_name']} / {agr['label'][:40]}: {status}  (total: {len(course_pairs)})")

        await browser.close()

    print(f"\nTotal raw course pairs: {len(course_pairs)}")
    return course_pairs


# ══════════════════════════════════════════════════════════════
# STEP 3 — UCSC course descriptions
# ══════════════════════════════════════════════════════════════

def build_ucsc_slug_map():
    """Map UCSC course prefix to catalog dept slug."""
    r = requests_get("https://catalog.ucsc.edu/en/current/general-catalog/courses/")
    if not r:
        return {}
    soup = BeautifulSoup(r.text, "html.parser")
    slug_map = {}
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        m = re.match(r".*/courses/([^/]+)/?$", href)
        if m:
            slug = m.group(1)
            text = a.get_text(strip=True)
            prefix = re.match(r"^([A-Z0-9]+)", text)
            if prefix:
                slug_map[prefix.group(1)] = slug
    return slug_map


def scrape_ucsc_dept(slug):
    """Return {code: {title, desc}} for a UCSC dept slug."""
    url = f"https://catalog.ucsc.edu/en/current/general-catalog/courses/{slug}/"
    r = requests_get(url)
    if not r:
        return {}
    soup = BeautifulSoup(r.text, "html.parser")
    courselist = soup.find("div", class_="courselist")
    if not courselist:
        return {}
    courses = {}
    for h2 in courselist.find_all("h2", class_="course-name"):
        a = h2.find("a")
        if not a:
            continue
        span = a.find("span")
        code = span.get_text(strip=True) if span else ""
        title = a.get_text(strip=True).replace(code, "").strip(" –-")
        desc_div = h2.find_next_sibling("div", class_="desc")
        desc = desc_div.get_text(separator=" ", strip=True) if desc_div else ""
        # Remove trailing metadata (credits, prerequisites, etc.)
        desc = re.sub(r"\s*\d+(\.\d+)?\s+Credits?.*", "", desc, flags=re.IGNORECASE | re.S).strip()
        desc = re.sub(r"\s*(Prerequisite|Requirement|Enrollment Restriction)[^.]*\.?.*", "", desc, flags=re.IGNORECASE | re.S).strip()
        if code:
            courses[code] = {
                "title": title,
                "desc": desc if len(desc) > 20 else "Description not found",
            }
    return courses


def scrape_ucsc_descriptions(all_ucsc_codes):
    print("\n" + "=" * 60)
    print("STEP 3: Scraping UCSC course descriptions")
    print("=" * 60)

    slug_map = build_ucsc_slug_map()
    print(f"UCSC dept slugs found: {len(slug_map)}")

    by_prefix = defaultdict(set)
    for code in all_ucsc_codes:
        m = re.match(r"([A-Z][A-Z0-9]*)\s+", code)
        if m:
            by_prefix[m.group(1)].add(code)

    ucsc_cache = {}
    fetched_slugs = set()

    for i, (prefix, codes) in enumerate(sorted(by_prefix.items())):
        slug = slug_map.get(prefix)
        if not slug:
            for c in codes:
                ucsc_cache[c] = {"title": "", "desc": "Description not found"}
            continue
        if slug in fetched_slugs:
            continue

        print(f"  [{i+1}/{len(by_prefix)}] {prefix} -> {slug} ({len(codes)} needed)")
        dept_courses = scrape_ucsc_dept(slug)
        found = 0
        for code in codes:
            if code in dept_courses:
                ucsc_cache[code] = dept_courses[code]
                found += 1
            else:
                ucsc_cache[code] = {"title": "", "desc": "Description not found"}
        print(f"    {found}/{len(codes)} found ({len(dept_courses)} on page)")
        fetched_slugs.add(slug)
        time.sleep(CATALOG_DELAY)

    n_found = sum(1 for v in ucsc_cache.values() if v["desc"] != "Description not found")
    print(f"\nUCSC: {n_found}/{len(ucsc_cache)} descriptions found")
    return ucsc_cache


# ══════════════════════════════════════════════════════════════
# STEP 4 — CCC course descriptions (Elumen + Foothill + SMC)
# ══════════════════════════════════════════════════════════════

# ── Elumen schools (DVC, De Anza, Cabrillo) ──────────────────

ELUMEN_SCHOOLS = {
    113: {"tenant": "deanza.elumenapp.com",   "catalog": "2025-2026",              "style": "deanza"},
    114: {"tenant": "dvc.elumenapp.com",       "catalog": "DVC2023-2024catalog",    "style": "dvc"},
    41:  {"tenant": "cabrillo.elumenapp.com",  "catalog": "2025-2026",              "style": "cabrillo"},
}


def _elumen_slug(code):
    """ACCT D001A -> acctd001a, MATH 121 -> math121, C S 2B -> cs2b"""
    return re.sub(r"\s+", "", code).lower()


def _parse_elumen_desc(html_text, style):
    """Extract course description from Elumen /content/ HTML response."""
    soup = BeautifulSoup(html_text, "html.parser")
    if style == "dvc":
        for div in soup.find_all("div", class_="l-body"):
            text = div.get_text(separator=" ", strip=True)
            if "Description:" in text:
                idx = text.find("Description:")
                desc = text[idx + len("Description:"):].strip()
                desc = re.sub(r"\s*(Student Learning Outcomes?|Advisory|Prerequisite|Course Note|Repeatability|Schedule Type).*", "", desc, flags=re.S).strip()
                return desc
    elif style == "deanza":
        for div in soup.find_all("div", class_="d-none"):
            text = div.get_text(separator=" ", strip=True)
            if "Course Description" in text:
                idx = text.find("Course Description")
                desc = text[idx:].split(":", 1)[-1].strip() if ":" in text[idx:idx+30] else text[idx + len("Course Description"):].strip()
                desc = re.sub(r"\s*(Student Learning Outcomes?|Advisory|Prerequisite|Transfer Credit|C-ID:|Repeatability).*", "", desc, flags=re.S).strip()
                return desc
    elif style == "cabrillo":
        for div in soup.find_all("div", class_="l-body"):
            text = div.get_text(separator=" ", strip=True)
            if len(text) > 50:
                desc = re.sub(r"\s*(Student Learning Outcomes?|Advisory|Prerequisite|Transfer Credit|Course Note|C-ID).*", "", text, flags=re.S).strip()
                return desc
    return None


def _fetch_elumen_course(school_cfg, code):
    """Return description string or None for an Elumen course."""
    catalog = school_cfg["catalog"]
    tenant  = school_cfg["tenant"]
    style   = school_cfg["style"]
    slug    = _elumen_slug(code)
    url = f"https://api-prod.elumenapp.com/catalog/sites/publish/content/{catalog},course,{slug}?tenant={tenant}"
    r = requests_get(url)
    if not r:
        return None
    # If 404-like, try fallback via dept navitems (handled in caller)
    desc = _parse_elumen_desc(r.text, style)
    return desc if desc and len(desc) > 20 else None


# Dept slug caches per school (prefix -> list of (slug, code) from navitems)
_elumen_dept_cache = {}   # (school_id, dept_slug) -> {code: slug}
_elumen_prefix_map = {}   # (school_id, prefix) -> dept_slug or None


def _get_dvc_dept_for_prefix(prefix):
    """Return DVC dept slug for a course prefix (best-effort static map)."""
    P = prefix.upper()
    TABLE = {
        "MATH":   "dvc-mathematics-course",
        "CHEM":   "dvc-chemistry-course",
        "PHYS":   "dvc-physics-course",
        "COMSC":  "dvc-computer-science-course",
        "CS":     "dvc-computer-science-course",
        "BIO":    "dvc-biological-science-course",
        "BIOL":   "dvc-biological-science-course",
        "PSYC":   "dvc-psychology-course",
        "PSYCH":  "dvc-psychology-course",
        "ECON":   "dvc-economics-course",
        "HIST":   "dvc-history-course",
        "ENGL":   "dvc-english-course",
        "SOC":    "dvc-sociology-course",
        "SOCIO":  "dvc-sociology-course",
        "PHILO":  "dvc-philosophy-course",
        "PHIL":   "dvc-philosophy-course",
        "ANTH":   "dvc-anthropology-course",
        "ANTHR":  "dvc-anthropology-course",
        "ART":    "dvc-art-digital-media-course",
        "ARTS":   "dvc-art-digital-media-course",
        "ARTDM":  "dvc-art-digital-media-course",
        "ARTH":   "dvc-art-history-course",
        "POLI":   "dvc-political-science-course",
        "POLSC":  "dvc-political-science-course",
        "SPAN":   "dvc-spanish-course",
        "FREN":   "dvc-french-course",
        "FR":     "dvc-french-course",
        "FRENCH": "dvc-french-course",
        "GERM":   "dvc-german-course",
        "GERMAN": "dvc-german-course",
        "GRMAN":  "dvc-german-course",
        "GRMN":   "dvc-german-course",
        "ITAL":   "dvc-italian-course",
        "JAPAN":  "dvc-japanese-course",
        "JAPN":   "dvc-japanese-course",
        "CHIN":   "dvc-chinese-course",
        "GEOL":   "dvc-geology-course",
        "DANC":   "dvc-dance-course",
        "DANCE":  "dvc-dance-course",
        "DRAMA":  "dvc-drama-course",
        "THTR":   "dvc-drama-course",
        "THEA":   "dvc-drama-course",
        "MUS":    "dvc-music-course",
        "MUSI":   "dvc-music-course",
        "FILM":   "dvc-film-television-electronic-media-course",
        "FTVE":   "dvc-film-television-electronic-media-course",
        "ENGIN":  "dvc-engineering-course",
        "ENGR":   "dvc-engineering-course",
        "BUS":    "dvc-business-course",
        "ACCT":   "dvc-business-accounting-course",
        "ACCTG":  "dvc-business-accounting-course",
        "ACTG":   "dvc-business-accounting-course",
        "ECE":    "dvc-early-childhood-education-course",
        "ETHN":   "dvc-ethnic-studies-course",
        "ES":     "dvc-ethnic-studies-course",
        "LING":   "dvc-english-course",
        "PHOTO":  "dvc-art-photography-course",
        "PHTG":   "dvc-art-photography-course",
    }
    return TABLE.get(P)


def _get_elumen_dept_courses(school_id, dept_slug):
    """Fetch navitems from a DVC dept page; returns {code: course_slug} map."""
    key = (school_id, dept_slug)
    if key in _elumen_dept_cache:
        return _elumen_dept_cache[key]
    cfg = ELUMEN_SCHOOLS[school_id]
    catalog = cfg["catalog"]
    tenant  = cfg["tenant"]
    url = f"https://api-prod.elumenapp.com/catalog/sites/publish/{catalog},{dept_slug}?tenant={tenant}&api=https://api-prod.elumenapp.com:443"
    r = requests_get(url)
    code_to_slug = {}
    if r:
        data = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
        html = data.get("html", "")
        slugs = re.findall(rf'class="navitem"[^>]*href="{re.escape(catalog)}/course/([^"]+)"', html)
        for slug in slugs:
            # Try to match slug back to a code
            code_to_slug[slug] = slug   # raw slug; caller will match
    _elumen_dept_cache[key] = code_to_slug
    time.sleep(CATALOG_DELAY)
    return code_to_slug


def _fetch_elumen_course_via_dept(school_id, code, prefix):
    """Fallback: scan dept navitems to find the right slug, then fetch content."""
    if school_id != 114:   # fallback only implemented for DVC
        return None
    dept_slug = _get_dvc_dept_for_prefix(prefix)
    if not dept_slug:
        return None
    raw_slugs = _get_elumen_dept_courses(school_id, dept_slug)
    # Derive expected slug from code
    expected = _elumen_slug(code)
    if expected in raw_slugs:
        return _fetch_elumen_course(ELUMEN_SCHOOLS[school_id], code)
    # Try partial match: code number part
    m = re.match(r"[A-Z ]+(\S+)$", code.strip())
    if m:
        num_part = m.group(1).lower()
        prefix_part = prefix.lower().replace(" ", "")
        for raw_slug in raw_slugs:
            if raw_slug.endswith(num_part) or raw_slug == prefix_part + num_part:
                cfg = ELUMEN_SCHOOLS[school_id]
                url = f"https://api-prod.elumenapp.com/catalog/sites/publish/content/{cfg['catalog']},course,{raw_slug}?tenant={cfg['tenant']}"
                r = requests_get(url)
                if r:
                    return _parse_elumen_desc(r.text, cfg["style"])
    return None


# De Anza dept prefix map: CCC prefix -> De Anza course-offerings slug
# (prefix -> slug that ends in '-courses')
_DEANZA_PREFIX_MAP = {}
_deanza_slugs_loaded = False

def _load_deanza_slugs():
    global _deanza_slugs_loaded
    if _deanza_slugs_loaded:
        return
    r = requests_get("https://api-prod.elumenapp.com/catalog/sites/publish/2025-2026,course-offerings?tenant=deanza.elumenapp.com&api=https://api-prod.elumenapp.com:443")
    if not r:
        _deanza_slugs_loaded = True
        return
    try:
        data = r.json()
    except Exception:
        _deanza_slugs_loaded = True
        return
    html = data.get("html", "")
    slugs = re.findall(r'href="2025-2026/([^"]+)"', html)
    dept_slugs = [s for s in slugs if s.endswith("-courses") and s != "repeating-courses"]
    # Build prefix map: first word of slug (uppercased) -> slug
    for slug in dept_slugs:
        first = slug.split("-")[0].upper()
        if first not in _DEANZA_PREFIX_MAP:
            _DEANZA_PREFIX_MAP[first] = slug
    _deanza_slugs_loaded = True
    time.sleep(CATALOG_DELAY)


# Cache of De Anza dept slug -> {course_code: course_slug}
_deanza_dept_courses = {}   # dept_slug -> {code_normalized: course_slug}


def _load_deanza_dept(dept_slug):
    if dept_slug in _deanza_dept_courses:
        return _deanza_dept_courses[dept_slug]
    r = requests_get(f"https://api-prod.elumenapp.com/catalog/sites/publish/2025-2026,{dept_slug}?tenant=deanza.elumenapp.com&api=https://api-prod.elumenapp.com:443")
    mapping = {}
    if r:
        try:
            data = r.json()
        except Exception:
            _deanza_dept_courses[dept_slug] = mapping
            return mapping
        html = data.get("html", "")
        soup = BeautifulSoup(html, "html.parser")
        for item in soup.find_all("div", class_="course-item"):
            data_code = item.get("data-code", "")
            a = item.find("a", href=True)
            if data_code and a:
                href = a["href"]   # e.g. "2025-2026/course/acctd001a"
                course_slug = href.split("/course/")[-1] if "/course/" in href else ""
                if course_slug:
                    mapping[data_code.upper()] = course_slug
    _deanza_dept_courses[dept_slug] = mapping
    time.sleep(CATALOG_DELAY)
    return mapping


_DEANZA_EXTRA_PREFIX = {
    # Alternate / non-obvious prefix → De Anza slug overrides
    "C S":     "cis-computer-sceince-and-information-systems-courses",
    "CS":      "cis-computer-sceince-and-information-systems-courses",
    "ACCT":    "accounting-courses",
    "ACCTG":   "accounting-courses",
    "ACTG":    "accounting-courses",
    "BUSAC":   "accounting-courses",
    "PSYC":    "psyc-psychology-courses",
    "PSYCH":   "psyc-psychology-courses",
    "PHYS":    "phys-physics-courses",
    "SOC":     "soc-sociology-courses",
    "SOCIO":   "soc-sociology-courses",
    "SOCIOL":  "soc-sociology-courses",
    "PHIL":    "phil-philosophy-courses",
    "PHILO":   "phil-philosophy-courses",
    "ANTH":    "anth-anthropology-courses",
    "ANTHR":   "anth-anthropology-courses",
    "ANTHRO":  "anth-anthropology-courses",
    "FREN":    "fren-french-courses",
    "FR":      "fren-french-courses",
    "FRENCH":  "fren-french-courses",
    "GERM":    "germ-german-language-courses",
    "GERMAN":  "germ-german-language-courses",
    "GRMAN":   "germ-german-language-courses",
    "GRMN":    "germ-german-language-courses",
    "ITAL":    "ital-italian-courses",
    "JAPAN":   "japan-japanese-courses",
    "JAPN":    "japan-japanese-courses",
    "CHIN":    "chin-chinese-courses",
    "MAND":    "chin-chinese-courses",
    "SPAN":    "span-spanish-courses",
    "DANC":    "danc-dance-courses",
    "DANCE":   "danc-dance-courses",
    "MUS":     "mus-music-courses",
    "MUSI":    "mus-music-courses",
    "ART":     "arts-art-courses",
    "ARTDM":   "artdm-art-digital-media-courses",
    "DM":      "artdm-art-digital-media-courses",
    "GID":     "gid-graphic-and-interactive-design-courses",
    "PHOTO":   "photo-photography-courses",
    "PHTG":    "photo-photography-courses",
    "FTVE":    "ftve-film-television-and-electronic-media-courses",
    "FILM":    "ftve-film-television-and-electronic-media-courses",
    "ENGR":    "engr-engineering-courses",
    "ENGIN":   "engr-engineering-courses",
    "ENGL":    "ewrt-english-writing-courses",
    "EWRT":    "ewrt-english-writing-courses",
    "ETHN":    "ethn-ethnic-studies-courses",
    "ES":      "ethn-ethnic-studies-courses",
    "CETH":    "ceth-comparative-ethnic-studies-courses",
    "CHLX":    "chlx-chicanx-latinx-studies-courses",
    "GEOL":    "geol-geology-courses",
    "ECON":    "econ-economics-courses",
    "HIST":    "hist-history-courses",
    "BUS":     "bus-business-courses",
    "BUSI":    "bus-business-courses",
    "INTL":    "intl-international-studies-courses",
    "WGS":     "wgss-womens-gender-and-sexuality-studies-courses",
    "WMN":     "wgss-womens-gender-and-sexuality-studies-courses",
    "WMST":    "wgss-womens-gender-and-sexuality-studies-courses",
    "WS":      "wgss-womens-gender-and-sexuality-studies-courses",
    "ECE":     "cd-child-development-courses",
    "C D":     "cd-child-development-courses",
    "GEOG":    "geog-geography-courses",
    "LING":    "ling-linguistics-courses",
    "EDAC":    "edac-educational-achievement-courses",
    "EDUC":    "educ-education-courses",
    "DRAMA":   "thtr-theatre-arts-courses",
    "THTR":    "thtr-theatre-arts-courses",
    "THEA":    "thtr-theatre-arts-courses",
    "TA":      "thtr-theatre-arts-courses",
    "TH ART":  "thtr-theatre-arts-courses",
    "AP":      "admj-administration-of-justice-courses",
    "ADMJ":    "admj-administration-of-justice-courses",
    "ASTRO":   "astr-astronomy-courses",
    "ASTR":    "astr-astronomy-courses",
    "HS":      "hlth-health-courses",
}


def _fetch_deanza_course(code):
    """Fetch De Anza description: look up via dept course-item listing."""
    _load_deanza_slugs()
    code_upper = code.upper()
    prefix = re.match(r"([A-Z][A-Z0-9 ]*?)\s+\S", code_upper)
    if not prefix:
        return None
    pref = prefix.group(1).strip()

    # Try extra map first (handles non-obvious prefixes), then auto map
    slug = _DEANZA_EXTRA_PREFIX.get(pref)
    if not slug:
        slug = _DEANZA_PREFIX_MAP.get(pref)
    if not slug:
        first_tok = pref.split()[0]
        slug = _DEANZA_EXTRA_PREFIX.get(first_tok) or _DEANZA_PREFIX_MAP.get(first_tok)
    if not slug:
        return None

    dept_map = _load_deanza_dept(slug)
    # Try exact code key
    course_slug = dept_map.get(code_upper)
    if not course_slug:
        # Try derived slug directly
        course_slug = _elumen_slug(code)

    cfg = ELUMEN_SCHOOLS[113]
    url = f"https://api-prod.elumenapp.com/catalog/sites/publish/content/{cfg['catalog']},course,{course_slug}?tenant={cfg['tenant']}"
    r = requests_get(url)
    if not r:
        return None
    return _parse_elumen_desc(r.text, cfg["style"])


# Cabrillo dept cache: dept_name -> {code: slug}
_cabrillo_dept_courses = {}   # dept_name -> {code_upper: course_slug}

# Prefix -> cabrillo dept name (best-effort)
_CABRILLO_PREFIX_MAP = {
    "MATH": "mathematics", "CHEM": "chemistry", "PHYS": "physics",
    "BIO": "biology", "BIOL": "biology", "CS": "computer-science",
    "COMSC": "computer-science", "ECON": "economics", "HIST": "history",
    "ENGL": "english", "SOC": "sociology", "SOCIO": "sociology",
    "PHIL": "philosophy", "PHILO": "philosophy", "PHILOS": "philosophy",
    "ANTH": "anthropology", "ANTHR": "anthropology", "ANTHRO": "anthropology",
    "ART": "art", "ARTH": "art-history", "AHIS": "art-history",
    "POLI": "political-science", "POLSC": "political-science", "PS": "political-science",
    "SPAN": "spanish", "FREN": "french", "FR": "french", "FRENCH": "french",
    "GERM": "german", "GERMAN": "german", "GRMAN": "german", "GRMN": "german",
    "ITAL": "italian", "JAPAN": "japanese", "JAPN": "japanese",
    "CHIN": "chinese", "GEOL": "geology", "DANC": "dance", "DANCE": "dance",
    "DRAMA": "theatre-arts", "THTR": "theatre-arts", "THEA": "theatre-arts",
    "TA": "theatre-arts", "TH ART": "theatre-arts",
    "MUS": "music", "MUSI": "music",
    "FILM": "digital-media", "MDIA": "digital-media", "MEDIA": "digital-media",
    "ENGIN": "engineering", "ENGR": "engineering",
    "BUS": "business", "ACCT": "accounting", "ACCTG": "accounting", "ACTG": "accounting",
    "ECE": "early-childhood-education", "ETHN": "ethnic-studies", "ES": "ethnic-studies",
    "PHOTO": "art-photography", "PHTG": "art-photography",
    "PSYC": "psychology", "PSYCH": "psychology",
    "ARTS": "art", "ARTDM": "art",
    "LING": "english", "EWRT": "english",
    "C S": "computer-science", "ICS": "computer-and-information-systems",
    "CIS": "computer-and-information-systems",
    "FTVE": "digital-media", "GID": "digital-media", "DM": "digital-media",
    "GEOG": "geography",
}


def _load_cabrillo_dept(dept_name):
    if dept_name in _cabrillo_dept_courses:
        return _cabrillo_dept_courses[dept_name]
    cfg = ELUMEN_SCHOOLS[41]
    url = f"https://api-prod.elumenapp.com/catalog/sites/publish/content/{cfg['catalog']},department,{dept_name}?tenant={cfg['tenant']}"
    r = requests_get(url)
    mapping = {}
    if r:
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]   # e.g. "2025-2026/course/math4"
            if "/course/" in href:
                text = a.get_text(strip=True)
                # text like "MATH 4 - Precalculus..." => extract code
                m = re.match(r"([A-Z][A-Z0-9 ]*?\s+\S+)\s*[-–]", text)
                if m:
                    code = m.group(1).strip().upper()
                    slug = href.split("/course/")[-1]
                    mapping[code] = slug
    _cabrillo_dept_courses[dept_name] = mapping
    time.sleep(CATALOG_DELAY)
    return mapping


def _fetch_cabrillo_course(code):
    prefix_m = re.match(r"([A-Z][A-Z0-9 ]*?)\s+\S", code.upper())
    if not prefix_m:
        return None
    pref = prefix_m.group(1).strip()
    dept_name = _CABRILLO_PREFIX_MAP.get(pref)
    if not dept_name:
        return None
    dept_map = _load_cabrillo_dept(dept_name)
    # Try exact match then derived slug
    course_slug = dept_map.get(code.upper()) or _elumen_slug(code)
    cfg = ELUMEN_SCHOOLS[41]
    url = f"https://api-prod.elumenapp.com/catalog/sites/publish/content/{cfg['catalog']},course,{course_slug}?tenant={cfg['tenant']}"
    r = requests_get(url)
    if not r:
        return None
    return _parse_elumen_desc(r.text, cfg["style"])


# ── Foothill College (id=51) ──────────────────────────────────

_foothill_dept_cache = {}   # prefix -> {code: {title, desc}}

_FOOTHILL_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def _load_foothill_prefix(prefix):
    if prefix in _foothill_dept_cache:
        return _foothill_dept_cache[prefix]
    url = f"https://catalog.foothill.edu/courses-az/{prefix.lower().replace(' ', '-')}/"
    r = session.get(url, headers=_FOOTHILL_HEADERS, timeout=20)
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
                # Trim trailing metadata
                desc = re.sub(r"\s*(Prerequisite|Advisory|Formerly|Transfer Credit|UC|CSU).*", "", desc, flags=re.S).strip()
                title = title_el.get_text(strip=True) if title_el else ""
                if code:
                    courses[code] = {"title": title, "desc": desc if len(desc) > 20 else "Description not found"}
    _foothill_dept_cache[prefix] = courses
    time.sleep(CATALOG_DELAY)
    return courses


# ── Santa Monica College (id=137) ────────────────────────────

_smc_subject_cache = {}   # subject_name -> {code: {title, desc}}

_SMC_PREFIX_MAP = {
    "MATH":   "Mathematics", "STAT":   "Mathematics",
    "CHEM":   "Chemistry",   "PHYS":   "Physics",
    "BIO":    "Biological Sciences", "BIOL": "Biological Sciences",
    "BIOSC":  "Biological Sciences",
    "CS":     "Computer Science",    "ICS":  "Computer Science",
    "COMSC":  "Computer Science",    "CIS":  "Computer Information Systems",
    "ECON":   "Economics",   "HIST":   "History",
    "ENGL":   "English – Composition",
    "SOC":    "Sociology",   "SOCIO":  "Sociology",  "SOCIOL": "Sociology",
    "PHIL":   "Philosophy",  "PHILO":  "Philosophy", "PHILOS": "Philosophy",
    "ANTH":   "Anthropology", "ANTHR": "Anthropology", "ANTHRO": "Anthropology",
    "ART":    "Art",         "ARTS":   "Art",         "ARTDM":  "Art",
    "AHIS":   "Art History", "ARTH":   "Art History",
    "POLI":   "Political Science", "POLSC": "Political Science",
    "PS":     "Political Science",  "POL SC": "Political Science",
    "SPAN":   "Spanish",     "FREN":   "French",      "FR":     "French",
    "FRENCH": "French",      "FRNCH":  "French",
    "GERM":   "German",      "GERMAN": "German",      "GRMAN":  "German",
    "GRMN":   "German",
    "ITAL":   "Italian",     "JAPAN":  "Japanese",    "JAPN":   "Japanese",
    "CHIN":   "Chinese",     "MAND":   "Chinese",
    "GEOL":   "Geology",
    "DANC":   "Dance: Technique and Performance",
    "DANCE":  "Dance: Technique and Performance",
    "DRAMA":  "Theatre Arts","THTR":   "Theatre Arts","THEA":   "Theatre Arts",
    "TA":     "Theatre Arts","TH ART": "Theatre Arts",
    "MUS":    "Music: Theory, Performance, and Application",
    "MUSI":   "Music: Theory, Performance, and Application",
    "FILM":   "Film Studies","FTVE":   "Film Studies","MDIA":   "Media Studies",
    "MEDIA":  "Media Studies",
    "ENGIN":  "Engineering", "ENGR":   "Engineering",
    "BUS":    "Business",    "ACCT":   "Accounting",
    "ACCTG":  "Accounting",  "ACTG":   "Accounting",  "BUSAC":  "Accounting",
    "ECE":    "Early Childhood Education",
    "ETHN":   "Ethnic Studies", "ES":  "Ethnic Studies",
    "PHOTO":  "Photography", "PHTG":   "Photography",
    "PSYC":   "Psychology",  "PSYCH":  "Psychology",
    "WGS":    "Women's, Gender, and Sexuality Studies",
    "WMN":    "Women's, Gender, and Sexuality Studies",
    "WMST":   "Women's, Gender, and Sexuality Studies",
    "WS":     "Women's, Gender, and Sexuality Studies",
    "GEOG":   "Geography",   "LING":   "Linguistics",
    "GID":    "Graphic Design", "ARTG":  "Graphic Design",
    "PHSC":   "Physics",
    "C D":    "Early Childhood Education",
    "CETH":   "Ethnic Studies", "CHLX": "Ethnic Studies",
    "EWRT":   "English – Composition",
    "AP":     "Administration of Justice",
    "ADMJ":   "Administration of Justice",
    "DM":     "Digital Media Post-Production",
    "PH":     "Philosophy",  "BUSI":   "Business",
    "INTL":   "Global Studies",
    "EDAC":   "Education",   "EDUC":   "Education",
    "ENSC":   "Environmental Studies",  "ESCI": "Environmental Studies",
    "ETECH":  "Engineering",
    "ASTRO":  "Astronomy",   "ASTR":   "Astronomy",
    "HS":     "Health Education",
}

_SMC_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def _load_smc_subject(subject_name):
    if subject_name in _smc_subject_cache:
        return _smc_subject_cache[subject_name]
    import urllib.parse
    url = f"https://catalog.smc.edu/current/courses/subject-finder.php?subject={urllib.parse.quote(subject_name)}"
    r = session.get(url, headers=_SMC_HEADERS, timeout=20)
    courses = {}
    if r and r.status_code == 200:
        soup = BeautifulSoup(r.text, "html.parser")
        for row in soup.find_all("tr", role="row"):
            th = row.find("th", class_="sorting_1")
            td = row.find("td")
            if not th or not td:
                continue
            code_span = th.find(class_="type")
            code = code_span.get_text(strip=True) if code_span else th.get_text(strip=True)
            multi = td.find(class_="multi-row")
            if not multi:
                continue
            title_el = multi.find(class_="title-holder")
            title = title_el.get_text(strip=True) if title_el else ""
            # Description: the last <p> that isn't C-ID/advisory/prereq/transfer
            desc = ""
            for p in multi.find_all("p"):
                text = p.get_text(strip=True)
                if text and not re.match(r"^(C-ID:|Cal-GETC|Formerly|Note:|This course fulfills|Satisfies)", text):
                    desc = text
            if not desc:
                # Fallback: any text after last <ul>
                desc = multi.get_text(separator=" ", strip=True)
                desc = re.sub(r".*(?:Advisory|Prerequisite):[^.]+\.\s*", "", desc, flags=re.S).strip()
            if code:
                courses[code] = {"title": title, "desc": desc if len(desc) > 20 else "Description not found"}
    _smc_subject_cache[subject_name] = courses
    time.sleep(CATALOG_DELAY)
    return courses


# ── Main CCC dispatcher ───────────────────────────────────────

def scrape_ccc_descriptions(course_pairs):
    print("\n" + "=" * 60)
    print("STEP 4: Scraping CCC course descriptions")
    print("=" * 60)

    # Group by (school_id, code)
    needed = {}   # (ccc_id, ccc_code) -> ccc_name
    for pair in course_pairs:
        key = (pair["ccc_id"], pair["ccc_code"])
        if key not in needed:
            needed[key] = pair["ccc_name"]

    ccc_cache = {}
    total = len(needed)
    n_found = 0
    n_processed = 0
    last_print = 0

    for (ccc_id, code), ccc_name in sorted(needed.items()):
        desc = None
        prefix_m = re.match(r"([A-Z][A-Z0-9 ]*?)\s+\S", code.upper())
        prefix = prefix_m.group(1).strip() if prefix_m else ""

        if ccc_id in ELUMEN_SCHOOLS:
            cfg = ELUMEN_SCHOOLS[ccc_id]
            if ccc_id == 113:     # De Anza
                desc = _fetch_deanza_course(code)
            elif ccc_id == 41:    # Cabrillo
                desc = _fetch_cabrillo_course(code)
            elif ccc_id == 114:   # DVC — try direct slug first, then dept fallback
                desc = _fetch_elumen_course(cfg, code)
                if not desc and prefix:
                    desc = _fetch_elumen_course_via_dept(ccc_id, code, prefix)
            time.sleep(CATALOG_DELAY)

        elif ccc_id == 51:        # Foothill
            if prefix:
                dept_courses = _load_foothill_prefix(prefix)
                info = dept_courses.get(code.upper()) or dept_courses.get(code)
                if info:
                    desc = info["desc"]

        elif ccc_id == 137:       # SMC
            if prefix:
                subject = _SMC_PREFIX_MAP.get(prefix.upper())
                if subject:
                    subj_courses = _load_smc_subject(subject)
                    info = subj_courses.get(code.upper()) or subj_courses.get(code)
                    if info:
                        desc = info["desc"]

        if desc and len(desc) > 20:
            ccc_cache[(ccc_id, code)] = {"desc": desc, "title": ""}
            n_found += 1
        else:
            ccc_cache[(ccc_id, code)] = {"desc": "Description not found", "title": ""}

        n_processed += 1
        if n_processed - last_print >= 50 or n_processed == total:
            print(f"  [{n_processed}/{total}] found so far: {n_found}  ({ccc_name})")
            last_print = n_processed

    print(f"\nCCC: {n_found}/{total} descriptions found")
    return ccc_cache


# ══════════════════════════════════════════════════════════════
# STEP 5 — Assemble CSV
# ══════════════════════════════════════════════════════════════

def assemble_csv(course_pairs, ucsc_cache, ccc_cache):
    print("\n" + "=" * 60)
    print("STEP 5: Assembling CSV")
    print("=" * 60)

    seen = set()
    output_rows = []

    for pair in course_pairs:
        dedup_key = (pair["ccc_id"], pair["ccc_code"], pair["ucsc_code"])
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        ccc_code  = pair["ccc_code"]
        ccc_title = pair["ccc_title"]
        ucsc_code = pair["ucsc_code"]
        ucsc_title_art = pair["ucsc_title"]

        ucsc_info  = ucsc_cache.get(ucsc_code, {"title": ucsc_title_art, "desc": "Description not found"})
        ucsc_title = ucsc_info.get("title") or ucsc_title_art
        ucsc_desc  = ucsc_info.get("desc", "Description not found")

        ccc_info = ccc_cache.get((pair["ccc_id"], ccc_code), {"title": ccc_title, "desc": "Description not found"})
        ccc_desc = ccc_info.get("desc", "Description not found")

        output_rows.append({
            "CCC Course":        f"{ccc_code} {ccc_title}".strip(),
            "CCC Description":   ccc_desc,
            "UCSC Course Code":  ucsc_code,
            "UCSC Course Title": ucsc_title,
            "UCSC Description":  ucsc_desc,
            "_ccc_name":         pair["ccc_name"],
        })

    print(f"Total deduplicated rows: {len(output_rows)}")
    return output_rows


# ══════════════════════════════════════════════════════════════
# STEP 6 — Validate
# ══════════════════════════════════════════════════════════════

def validate(output_rows):
    print("\n" + "=" * 60)
    print("STEP 6: Validation")
    print("=" * 60)

    total = len(output_rows)
    if total == 0:
        print("ERROR: 0 rows!")
        return

    ccc_miss  = sum(1 for r in output_rows if r["CCC Description"]  == "Description not found")
    ucsc_miss = sum(1 for r in output_rows if r["UCSC Description"] == "Description not found")
    schools   = len(set(r["_ccc_name"] for r in output_rows))

    print(f"Total rows:         {total}")
    print(f"Unique CCC schools: {schools}")
    print(f"CCC desc missing:   {ccc_miss} ({ccc_miss/total*100:.1f}%) {'⚠' if ccc_miss/total > 0.5 else '✓'}")
    print(f"UCSC desc missing:  {ucsc_miss} ({ucsc_miss/total*100:.1f}%) {'⚠' if ucsc_miss/total > 0.2 else '✓'}")

    import random as rnd
    rnd.seed(99)
    for i, r in enumerate(rnd.sample(output_rows, min(5, total))):
        print(f"\n  [{i+1}] {r['_ccc_name']}")
        print(f"    CCC:  {r['CCC Course'][:70]}")
        print(f"    Desc: {r['CCC Description'][:90]}...")
        print(f"    UCSC: {r['UCSC Course Code']} — {r['UCSC Course Title'][:50]}")
        print(f"    Desc: {r['UCSC Description'][:90]}...")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

async def main():
    course_pairs = await scrape_assist()

    if not course_pairs:
        print("ERROR: No course pairs found!")
        return

    all_ucsc_codes = set(p["ucsc_code"] for p in course_pairs)
    print(f"\nUnique UCSC codes: {len(all_ucsc_codes)}")
    print(f"Unique CCC schools: {len(set(p['ccc_name'] for p in course_pairs))}")

    ucsc_cache = scrape_ucsc_descriptions(all_ucsc_codes)
    ccc_cache  = scrape_ccc_descriptions(course_pairs)
    output_rows = assemble_csv(course_pairs, ucsc_cache, ccc_cache)
    validate(output_rows)

    out_path = CCC_UCSC_MERGED
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", "CCC Course", "CCC Description",
                         "UCSC Course Code", "UCSC Course Title", "UCSC Description"])
        for i, row in enumerate(output_rows):
            writer.writerow([i,
                             row["CCC Course"],
                             row["CCC Description"],
                             row["UCSC Course Code"],
                             row["UCSC Course Title"],
                             row["UCSC Description"]])

    print(f"\nSaved {len(output_rows)} rows to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
