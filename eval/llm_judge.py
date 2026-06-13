"""
LLM-as-judge for TransfersAI.

Evaluates whether two courses are equivalent for transfer credit
using Claude as an independent judge. Used to audit the XGBoost
reranker on hard cases and understand systematic failure modes.
"""

import os
import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Load .env from project root if present (ANTHROPIC_API_KEY, etc.)
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def judge_pair(
    source_title: str,
    source_dept: str,
    source_desc: str,
    target_code: str,
    target_title: str,
    target_desc: str,
    source_institution: str = "Virginia Community College System",
    target_institution: str = "William & Mary",
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """
    Ask Claude whether two courses are equivalent for transfer credit.

    Returns:
        {
            "judgment": "equivalent" | "not_equivalent",
            "confidence": "high" | "medium" | "low",
            "reasoning": str,
            "raw": str,         # raw Claude response for debugging
        }
    """
    prompt = f"""You are an expert academic registrar evaluating transfer credit equivalency.

SOURCE COURSE ({source_institution}):
  Department: {source_dept}
  Title: {source_title}
  Description: {source_desc or "(no description provided)"}

TARGET COURSE ({target_institution}):
  Code: {target_code}
  Title: {target_title}
  Description: {target_desc or "(no description provided)"}

Are these courses equivalent for transfer credit? A course is equivalent if a student who completed the source course has mastered substantially the same material as the target course — same subject area, comparable depth and level.

Respond with valid JSON only, no other text:
{{
  "judgment": "equivalent" or "not_equivalent",
  "confidence": "high" or "medium" or "low",
  "reasoning": "one sentence explaining the key factor"
}}"""

    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    try:
        # Strip markdown code fences if present
        text = raw
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
        judgment = parsed.get("judgment", "").lower()
        if judgment not in ("equivalent", "not_equivalent"):
            judgment = "not_equivalent"
        return {
            "judgment": judgment,
            "confidence": parsed.get("confidence", "low").lower(),
            "reasoning": parsed.get("reasoning", ""),
            "raw": raw,
            "error": None,
        }
    except (json.JSONDecodeError, KeyError) as e:
        return {
            "judgment": "not_equivalent",
            "confidence": "low",
            "reasoning": "",
            "raw": raw,
            "error": str(e),
        }
