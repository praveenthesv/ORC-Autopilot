import os
import json
import re
import base64
import io
from typing import Any, Dict, List

import requests
from flask import Flask, request, jsonify
from pdfminer.high_level import extract_text

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"


SKILL_SYNONYMS = {
    "orc": "oracle recruiting cloud",
    "oracle recruiting cloud (orc)": "oracle recruiting cloud",
    "oracle recruiting cloud": "oracle recruiting cloud",
    "core hr": "core hr",
    "redwood development": "redwood",
    "redwood development & migration": "redwood",
    "rest apis": "rest api",
    "vb customization": "visual builder",
    "vb": "visual builder",
    "otbi/obiee": "obiee",
}


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def normalize_text(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[()\[\]]", " ", value)
    value = re.sub(r"[^a-z0-9+#/.& -]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def canonicalize_skill(skill: str) -> str:
    s = normalize_text(skill)
    return SKILL_SYNONYMS.get(s, s)


def normalize_list(text: str) -> List[str]:
    if not text:
        return []
    raw = [x.strip() for x in re.split(r"[,\n;]+", text) if x.strip()]
    return [canonicalize_skill(x) for x in raw]


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        key = normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def safe_string(value: Any, max_len: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:max_len]
    return str(value)[:max_len]


# -----------------------------------------------------------------------------
# Resume text extraction
# -----------------------------------------------------------------------------

def extract_text_from_pdf_base64(resume_b64: str) -> str:
    pdf_bytes = base64.b64decode(resume_b64)
    with io.BytesIO(pdf_bytes) as pdf_stream:
        text = extract_text(pdf_stream) or ""
    return text


def extract_years_from_text(text: str) -> int:
    t = text.lower()

    # Handles: "over 11 years", "11 years", "12+ years"
    matches = re.findall(r"(\d{1,2})\s*\+?\s*years", t)
    years = [int(m) for m in matches] if matches else []

    # Handles: "over 11 years of global experience"
    matches2 = re.findall(r"over\s+(\d{1,2})\s+years", t)
    years.extend(int(m) for m in matches2)

    return max(years) if years else 0


def extract_education_from_text(text: str) -> str:
    t = text.lower()

    if "bachelor of engineering" in t or "b.e." in t or "(b.e.)" in t:
        return "B.E."
    if "b.tech" in t or "btech" in t:
        return "B.TECH"
    if "m.tech" in t or "mtech" in t:
        return "M.TECH"
    if "master of business administration" in t or " mba " in f" {t} ":
        return "MBA"
    if "bachelor" in t:
        return "BACHELOR"
    if "master" in t:
        return "MASTER"
    return ""


def extract_section(text: str, start_header: str, end_headers: List[str]) -> str:
    lines = text.splitlines()
    capture = False
    captured: List[str] = []

    start_header_norm = normalize_text(start_header)
    end_headers_norm = [normalize_text(h) for h in end_headers]

    for line in lines:
        line_norm = normalize_text(line)

        if not capture and start_header_norm == line_norm:
            capture = True
            continue

        if capture:
            if line_norm in end_headers_norm:
                break
            captured.append(line)

    return "\n".join(captured).strip()


def extract_certifications_rule_based(text: str) -> str:
    section = extract_section(
        text,
        "Certifications",
        ["Notable Projects", "Latest Achievement", "Technical Stack", "Functional Architecture Highlights by Module"],
    )

    if not section:
        return ""

    certs = []
    for line in section.splitlines():
        line = line.strip(" •\t-")
        if line:
            certs.append(line)

    certs = unique_keep_order(certs)
    return ", ".join(certs)[:2000]


def extract_skills_rule_based(text: str) -> str:
    section = extract_section(
        text,
        "Core Competencies",
        ["Professional Experience", "Education", "Certifications"],
    )

    skills: List[str] = []
    if section:
        for line in section.splitlines():
            line = line.strip(" •\t-")
            if line:
                parts = [p.strip() for p in re.split(r",|/", line) if p.strip()]
                skills.extend(parts)

    skills = unique_keep_order(skills)
    return ", ".join(skills)[:2000]


def extract_experience_summary_rule_based(text: str) -> str:
    section = extract_section(
        text,
        "Professional Experience",
        ["Education", "Certifications", "Notable Projects"],
    )

    if not section:
        return ""

    lines = []
    for line in section.splitlines():
        clean = line.strip()
        if clean:
            lines.append(clean)

    return " | ".join(lines)[:2000]


# -----------------------------------------------------------------------------
# Groq helpers
# -----------------------------------------------------------------------------

def groq_chat_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    if not GROQ_API_KEY:
        return {}

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        },
        timeout=60,
    )

    if not response.ok:
        return {}

    ai_text = response.json()["choices"][0]["message"]["content"].strip()
    ai_text = ai_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(ai_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", ai_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}
    return {}


def flatten_experience(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value[:2000]

    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                position = safe_string(item.get("position", ""), 200)
                company = safe_string(item.get("company", ""), 200)
                duration = safe_string(item.get("duration", ""), 100)
                line = " - ".join([x for x in [position, company] if x])
                if duration:
                    line = f"{line} ({duration})" if line else duration
                if line:
                    parts.append(line)
            else:
                parts.append(safe_string(item, 300))
        return " | ".join(parts)[:2000]

    if isinstance(value, dict):
        return json.dumps(value)[:2000]

    return safe_string(value, 2000)


def extract_resume_fields_with_groq(resume_text: str) -> Dict[str, Any]:
    if not GROQ_API_KEY:
        return {
            "parsed_skills": "",
            "parsed_certifications": "",
            "parsed_experience": "",
            "parsed_summary": "",
        }

    prompt = f"""
Extract candidate details from the resume text below.
Return ONLY valid JSON with these keys:
- parsed_skills (comma separated)
- parsed_certifications
- parsed_experience
- parsed_summary

Rules:
- Use ONLY facts explicitly present in the resume text.
- DO NOT infer or assume anything not written.
- Keep parsed_experience concise.

Resume Text:
{resume_text}
"""

    parsed = groq_chat_json(
        "You are an AI resume parser. Return only JSON.",
        prompt,
    )

    result = {
        "parsed_skills": safe_string(parsed.get("parsed_skills", ""), 2000),
        "parsed_certifications": safe_string(parsed.get("parsed_certifications", ""), 2000),
        "parsed_experience": flatten_experience(parsed.get("parsed_experience", "")),
        "parsed_summary": safe_string(parsed.get("parsed_summary", ""), 4000),
    }
    return result


def build_ai_summary(candidate_name: str, role_name: str, score: int, decision: str, gaps: str) -> str:
    if not GROQ_API_KEY:
        return f"Candidate {candidate_name} evaluated for {role_name}. Score: {score}. Decision: {decision}."

    prompt = f"""
Return ONLY valid JSON with key:
- summary

Candidate Name: {candidate_name}
Role: {role_name}
Score: {score}
Decision: {decision}
Gaps: {gaps}

Write a short recruiter-friendly summary in 2-3 lines.
No markdown. No explanation.
"""

    parsed = groq_chat_json(
        "You are a recruiter assistant. Return only JSON.",
        prompt,
    )
    return safe_string(parsed.get("summary", ""), 4000)


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------

def skill_matches(required_skill: str, candidate_skill: str) -> bool:
    req = canonicalize_skill(required_skill)
    cand = canonicalize_skill(candidate_skill)

    if not req or not cand:
        return False

    if req == cand:
        return True

    # substring/abbreviation friendly matching
    if req in cand or cand in req:
        return True

    return False


def calculate_score(
    candidate_exp: float,
    required_exp: float,
    candidate_skills: str,
    must_have_skills: str,
    nice_to_have_skills: str,
    candidate_education: str,
    education_required: str,
    candidate_certifications: str,
    certifications_required: str,
):
    score = 0
    gaps = []

    # Experience = 40
    if required_exp and float(required_exp) > 0:
        exp_score = min(float(candidate_exp) / float(required_exp), 1) * 40
        score += exp_score
        if float(candidate_exp) < float(required_exp):
            gaps.append(f"Need at least {required_exp} years experience")
    else:
        score += 40

    # Must-have = 30
    must_list = normalize_list(must_have_skills)
    cand_list = normalize_list(candidate_skills)

    matched_must = []
    missing_must = []
    for required in must_list:
        found = any(skill_matches(required, cand) for cand in cand_list)
        if found:
            matched_must.append(required)
        else:
            missing_must.append(required)

    if must_list:
        score += (len(matched_must) / len(must_list)) * 30
        if missing_must:
            gaps.append("Missing must-have skills: " + ", ".join(missing_must))
    else:
        score += 30

    # Nice-to-have = 10
    nice_list = normalize_list(nice_to_have_skills)
    matched_nice = []
    for required in nice_list:
        found = any(skill_matches(required, cand) for cand in cand_list)
        if found:
            matched_nice.append(required)

    if nice_list:
        score += (len(matched_nice) / len(nice_list)) * 10
    else:
        score += 10

    # Education = 10
    if education_required and normalize_text(education_required) in normalize_text(candidate_education):
        score += 10
    elif education_required:
        gaps.append(f"Education requirement not met: {education_required}")

    # Certifications = 10
    if certifications_required and normalize_text(certifications_required) in normalize_text(candidate_certifications):
        score += 10
    elif certifications_required:
        gaps.append(f"Missing certification: {certifications_required}")

    return round(score), "; ".join(gaps) if gaps else "None identified"


# -----------------------------------------------------------------------------
# Main endpoint
# -----------------------------------------------------------------------------
@app.route("/evaluate-candidate", methods=["POST"])
def evaluate_candidate():
    try:
        data = request.get_json(force=True)
        if isinstance(data, str):
            data = json.loads(data)

        # Raw candidate fields from form
        name = data.get("name", "")
        skills = data.get("skills", "")
        experience = float(data.get("experience", 0) or 0)
        education = data.get("education", "")
        certifications = data.get("certifications", "")

        # Job requirement fields
        role_name = data.get("role_name", "General Role")
        must_have_skills = data.get("must_have_skills", "")
        nice_to_have_skills = data.get("nice_to_have_skills", "")
        required_experience = float(data.get("required_experience", 0) or 0)
        education_required = data.get("education_required", "")
        certifications_required = data.get("certifications_required", "")

        # Resume
        resume_b64 = data.get("resume_base64", "")
        resume_text = ""

        parsed_resume = {
            "parsed_skills": "",
            "parsed_certifications": "",
            "parsed_experience": "",
            "parsed_summary": "",
            "parsed_years": 0,
            "parsed_education": "",
        }

        if resume_b64:
            resume_text = extract_text_from_pdf_base64(resume_b64)

            # STRICT RULE-BASED FIELDS
            parsed_resume["parsed_years"] = extract_years_from_text(resume_text)
            parsed_resume["parsed_education"] = extract_education_from_text(resume_text)

            # RULE-BASED EXTRACTION FOR SKILLS/CERTS/EXP IF POSSIBLE
            rule_skills = extract_skills_rule_based(resume_text)
            rule_certs = extract_certifications_rule_based(resume_text)
            rule_exp_summary = extract_experience_summary_rule_based(resume_text)

            # GROQ FALLBACK / ENRICHMENT
            groq_parsed = extract_resume_fields_with_groq(resume_text)

            parsed_resume["parsed_skills"] = rule_skills or groq_parsed.get("parsed_skills", "")
            parsed_resume["parsed_certifications"] = rule_certs or groq_parsed.get("parsed_certifications", "")
            parsed_resume["parsed_experience"] = rule_exp_summary or groq_parsed.get("parsed_experience", "")
            parsed_resume["parsed_summary"] = groq_parsed.get("parsed_summary", "")

        # Final values for scoring
        parsed_skills = safe_string(parsed_resume.get("parsed_skills", "") or skills, 2000)
        parsed_education = safe_string(parsed_resume.get("parsed_education", "") or education, 2000)
        parsed_certifications = safe_string(parsed_resume.get("parsed_certifications", "") or certifications, 2000)
        parsed_experience_text = safe_string(parsed_resume.get("parsed_experience", ""), 2000)
        parsed_years = float(parsed_resume.get("parsed_years", 0) or 0)

        final_years = parsed_years if parsed_years > 0 else experience

        score, gaps = calculate_score(
            candidate_exp=final_years,
            required_exp=required_experience,
            candidate_skills=parsed_skills,
            must_have_skills=must_have_skills,
            nice_to_have_skills=nice_to_have_skills,
            candidate_education=parsed_education,
            education_required=education_required,
            candidate_certifications=parsed_certifications,
            certifications_required=certifications_required,
        )

        if score >= 80:
            decision = "AUTO_SEND"
        elif score >= 60:
            decision = "REVIEW"
        else:
            decision = "REJECT"

        summary = build_ai_summary(name, role_name, score, decision, gaps)

        return jsonify({
            "score": score,
            "decision": decision,
            "confidence": 0.9 if decision != "REJECT" else 0.75,
            "summary": safe_string(summary, 4000),
            "gaps": safe_string(gaps, 4000),
            "parsed_skills": safe_string(parsed_skills, 2000),
            "parsed_education": safe_string(parsed_education, 2000),
            "parsed_certifications": safe_string(parsed_certifications, 2000),
            "parsed_experience": safe_string(parsed_experience_text, 2000),
            "parsed_years": final_years,
            "parsed_summary": safe_string(parsed_resume.get("parsed_summary", ""), 4000),
            "resume_text": safe_string(resume_text, 4000),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
