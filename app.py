import os
import json
import re
import base64
import io
import requests
from flask import Flask, request, jsonify
from pdfminer.high_level import extract_text

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"


def normalize_list(text):
    if not text:
        return []
    return [x.strip().lower() for x in re.split(r"[,\n;]+", text) if x.strip()]


def calculate_score(candidate_exp, required_exp,
                    candidate_skills, must_have_skills, nice_to_have_skills,
                    candidate_education, education_required,
                    candidate_certifications, certifications_required):
    score = 0
    gaps = []

    if required_exp and float(required_exp) > 0:
        exp_score = min(float(candidate_exp) / float(required_exp), 1) * 40
        score += exp_score
        if float(candidate_exp) < float(required_exp):
            gaps.append(f"Need at least {required_exp} years experience")
    else:
        score += 40

    must_list = normalize_list(must_have_skills)
    cand_list = normalize_list(candidate_skills)
    matched_must = [s for s in must_list if s in cand_list]

    if must_list:
        score += (len(matched_must) / len(must_list)) * 30
        missing = [s for s in must_list if s not in cand_list]
        if missing:
            gaps.append("Missing must-have skills: " + ", ".join(missing))
    else:
        score += 30

    nice_list = normalize_list(nice_to_have_skills)
    matched_nice = [s for s in nice_list if s in cand_list]
    if nice_list:
        score += (len(matched_nice) / len(nice_list)) * 10

    if education_required and education_required.lower() in (candidate_education or "").lower():
        score += 10
    elif education_required:
        gaps.append(f"Education requirement not met: {education_required}")

    if certifications_required and certifications_required.lower() in (candidate_certifications or "").lower():
        score += 10
    elif certifications_required:
        gaps.append(f"Missing certification: {certifications_required}")

    return round(score), "; ".join(gaps) if gaps else "None identified"


def extract_text_from_pdf_base64(resume_b64: str) -> str:
    pdf_bytes = base64.b64decode(resume_b64)
    with io.BytesIO(pdf_bytes) as pdf_stream:
        return extract_text(pdf_stream) or ""


def extract_years_from_text(text: str) -> int:
    matches = re.findall(r"(\d{1,2})\s*\+?\s*years", text.lower())
    if matches:
        return max(int(m) for m in matches)
    return 0


def extract_education_from_text(text: str) -> str:
    t = text.lower()
    if "bachelor of engineering" in t or "b.e." in t:
        return "B.E."
    if "btech" in t or "b.tech" in t:
        return "BTECH"
    if "mba" in t or "master of business administration" in t:
        return "MBA"
    if "m.tech" in t or "mtech" in t:
        return "M.TECH"
    if "bachelor" in t:
        return "BACHELOR"
    if "master" in t:
        return "MASTER"
    return ""


def extract_resume_fields_with_groq(resume_text: str) -> dict:
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
- DO NOT infer or assume.

Resume Text:
{resume_text}
"""

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are an AI resume parser. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=60,
    )

    if not response.ok:
        return {
            "parsed_skills": "",
            "parsed_certifications": "",
            "parsed_experience": "",
            "parsed_summary": "",
        }

    ai_text = response.json()["choices"][0]["message"]["content"].strip()
    ai_text = ai_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(ai_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", ai_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))

    return {
        "parsed_skills": "",
        "parsed_certifications": "",
        "parsed_experience": "",
        "parsed_summary": "",
    }


@app.route("/evaluate-candidate", methods=["POST"])
def evaluate_candidate():
    try:
        data = request.get_json(force=True)
        if isinstance(data, str):
            data = json.loads(data)

        name = data.get("name", "")
        skills = data.get("skills", "")
        experience = float(data.get("experience", 0) or 0)
        education = data.get("education", "")
        certifications = data.get("certifications", "")

        role_name = data.get("role_name", "General Role")
        must_have_skills = data.get("must_have_skills", "")
        nice_to_have_skills = data.get("nice_to_have_skills", "")
        required_experience = float(data.get("required_experience", 0) or 0)
        education_required = data.get("education_required", "")
        certifications_required = data.get("certifications_required", "")

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

            # strict rule‑based fields
            parsed_resume["parsed_years"] = extract_years_from_text(resume_text)
            parsed_resume["parsed_education"] = extract_education_from_text(resume_text)

            # Groq only for skills + certs + summary
            groq_parsed = extract_resume_fields_with_groq(resume_text)
            parsed_resume.update(groq_parsed)

        parsed_skills = parsed_resume.get("parsed_skills", "") or skills
        parsed_education = parsed_resume.get("parsed_education", "") or education
        parsed_certifications = parsed_resume.get("parsed_certifications", "") or certifications
        parsed_experience_text = parsed_resume.get("parsed_experience", "")
        parsed_years = float(parsed_resume.get("parsed_years", 0) or 0)

        final_years = parsed_years if parsed_years > 0 else experience

        score, gaps = calculate_score(
            candidate_exp=final_years,
            required_exp=required_experience,
            candidate_skills=parsed_skills or skills,
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

        summary = ""
        gaps_text = gaps

        return jsonify({
            "score": score,
            "decision": decision,
            "confidence": 0.9 if decision != "REJECT" else 0.75,
            "summary": summary,
            "gaps": gaps_text,
            "parsed_skills": parsed_skills,
            "parsed_education": parsed_education,
            "parsed_certifications": parsed_certifications,
            "parsed_experience": parsed_experience_text,
            "parsed_years": final_years,
            "parsed_summary": parsed_resume.get("parsed_summary", ""),
            "resume_text": resume_text[:4000] if resume_text else "",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
