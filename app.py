import os
import json
import re
import requests
from flask import Flask, request, jsonify

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

    # Experience: 40 points
    if required_exp and required_exp > 0:
        exp_score = min(float(candidate_exp) / float(required_exp), 1) * 40
        score += exp_score
        if float(candidate_exp) < float(required_exp):
            gaps.append(f"Need at least {required_exp} years experience")
    else:
        score += 40

    # Must-have skills: 30 points
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

    # Nice-to-have skills: 10 points
    nice_list = normalize_list(nice_to_have_skills)
    matched_nice = [s for s in nice_list if s in cand_list]
    if nice_list:
        score += (len(matched_nice) / len(nice_list)) * 10

    # Education: 10 points
    if education_required and education_required.lower() in (candidate_education or "").lower():
        score += 10
    elif education_required:
        gaps.append(f"Education requirement not met: {education_required}")

    # Certifications: 10 points
    if certifications_required and certifications_required.lower() in (candidate_certifications or "").lower():
        score += 10
    elif certifications_required:
        gaps.append(f"Missing certification: {certifications_required}")

    return round(score), "; ".join(gaps) if gaps else "None identified"


def build_ai_summary(candidate_name, role_name, score, decision, gaps):
    if not GROQ_API_KEY:
        return f"Candidate {candidate_name} evaluated for {role_name}. Score: {score}. Decision: {decision}.", gaps

    prompt = f"""
Return ONLY valid JSON with keys:
- summary
- gaps

Candidate Name: {candidate_name}
Role: {role_name}
Score: {score}
Decision: {decision}
Gaps: {gaps}

Write a short recruiter-friendly summary and a short gaps sentence.
No markdown. No explanation.
"""

    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a recruiter assistant. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            },
            timeout=60
        )

        if not response.ok:
            return f"Candidate {candidate_name} evaluated for {role_name}. Score: {score}. Decision: {decision}.", gaps

        ai_text = response.json()["choices"][0]["message"]["content"].strip()
        ai_text = ai_text.replace("```json", "").replace("```", "").strip()

        try:
            ai_json = json.loads(ai_text)
            return ai_json.get("summary", ""), ai_json.get("gaps", gaps)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", ai_text, re.DOTALL)
            if match:
                ai_json = json.loads(match.group(0))
                return ai_json.get("summary", ""), ai_json.get("gaps", gaps)

    except Exception:
        pass

    return f"Candidate {candidate_name} evaluated for {role_name}. Score: {score}. Decision: {decision}.", gaps


@app.route("/evaluate-candidate", methods=["POST"])
def evaluate_candidate():
    try:
        data = request.get_json(force=True)
        if isinstance(data, str):
            data = json.loads(data)

        # Candidate fields
        name = data.get("name", "")
        skills = data.get("skills", "")
        experience = data.get("experience", 0)
        education = data.get("education", "")
        certifications = data.get("certifications", "")
        job_requirement = data.get("job_requirement", "")

        # Job requirement fields (optional if passed as structured text)
        role_name = data.get("role_name", "General Role")
        must_have_skills = data.get("must_have_skills", "")
        nice_to_have_skills = data.get("nice_to_have_skills", "")
        required_experience = data.get("required_experience", 0)
        education_required = data.get("education_required", "")
        certifications_required = data.get("certifications_required", "")

        # Calculate deterministic score
        score, gaps = calculate_score(
            candidate_exp=experience,
            required_exp=required_experience,
            candidate_skills=skills,
            must_have_skills=must_have_skills,
            nice_to_have_skills=nice_to_have_skills,
            candidate_education=education,
            education_required=education_required,
            candidate_certifications=certifications,
            certifications_required=certifications_required
        )

        # Decision
        if score >= 80:
            decision = "AUTO_SEND"
        elif score >= 60:
            decision = "REVIEW"
        else:
            decision = "REJECT"

        # Optional AI summary/gaps wording
        summary, ai_gaps = build_ai_summary(name, role_name, score, decision, gaps)
        final_gaps = ai_gaps or gaps

        return jsonify({
            "score": score,
            "decision": decision,
            "confidence": 0.9 if decision != "REJECT" else 0.75,
            "summary": summary,
            "gaps": final_gaps
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
