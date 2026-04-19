import os
import json
import re
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/evaluate-candidate', methods=['POST'])
def evaluate_candidate():
    data = request.get_json(force=True)
    if isinstance(data, str):
        data = json.loads(data)

    name = data.get('name', '')
    skills = data.get('skills', '')
    experience = data.get('experience', '')
    education = data.get('education', '')
    certifications = data.get('certifications', '')
    job_requirement = data.get('job_requirement', '')

    prompt = f"""
Return ONLY raw JSON. No markdown. No explanation. No code fences.

Schema:
{{
  "score": 0,
  "decision": "AUTO_SEND",
  "confidence": 0.0,
  "summary": "",
  "gaps": ""
}}

Candidate:
Name: {name}
Skills: {skills}
Experience: {experience}
Education: {education}
Certifications: {certifications}

Job Requirement:
{job_requirement}
"""


    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        return jsonify({'error': 'GROQ_API_KEY is not set'}), 500

    groq_response = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {groq_api_key}',
            'Content-Type': 'application/json'
        },
        json={
            'model': 'llama-3.1-8b-instant',
            'messages': [
                {'role': 'system', 'content': 'You are an AI hiring screening assistant. Return ONLY valid JSON.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.2
        },
        timeout=60
    )

    if not groq_response.ok:
        return jsonify({
            'error': 'Groq API request failed',
            'status_code': groq_response.status_code,
            'details': groq_response.text
        }), 500

    ai_text = groq_response.json()['choices'][0]['message']['content'].strip()

    # remove markdown fences
    ai_text = ai_text.replace('```json', '').replace('```', '').strip()

    # try direct parse first
    try:
        ai_json = json.loads(ai_text)
    except json.JSONDecodeError:
        # fallback: extract first JSON object from the text
        match = re.search(r'\{.*\}', ai_text, re.DOTALL)
        if not match:
            return jsonify({
                'error': 'AI did not return valid JSON',
                'raw_response': ai_text
            }), 500
        ai_json = json.loads(match.group(0))

    return jsonify(ai_json)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
