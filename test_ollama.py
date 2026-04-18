from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN")
)

prompt = """You are an industrial troubleshooting assistant.
Answer ONLY using the retrieved evidence below.
Do not invent repair steps that are not supported by the evidence.
If evidence is weak, say so clearly.

Return valid JSON only with this exact schema:
{
  "likely_causes": [{"cause": "", "why": ""}],
  "recommended_checks": [""],
  "manual_references": [{"source": "", "page": "", "reason": ""}],
  "similar_incidents": [{"machine_id": "", "fault_code": "", "summary": ""}],
  "escalation_needed": false,
  "confidence": "low|medium|high"
}

Inputs:
Machine ID: M01
Fault Code: E102
Technician Query: The machine has vibration and abnormal bearing noise. What should I inspect first?

Retrieved OEM Manual Evidence:
None

Retrieved Historical Incident Evidence:
[LOG 1] Row: 0 | Machine ID: M01 | Fault Code: E102
Machine ID: M01
Fault Code: E102
Machine Type: pump
Symptoms: abnormal bearing noise and vibration during operation
Root Cause: bearing wear
Action Taken: inspect bearing housing, replace bearing, relubricate
Outcome: resolved

[LOG 2] Row: 3 | Machine ID: M01 | Fault Code: E102
Machine ID: M01
Fault Code: E102
Machine Type: pump
Symptoms: recurring vibration after restart
Root Cause: shaft misalignment
Action Taken: realign shaft and recheck coupling
Outcome: resolved
"""

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {
            "role": "system",
            "content": "Return only valid JSON. Do not add markdown or extra text. Do not output reasoning. Do not output <think>."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.1,
    max_tokens=500,
)

print(response.choices[0].message.content)