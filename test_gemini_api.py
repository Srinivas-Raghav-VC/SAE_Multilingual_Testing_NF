"""
Quick smoke test for the Gemini (Google Generative AI) API.

Usage (from your venv):
    python test_gemini_api.py

Requirements:
    - Environment variable GOOGLE_API_KEY must be set.
    - `google-generativeai` must be installed in the venv.
"""

import json
import os
import re

import google.generativeai as genai


def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set in environment. "
            "Export it before running this script."
        )

    genai.configure(api_key=api_key)
    # Use same model as evaluation pipeline.
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = 'Reply ONLY with JSON: {"ok": true, "message": "Gemini API test passed"}'
    print("Sending test prompt to Gemini...")
    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()

    print("\nRaw response:")
    print(text)

    # Strip markdown code fences like ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"\nGemini responded, but output was not valid JSON: {e}\n{text}"
        )

    print("\nParsed JSON:")
    print(data)


if __name__ == "__main__":
    main()
