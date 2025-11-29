import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # loads values from .env in your root folder


def check_openai():
    print("\n=== Checking OpenAI ===")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        print("❌ OPENAI_API_KEY is missing in .env")
        return

    try:
        client = OpenAI(api_key=api_key)
        # Tiny cheap call just to see if the key is valid
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        print(f"✅ OpenAI connection OK (model={model})")
    except Exception as e:
        print("❌ OpenAI connection FAILED")
        print("   Error:", repr(e))


def check_azure_ocr():
    print("\n=== Checking Azure OCR ===")
    endpoint = os.getenv("AZURE_OCR_ENDPOINT", "").rstrip("/")
    key = os.getenv("AZURE_OCR_KEY")

    if not endpoint:
        print("❌ AZURE_OCR_ENDPOINT is missing in .env")
        return
    if not key:
        print("❌ AZURE_OCR_KEY is missing in .env")
        return

    url = endpoint + "/vision/v3.2/read/analyze"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/json",
    }
    # Microsoft sample image (public) – just to test the call
    body = {
        "url": "https://aka.ms/azsdk/formrecognizer/sample.jpg"
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=15)
        if resp.status_code == 202:
            print("✅ Azure OCR connection OK (got 202 Accepted)")
        elif resp.status_code in (401, 403):
            print("❌ Azure OCR auth FAILED (check key/endpoint)")
            print("   Status:", resp.status_code, "Body:", resp.text[:300])
        else:
            print("❌ Azure OCR call returned unexpected status")
            print("   Status:", resp.status_code, "Body:", resp.text[:300])
    except Exception as e:
        print("❌ Azure OCR connection FAILED")
        print("   Error:", repr(e))


def check_supabase():
    print("\n=== Checking Supabase ===")
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_KEY")

    if not url:
        print("❌ SUPABASE_URL is missing in .env")
        return
    if not key:
        print("❌ SUPABASE_KEY is missing in .env")
        return

    # We just ping the REST endpoint root; 404 is okay, 401/403 means auth issue.
    test_url = url + "/rest/v1/"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }

    try:
        resp = requests.get(test_url, headers=headers, timeout=15)
        if resp.status_code in (200, 404):
            print(f"✅ Supabase key accepted (status={resp.status_code})")
        elif resp.status_code in (401, 403):
            print("❌ Supabase auth FAILED (check key / service_role vs anon)")
            print("   Status:", resp.status_code, "Body:", resp.text[:300])
        else:
            print("❌ Supabase call returned unexpected status")
            print("   Status:", resp.status_code, "Body:", resp.text[:300])
    except Exception as e:
        print("❌ Supabase connection FAILED")
        print("   Error:", repr(e))


if __name__ == "__main__":
    check_openai()
    check_azure_ocr()
    check_supabase()
    print("\n=== Done ===")
