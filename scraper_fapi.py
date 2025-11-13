from flask import Flask, jsonify
from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
import time

app = Flask(__name__)

# --- MongoDB Connection ---
client = MongoClient("mongodb://localhost:27017/")
db = client["NHS_DBS"]
conditions_col = db["conditions"]

# --- Constants ---
BASE_URL = "https://www.nhsinform.scot"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

# --- Keyword Map ---
keyword_map = {
    "symptoms": [
        "symptom", "sign", "stages", "types", "pain", "swelling", "stiffness", "bruising",
        "discomfort", "ache", "common signs", "talking about", "what happened",
        "symptoms of", "in leukaemia", "acute leukaemia", "looking paler than usual",
        "feeling tired", "anaemia", "bruises", "may bruise more easily"
    ],
    "causes": [
        "cause", "risk factor", "what causes", "fracture", "injury", "smoking",
        "developed", "reaction", "drink alcohol", "exposure to", "the cause or causes of"
    ],
    "diagnosis": [
        "diagnos", "test", "screen", "procedure", "scan", "x-ray", "assessment", "examination",
        "testing", "tests", "challenge"
    ],
    "warnings": [
        "emergency", "warning", "seek medical advice", "complications", "when to get medical help",
        "fall", "risk", "not improved", "driving", "hospital", "avoid", "dial 999",
        "get medical advice", "is spread"
    ],
    "recommendations": [
        "treat", "recommend", "manage", "self-care", "prevention", "work", "recovery", "exercise",
        "rehabilitation", "massage", "raise", "balance", "diet", "active", "stop smoking",
        "return", "healthcare professional", "what to do", "who is affected", "injections",
        "support", "is treated", "lifestyle", "how to do tasks at home", "vaccination",
        "find your local services", "care homes"
    ]
}

# --- Helper Functions ---
def get_all_diseases():
    url = BASE_URL + "/illnesses-and-conditions/a-to-z"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    diseases = []
    for link in soup.select("div.az_list_indivisual a"):
        disease_name = link.get_text(strip=True)
        disease_url = link.get("href")
        if not disease_url.startswith(BASE_URL):
            disease_url = BASE_URL + disease_url
        diseases.append({"name": disease_name, "url": disease_url})
    return diseases


def extract_sections(soup):
    """Extract all h2 sections and their content."""
    sections = {}
    for h2 in soup.find_all("h2"):
        title = h2.get_text(strip=True)
        content = []
        sibling = h2.find_next_sibling()
        while sibling and sibling.name not in ["h2", "h1"]:
            if sibling.name == "ul":
                content.extend([li.get_text(strip=True) for li in sibling.find_all("li")])
            elif sibling.name == "p":
                content.append(sibling.get_text(strip=True))
            sibling = sibling.find_next_sibling()
        sections[title] = content
    return sections


def map_sections(sections: dict, condition_name: str):
    """Process and structure scraped NHS data."""
    mapped = {k: set() for k in keyword_map.keys()}  # sets to prevent duplicates


    for title, content in sections.items():
        title_lower = title.lower()
        content_text = " ".join(content)

        for category, keywords in keyword_map.items():
            if any(kw in title_lower for kw in keywords):
                mapped[category].add(content_text)
                break  

    for title, content in sections.items():
        title_lower = title.lower()
        if not any(kw in title_lower for kws in keyword_map.values() for kw in kws):
            content_text = " ".join(content)
            sentences = content_text.split(".")
            for sent in sentences:
                sent_lower = sent.lower().strip()
                if not sent_lower:
                    continue
                for category, keywords in keyword_map.items():
                    if any(kw in sent_lower for kw in keywords):
                        mapped[category].add(sent.strip())
                        break

    
    for cat in mapped:
        mapped[cat] = " ".join(sorted(mapped[cat]))

    return mapped


def get_disease_details(disease):
    """Scrape and process disease page."""
    tries = 3
    for attempt in range(tries):
        try:
            response = requests.get(disease["url"], headers=HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            sections = extract_sections(soup)
            structured = map_sections(sections, disease["name"])

            details = {
                "condition": disease["name"],
                "symptoms": structured["symptoms"],
                "causes": structured["causes"],
                "diagnosis": structured["diagnosis"],
                "recommendations": structured["recommendations"],
                "warnings": structured["warnings"]
            }
            return details
        except Exception as e:
            print(f" Attempt {attempt + 1} failed for {disease['name']}: {e}")
            time.sleep(3)
    return None


# --- Flask Routes ---
@app.route("/scrape", methods=["GET"])
def scrape_and_store():
    diseases = get_all_diseases()
    inserted, skipped, failed = 0, 0, 0

    for disease in diseases:
        try:
            if conditions_col.find_one({"condition": disease["name"]}):
                print(f" Already exists: {disease['name']}")
                skipped += 1
                continue

            details = get_disease_details(disease)
            if details:
                conditions_col.insert_one(details)
                print(f" Inserted: {details['condition']}")
                inserted += 1
            else:
                print(f" Skipped (failed): {disease['name']}")
                failed += 1

            time.sleep(1.5)
        except Exception as e:
            print(f" Error with {disease['name']}: {e}")
            failed += 1

    return jsonify({
        "status": "done",
        "inserted": inserted,
        "skipped": skipped,
        "failed": failed,
        "total_expected": len(diseases)
    })


@app.route("/count", methods=["GET"])
def count_documents():
    count = conditions_col.count_documents({})
    return jsonify({"total_conditions": count})


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to NHS Scraper API ",
        "endpoints": {
            "/scrape": "Scrape NHS and store structured data in MongoDB",
            "/count": "Show number of stored conditions"
        }
    })


if __name__ == "__main__":
    app.run(debug=True)

