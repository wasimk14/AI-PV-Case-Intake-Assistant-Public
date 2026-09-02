import pymupdf  # PyMuPDF
import json
import re
import os
import requests
from pathlib import Path
BASE_DIR = Path.cwd()
pdf_path = BASE_DIR/"input"
pdf_file = BASE_DIR/"input"/"Case Snapshot.pdf"
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(BASE_DIR / ".env")
client = OpenAI(timeout=180.0)

# =========================================================
# Extract complete text
# =========================================================

def extract_pdf_text(pdf_files):
    doc = pymupdf.open(pdf_files)

    pages = []

    for page in doc:
        text = page.get_text("text")
        pages.append(text)

    doc.close()

    return "\n".join(pages)

# ===================================================================
# Preserve almost everything and just normalize excessive whitespace.
# ===================================================================

def clean_extracted_text(text):

    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # Remove trailing/leading whitespace from each line
    lines = [line.strip() for line in text.splitlines()]

    # Remove empty lines
    lines = [line for line in lines if line]

    return "\n".join(lines)

# ===================================================================
# Temporary test input for final pipeline validation
# ===================================================================

test_pdf = pdf_file

extracted_text = extract_pdf_text(test_pdf)
cleaned_text = clean_extracted_text(extracted_text)

print(len(extracted_text))
print(len(cleaned_text))

# ===================================================================
# EXTRACTION PROMPT
# ===================================================================

pv_extraction_prompt = """
You are assisting with Pharmacovigilance case intake.

Extract adverse events and relevant medical history from the source document.

IMPORTANT RULES:
1. Extract information only if supported by the source document.
2. Do not invent or infer missing clinical information.
3. Search the entire document, including structured fields and free-text sections
   such as emails, requests, event descriptions, narratives, and notes.
4. Do not rely only on structured fields such as "Reported Term".
5. Preserve the reporter/source wording as the verbatim term whenever possible.
6. If a date or outcome cannot be reliably established from the source, return null.
7. A condition may be both medical history and an adverse event when the source
   clearly indicates that it existed previously and subsequently worsened or changed.

ADVERSE EVENT RULES:
- Identify adverse events from both structured fields and contextual free text.
- Preserve the reported wording rather than converting the event into a clinical
  diagnosis that is not explicitly stated in the source.
- Capture lack of therapeutic effect when the source clearly states or describes
  that the medication was not producing the desired or expected result, was not
  working, was ineffective, had stopped working, or was providing only partial
  or insufficient benefit.
- Capture an adverse event onset date only when the source clearly states or
  contextually links a date or time period to the onset of that specific event.
- Distinguish adverse event onset dates from communication dates, report dates,
  document dates, follow-up dates, timestamps, and dates of healthcare interactions.
- If the event onset cannot be reliably established from the source, return null.

MEDICAL HISTORY RULES:
- Identify conditions that the source indicates existed before the current adverse
  event or treatment/exposure.
- Consider contextual descriptions of prior, pre-existing, chronic, recurrent,
  or previously experienced conditions.
- Do not require explicit phrases such as "medical history" or "history of".
- Preserve the source wording whenever possible.
- Do not convert informal descriptions into diagnoses that are not explicitly
  supported by the source.
- Do not infer medical history solely from medications, indications, laboratory
  results, or other indirect evidence unless the condition itself is supported
  by the source.
  
MEDICATION RULES:

- Identify medications from both structured medication fields and contextual
  free text throughout the source document.

- Preserve the product name as reported in the source under verbatim_product_name.

- verbatim_product_name should contain the product/drug name itself and should
  not include dosing instructions, titration instructions, strength, frequency,
  route, or other administration information when these can be separated.

- Extract generic_name only when the generic or active ingredient is explicitly
  provided in the source document.

- Do not use external pharmaceutical knowledge to infer a generic name,
  active ingredient, strength, indication, route, or other drug information.


DRUG ROLE:

- Classify each medication using drug_role as one of:
    "Suspect"
    "Concomitant"
    "Past"

- "Suspect" = the source identifies or contextually describes the medication
  as suspected in relation to the reported adverse event(s).

- "Concomitant" = the medication is being taken during the relevant case period
  but is not identified or described as suspected.

- "Past" = the source clearly indicates that the medication was taken previously
  and is no longer being taken during the relevant case period.

- Determine drug_role from the complete source context and not merely from the
  section in which the medication name appears.

- Do not classify a medication as Suspect merely because it appears close to an
  adverse event in the extracted text.

- If the medication role cannot be reliably determined, return null for drug_role.  

Return only valid JSON using this structure:

{
  "adverse_events": [
    {
      "verbatim_term": null,
      "event_type": "clinical_event | special_situation",
      "meddra_llt": [],
      "meddra_pt": [],
      "onset_date": null,
      "outcome": null,
      "reporter_causality": "Not reported"
    }
  ],
  "medical_history": [
    {
      "verbatim_term": null,
      "start_date": null,
      "end_date": null
    }
  ],
  "medications": [
    {
      "verbatim_product_name": null,
      "generic_name": null,
      "drug_role": null,
      "indication": null
    }
  ],
  "labs_diagnostics": [
    {
      "verbatim_test_name": null,
      "result": null,
    }
  ]
}

SOURCE DOCUMENT:
"""

case_prompt = pv_extraction_prompt + "\n" + cleaned_text

response = client.responses.create(
    model="gpt-4.1-mini",
    input=case_prompt
)

# MeddDRA

# ===================================================================
# MEDDRA PROMPT
# ===================================================================

BIOPORTAL_API_KEY = os.getenv("BIOPORTAL_API_KEY")

MEDDRA_SELECTION_PROMPT = """
You are assisting with MedDRA coding in a Pharmacovigilance workflow.

You will receive:
1. A reported adverse-event search phrase.
2. Candidate MedDRA LLTs retrieved from BioPortal.

Select the SINGLE best LLT from the supplied candidates.

CRITICAL SELECTION PRIORITY:

Apply these rules in this exact order:

1. Reject any candidate that introduces an unreported clinical characteristic
   such as morphology, diagnosis, severity, symptom, cause, or additional
   anatomical site.

2. Among the remaining candidates, prefer the closest anatomical match.

3. Among equally faithful candidates, prefer the more specific term.

Never sacrifice Rule 1 to obtain a better anatomical or more specific match.

RULES:

1. Select ONLY from the supplied BioPortal candidates.

2. The selected LLT must preserve the meaning of the reported term as
   closely as possible.

3. NEVER select a candidate that introduces an unreported medical concept,
   anatomical site, morphology, severity, diagnosis, cause, or qualifier.

4. Anatomical specificity must be source-faithful.
   A candidate containing an additional or different anatomical site is
   NOT a closer match merely because it contains part of the reported site.

5. If no candidate preserves the reported specificity without introducing
   unsupported information, choose the closest broader term.

6. A broader accurate term is ALWAYS preferable to a more specific but
   partially incorrect term.

Return ONLY valid JSON:

{
    "term": "selected candidate term",
    "code": "selected candidate code"
}
"""

def search_meddra(term, limit=10):
    # limit: limit to return many search matches returned
    url = "https://data.bioontology.org/search"

    params = {
        "q": term,
        "ontologies": "MEDDRA",
        "apikey": BIOPORTAL_API_KEY
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    results = []

    for item in data.get("collection", [])[:limit]:
        results.append({
            "term": item.get("prefLabel"),
            "code": item.get("@id", "").split("/")[-1],
            "concept_url": item.get("@id")
        })

    return results

# ===================================================================
# Tutorial Test — MedDRA candidate search
# ===================================================================

results = search_meddra("fever")

for result in results:
    print(result)

# ===================================================================
# # Helper function get_meddra_pt
# ===================================================================

from urllib.parse import quote
# quote() is for URL encoding.

def get_meddra_pt(concept_url):
    encoded_url = quote(concept_url, safe="")

    url = (
            "https://data.bioontology.org/ontologies/MEDDRA/classes/"
            + encoded_url
    )

    params = {
        "apikey": BIOPORTAL_API_KEY,
        "include": "all"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    concept = response.json()
    properties = concept.get("properties", {})  # value is expected to be a dictionary.

    classified_as = properties.get(
        "http://purl.bioontology.org/ontology/MEDDRA/classified_as"
    )

    # Selected LLT
    #     ↓
    # classified_as
    #     ↓
    # Corresponding PT

    # CASE 1:
    # LLT has a separate PT
    if classified_as:
        pt_url = classified_as[0]
        # classified_as[0] expecting the one PT it maps to. BioPortal represents the property as a list, hence [0] extracts its single expected value.

        response = requests.get(
            "https://data.bioontology.org/ontologies/MEDDRA/classes/"
            + quote(pt_url, safe=""),
            params=params
        )

        response.raise_for_status()

        pt = response.json()

        return {
            "pt": pt.get("prefLabel"),
            "pt_code": pt.get("@id", "").split("/")[-1]
        }

    # CASE 2:
    # No separate classified_as relationship.
    # Treat selected concept as the corresponding PT.
    return {
        "pt": concept.get("prefLabel"),
        "pt_code": concept.get("@id", "").split("/")[-1]
    }

# ===================================================================
# Tutorial Test — MedDRA LLT → PT lookup
# ===================================================================

candidate = results[3]

pt = get_meddra_pt(candidate["concept_url"])

print("LLT:", candidate["term"])
print("LLT Code:", candidate["code"])
print("PT:", pt["pt"])
print("PT Code:", pt["pt_code"])
print(candidate["concept_url"])

def select_meddra_llt(search_term, candidates):

    candidate_list = [
        {
            "term": candidate["term"],
            "code": candidate["code"]
        }
        for candidate in candidates
    ]

    user_input = {
        "reported_term": search_term,
        "candidates": candidate_list
    }

    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=MEDDRA_SELECTION_PROMPT,
        input=json.dumps(user_input)
    )

    return json.loads(response.output_text)

# ===================================================================
# Tutorial Test — AI-assisted MedDRA LLT selection
# ===================================================================

candidates = search_meddra("rash on inner arm")

selected = select_meddra_llt(
    "rash on inner arm",
    candidates
)

print(selected)

def code_meddra_term(search_term):

    # 1. Search BioPortal
    candidates = search_meddra(search_term)

    if not candidates:
        return None

    # 2. Select best LLT
    selected = select_meddra_llt(
        search_term,
        candidates
    )

    # 3. Find selected candidate
    selected_candidate = next(
        candidate for candidate in candidates
        if candidate["code"] == selected["code"]
    )

    # 4. Get PT
    pt = get_meddra_pt(
        selected_candidate["concept_url"]
    )

    # 5. Return final coding
    return {
        "llt": selected["term"],
        "llt_code": selected["code"],
        "pt": pt["pt"],
        "pt_code": pt["pt_code"]
    }

# ===================================================================
# Tutorial Test — End-to-end MedDRA coding
# ===================================================================

test_terms = [
    "hives",
    "fever"
]

for term in test_terms:
    result = code_meddra_term(term)
    print("\nReported:", term)
    print(result)

# ===================================================================
# NARRATIVE PROMPT
# ===================================================================

NARRATIVE_PROMPT = """
You are generating a Pharmacovigilance case narrative from a structured,
review-ready case record.

Use ONLY the information contained in the supplied structured JSON.
Do not use outside medical knowledge.
Do not infer or invent information that is not present.

GENERAL RULES:
1. Write the narrative in professional Pharmacovigilance case-narrative style.
2. Present information in a clear and logical chronological sequence.
3. Preserve medically relevant details from the structured case.
4. Incorporate relevant initial and follow-up information into one coherent narrative.
5. Do not mention MedDRA LLT/PT codes, IME status, USPI listedness, or internal
   processing fields unless specifically required by the narrative template.
6. Do not repeat the same information unnecessarily.
7. Do not invent dates, patient characteristics, treatment details, outcomes,
   causality, laboratory values, normal ranges, batch numbers, expiration dates,
   or follow-up information.
8. Where the template provides alternative sentences, select only the alternative
   supported by the structured case.
9. Omit optional statements when the required information is not available,
   unless the template specifically provides wording for missing information.
10. Maintain the reporter's meaning. Do not strengthen or reinterpret causality,
    seriousness, outcome, or medical conclusions.
11. If patient age or gender is not reported, or cannot be ascertained from written source information, capture
    this in a grammatically correct lead sentence.  

NARRATIVE TEMPLATE AND RULES:

OPENING:
Case reference number (202X-XXXXX-XX) is a (case type) case initially received
on DD-MMM-YYYY from a (reporter type) and concerned a xx-year-old (male/female) patient.

OR

Case reference number (202X-XXXXX-XX) is a (case type) case initially received
on DD-MMM-YYYY from a (reporter type) and concerned a patient of unspecified age and gender.

OR

Case reference number (202X-XXXXX-XX) is a (case type) case initially received
on DD-MMM-YYYY from a (reporter type) and concerned a (male/female) patient of unspecified age.

OR

Case reference number (202X-XXXXX-XX) is a (case type) case initially received
on DD-MMM-YYYY from a (reporter type) and concerned a xx-year-old patient of unspecified gender.

MEDICAL HISTORY:
The patient's medical history included (list reported concurrent conditions).
Omit this sentence if no medical history is reported.

CONCOMITANT MEDICATIONS:
The patient's concomitant medications included brand name (generic name).

If a generic name was reported, write it in lower case.
If a registered/brand name was reported, preserve capitalization and place the
generic name in brackets in lower case when available.
Omit this sentence if no concomitant medications are reported.

SUSPECT DRUG:
On DD-MMM-YYYY, the patient initiated treatment with (Suspect Drug Name)
(generic name) (dose, frequency) for (indication).

If batch/lot number and expiration date are available, include:
The batch number used was XXX and expiration date was DD-MMM-YYYY.

Do not mention batch number or expiration date if they were not reported.

ADVERSE EVENTS:
Describe the reported adverse events in chronological order and include as much
relevant reported clinical detail as available.

When the interval from the last suspect-drug dose to event onset can be
determined:
On DD-MMM-YYYY, X days after the last (Suspect Drug Name) dose, the patient
experienced/developed (event details).

When the interval cannot be determined:
On DD-MMM-YYYY, an unspecified time after the last (Suspect Drug Name) dose,
the patient experienced/developed (event details).

Do not calculate or state a time interval unless the required dates are available.

LABORATORY / DIAGNOSTIC INFORMATION:
Include all reported laboratory results in the narrative.
Describe abnormal results with the available relevant details, including result,
unit and normal/reference range when provided.

When applicable:
On DD-MMM-YYYY, (laboratory test) revealed/showed/was measured at (result),
with the reported normal/reference range.

If laboratory tests are reported but normal ranges are unavailable, state:
The normal ranges for the laboratory tests performed were not reported.

ACTION TAKEN:
Use the appropriate statement according to the structured data:

On DD-MMM-YYYY, treatment with (Suspect Drug Name) was
(discontinued/interrupted/increased/decreased) in response to the event.

OR

Treatment with (Suspect Drug Name) was maintained.

OR

It was not reported whether the patient continued treatment with
(Suspect Drug Name).

OUTCOME:
Use the appropriate statement supported by the structured data:

On DD-MMM-YYYY, the patient recovered from the event.

OR

At the time of this report, the patient had not recovered from/was recovering
from the event.

OR

At the time of this report, the outcome of the event(s) was not provided.

CAUSALITY:
If reporter causality is available:
The reporter assessed the event as (causality assessment) related to
(Suspect Drug Name).

Include reported information supporting the assessment and other relevant
etiological factors only when they are present in the structured case.

If reporter causality is unavailable:
The reporter did not provide a causality assessment.

FOLLOW-UP CONSENT:
Include one of the following ONLY when explicitly reported:

The patient did not give consent to be contacted.

OR

The patient did not give consent for their healthcare professional to be contacted.

OUTPUT:
Return only the completed PV case narrative as continuous professional prose.
Do not return JSON, headings, bullet points, explanations, or commentary.

STRUCTURED CASE:
"""

# ===================================================================
# Tutorial Test — PV narrative generation
# ===================================================================

narrative_response = client.responses.create(
    model="gpt-4.1-mini",
    input=NARRATIVE_PROMPT
)

narrative = narrative_response.output_text

print(narrative)

# ===================================================================
# Final Pipeline — Structured Case + MedDRA Enrichment
# ===================================================================

# Convert OpenAI extraction output to Python dictionary
final_result = json.loads(response.output_text)

# Add MedDRA coding + IME seriousness + USPI listedness
for event in final_result["adverse_events"]:

    verbatim = event["verbatim_term"]

    meddra = code_meddra_term(verbatim)

    if meddra:
        event["meddra_llt"] = [{
            "term": meddra["llt"],
            "code": meddra["llt_code"]
        }]

        event["meddra_pt"] = [{
            "term": meddra["pt"],
            "code": meddra["pt_code"]
        }]


# =========================================================
# Generate PV narrative from FINAL enriched case
# =========================================================
narrative_response = client.responses.create(
    model="gpt-4.1-mini",
    input=NARRATIVE_PROMPT + json.dumps(final_result, indent=2)
)

narrative = narrative_response.output_text


# =========================================================
# FINAL OUTPUT
# =========================================================

# Display final enriched case
print("FINAL STRUCTURED CASE")
print(json.dumps(final_result, indent=2))

print("\n" + "=" * 80 + "\n")

# Display generated narrative
print("GENERATED PV NARRATIVE")
print(narrative)

print("\n" + "=" * 80 + "\n")

# =========================================================
# REPORTABILITY GUIDANCE
# =========================================================

print("REPORTABILITY ASSESSMENT GUIDANCE")

print(
    "\nThe following tables are provided as reference guidance only. "
    "The Case Processor/Safety Professional should determine final reportability "
    "according to applicable reporting requirements."
)

# =========================================================
# SUBMISSION TIMELINE GUIDANCE
# =========================================================

print("\nSUBMISSION TIMELINE GUIDANCE")

print("Serious Case Submission timeline to USA FDA: 15 calendar days")
print("Serious Case Submission timeline to EMA: 15 calendar days")
print("Non-Serious Case Submission timeline to EMA: 90 calendar days")
print("For other regions, please check applicable reporting rules.")

print(
    "\nNote: Reporting timelines are provided as guidance only. "
    "Final reporting requirements should be verified against current "
    "regulations and applicable company procedures."
)