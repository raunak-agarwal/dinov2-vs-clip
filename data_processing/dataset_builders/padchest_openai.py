"""
Use OpenAI API to translate PadChest Spanish x-ray reports to English.

Usage:
python padchest_openai.py --input_file ../data/PadChest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv \
    --output_file ../data/PadChest/PadChest-Translated.csv --cache_file ../data/PadChest/translations.csv
"""
import os
import json

import pandas as pd
from openai import AzureOpenAI
import argparse
from tqdm import tqdm
import hashlib
import time

client = AzureOpenAI(
  api_key = os.getenv("OPENAI_API_KEY"),
  api_version = "2024-02-01",
  azure_endpoint = "https://insert-endpoint.openai.azure.com/" #Insert your own endpoint
)

SYSTEM_PROMPT = """You are a language model that translates x-ray reports from Spanish to English. \
You will be provided a report and some localized label data that was extracted from the report.\
The labels were extracted using an old NLP model and are not 100% accurate\n
You respond to the user with a json of the following format: {"output": "translated text"}\n
Do not add any additional information to the output. The output needs to be a direct translation of the input text."""
BASE_USER_PROMPT = "Please translate the following x-ray report into English and return a json: "

padchest_columns = ['ImageID', 'StudyID', 'PatientID', 'PatientBirth','PatientSex_DICOM', 'Projection', 'Report', 'Labels', 'LabelsLocalizationsBySentence']

def parse_args():
    parser = argparse.ArgumentParser(description="Translate PadChest reports using OpenAI API")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file containing PadChest data")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output CSV file to store translated reports")
    parser.add_argument("--cache_file", type=str, default="padchest-cache.json", help="Path to the cache JSON file to store processed reports")
    return parser.parse_args()


def fast_deduplicate(lst):
    # Deduplicate a list or a list of lists
    def to_hashable(item):
        if isinstance(item, list):
            return tuple(to_hashable(i) for i in item)
        return item

    seen = set()
    result = []
    for item in lst:
        hashable_item = to_hashable(item)
        if hashable_item not in seen:
            seen.add(hashable_item)
            result.append(item)
    return result

def fix_labels(row):
    # Deduplicate the labels in the LabelsLocalizationsBySentence column
    try:
        Labels = eval(row.Labels)
        Labels = fast_deduplicate(Labels)
        LabelsLocalizationsBySentence = eval(row.LabelsLocalizationsBySentence)
        LabelsLocalizationsBySentence = fast_deduplicate(LabelsLocalizationsBySentence)
        
        if "nan" in LabelsLocalizationsBySentence:
            LabelsLocalizationsBySentence = []
        if "nan" in Labels:
            Labels = []
    except:
        LabelsLocalizationsBySentence = []
        Labels = []
    
    row['LabelsLocalizationsBySentence'] = LabelsLocalizationsBySentence
    row['Labels'] = ",".join(Labels)
    return row

def load_cache(cache_file):
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, sep="\t")
    else:
        return pd.DataFrame(columns=['report_hash', 'Report', 'TranslatedReport'])

def write_cache(cache_dict, cache_file):
    # convert cache_dict to a dataframe. report_hash is the key right now and the values are a tuple of (Report, TranslatedReport)
    report_hashes = list(cache_dict.keys())
    reports = [cache_dict[report_hash][0] for report_hash in report_hashes]
    translated_reports = [cache_dict[report_hash][1] for report_hash in report_hashes]
    cache_df = pd.DataFrame({
        "report_hash": report_hashes,
        "Report": reports,
        "TranslatedReport": translated_reports
    })
    cache_df.to_csv(cache_file, index=False, sep="\t")

def call_openai(REPORT="torax suci . valor con clinic pacient .", EXTRA=""):
    USER_PROMPT = BASE_USER_PROMPT + REPORT + "\nLocalized labels: " + str(EXTRA)
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_format={ "type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ]
    )
    output = response.choices[0].message.content
    completion_tokes = response.usage.completion_tokens
    prompt_tokens = response.usage.prompt_tokens
    
    return output, completion_tokes, prompt_tokens

def is_valid_json_output_format(s):
    # Check if the string output by OpenAI is a valid JSON with the required structure {"output": "translated text"}
    try:
        # Parse the string as JSON
        data = json.loads(s)
        # Check if the JSON is a dictionary with the required structure. also check if it is not empty
        if isinstance(data, dict) and "output" in data and isinstance(data["output"], str) and data["output"].strip():
            return True
        else:
            return False
    except (json.JSONDecodeError, TypeError):
        return False
            
args = parse_args()

padchest = pd.read_csv(args.input_file)
padchest = padchest.dropna(subset=["Report"])
print(f"Loaded {len(padchest)} reports from PadChest dataset.")

padchest = padchest[padchest_columns]

print("Fixing labels in PadChest data...")
padchest = padchest.apply(fix_labels, axis=1)

print("Loading cache...")
read_cache = load_cache(args.cache_file)
print(f"Read {len(read_cache)} rows from cache")

# Convert into a dictionary with keys as read_cache['report_hash'] and values as a tuple of (Report, TranslatedReport)
cache_dict = {row['report_hash']: (row['Report'], row['TranslatedReport']) for _, row in read_cache.iterrows()}

texts_to_translate = set(padchest['Report'].tolist())

cache_write_counter = 0
for i, row in tqdm(padchest.iterrows(), total=len(padchest)):
    Report = row['Report']
    report_hash = hashlib.sha256(Report.encode()).hexdigest()
    if report_hash not in cache_dict:
        time.sleep(0.05)
        translated_text, completion_tokes, prompt_tokens = call_openai(REPORT=Report, EXTRA=row['LabelsLocalizationsBySentence'])
        if is_valid_json_output_format(translated_text):
            translated_text = json.loads(translated_text)["output"]
            print(f"Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokes}")
            print(f"Original text: {Report}\nTranslated text: {translated_text}")
            cache_dict[report_hash] = (Report, translated_text)
            cache_write_counter += 1
            
            # Write cache every 20 API calls
            if cache_write_counter >= 20:
                write_cache(cache_dict, args.cache_file)
                cache_write_counter = 0
        else:
            print(f"Skipping translation for row {i} as the output is not a valid JSON.")
    else:
        print(f"Skipping translation for row {i} as it is already in the cache.")
        continue
    print()

    
write_cache(cache_dict, args.cache_file)

padchest['TranslatedReport'] = ""
def get_translated_report(row):
    report_hash = hashlib.sha256(row['Report'].encode()).hexdigest()
    try:
        translated_text = cache_dict[report_hash][1]
        translated_text = " ".join(translated_text.split()).strip().strip(".").lower()
    except:
        translated_text = ""
    row['TranslatedReport'] = translated_text
    return row

for i, row in tqdm(padchest.iterrows(), total=len(padchest)):
    row = get_translated_report(row)
    padchest.at[i, 'TranslatedReport'] = row['TranslatedReport']
    
# drop the rows where TranslatedReport is "" and nan
padchest = padchest[padchest['TranslatedReport'].notna()]
padchest = padchest[padchest['TranslatedReport'] != ""]



print(f"Writing to output file: {args.output_file}")
padchest.to_csv(args.output_file, index=False)


  