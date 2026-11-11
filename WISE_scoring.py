"""
Author
    * Venkata Annadi
    * Yeojeong Kim
This scirpt is to calculate WISE score between GT and hypothesis for Weighted Intent & Semantic Evaluation.
"""

import argparse
import os
import sys
import json
import math
import re
from typing import Any
from google import genai

SYSTEM_PROMPT = """
### ROLE
    You are an expert ASR (Automatic Speech Recognition) Analyst. You will compare English Ground Truth (GT) and Hypothesis (HYP).
### TASK
    1. Identify every mismatching word.
    2. Assign an ERROR_TYPE:
        NEGATION: Inverting meaning (e.g., "do" vs "don't").
        NAMED_ENTITY: (person/org/location/brand/product/title)
        COMMON_ENTITY: (generic nouns like "bag", "table", "coffee")
        VALUE: Numbers, dates, times, currency.
        ACTION: Verbs that change the intended task.
        HALLUCINATION: HYP contains text not found in GT.
        OMISSION: GT contains text not found in HYP.
        SAFETY: Benign words misheard as profanity or slurs.
        FORMALITY: Grammar, tense, or filler word (um/ah) differences.
    3. Assign a SEVERITY_LEVEL and SEVERITY_SCORE for each word:
        * CRITICAL (9-10): Reverses meaning (negation), misrecognizes Named entities, introduces safety risk, or breaks a command.
        * MAJOR (6-8): The intent is understood but key information is wrong like Values, Common entities (e.g., "set a timer for 5 minutes" → "set a timer for 15 minutes").
        * MINOR (3-5): Understandable but requires mental effort to correct.
        * NEGLIGIBLE (1-2): FORM, Cosmetic, punctuation, grammatical (Tense/Morphological variation)
            Examples - helped --> help, is running --> runs, was --> is, singular --> plural.
        FILLER(0): removal of filler words.
        *** This MUST be assigned based on the impact of that word on the whole sentence. ***
    4. Calculate the semantic_distance for each word(0 being highly similar, 1 being not at all similar) between GT and Hypothesis.
    5. Calculate the costs for each error:
        - substitution cost = SEVERITY_SCORE * semantic_distance
        - deletion cost = SEVERITY_SCORE of the missing GT word * del_type. 
            1. Must get del_type float using the below information - 
                * FILLER/FORMALITY: 0.1
                * Non-critical function words (a/the/to/of): 0.2–0.3
                * Content words (nouns/verbs/adjectives): 0.4–0.6
                * COMMON ENTITY (generic nouns like "bag", "table", "coffee"): 0.6-0.8
                * VALUE/NEGATION/ACTION: 0.8-1.0
                * NAMED ENTITY (person/org/location/brand/product/title): 1.0
        - insertion cost = 0.5 * SEVERITY_SCORE of the inserted word.
        - sum of all costs must be 0 IF AND ONLY IF the Hypothesis is exact match to GT otherwise it must be some value greater than 0.
    6. Give the entire GT sentence with severity level (Must follow above SEVERITY_LEVEL instructions and strictly greater than or equal to 1 if its not a filler word) as severity_words. Format must be sentence of GT_word(SEVERITY_LEVEL)
    7. Then add an 'IntentJump':
        - IntentJump = 1 if the ASR output changes or cancel the original intent or if it breaks an idiom; otherwise 0
        - It must be 1 ONLY if the intent/meaning of the GT is totally flipped or broken in the Hypothesis. 
    8. Calculate the sentence_semantic_distance (0 being highly similar, 1 being not at all similar) between GT and Hypothesis.

### Output format
*** MUST FOLLOW THIS OUTPUT SCHEMA (JSON ONLY) ***
{
  "mismatches": [
    {
      "gt_span": "string",
      "hyp_span": "string",
      "error_type": "string",
      "SEVERITY_LEVEL": "string",
      "SEVERITY_SCORE": "int",
      "reason": "string", 
      "del_type": "float"
    }
  ],
  "IntentJump": "int",
  "sum of all costs": "float",
  "sum of GT SEVERITY_LEVEL": "int",
  "sentence_semantic_distance": "float",
  "severity_words": "string"
}
"""

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default
        
def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default
        
def build_user_prompt(gt, hyp):
    return f"GT:\n{gt}\n\nHYP:\n{hyp}"
    
def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]
        
def compute_gt_severity_sum_from_severity_words(severity_words):
    if not severity_words:
        return 0
    # Sum all integers that appear in parentheses: word(9)
    nums = re.findall(r"\((\d+)\)", severity_words)
    return sum(int(n) for n in nums)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_file", type=str, required=True)
    ap.add_argument("--hyp_file", type=str, required=True)
    ap.add_argument("--output_tsv", type=str, required=True)
    ap.add_argument("--model", type=str, default="gemini-2.5-pro")
    args = ap.parse_args()

    #Checking if gt and hyp file are valid
    gt_lines = read_lines(args.gt_file)
    hyp_lines = read_lines(args.hyp_file)
    if len(gt_lines) != len(hyp_lines):
        print(f"ERROR: Line count mismatch: GT={len(gt_lines)} HYP={len(hyp_lines)}", file=sys.stderr)
        return 2
        
    # Keep your existing client usage pattern
    client = genai.Client(api_key="") #add your own api key
    os.makedirs(os.path.dirname(args.output_tsv) or ".", exist_ok=True)
    # Token counters (your originals)
    total_token_count = 0
    num_rows = 0

    # Keep gen config stable
    gen_cfg = {
        "system_instruction": SYSTEM_PROMPT,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "seed": 42,
        "response_mime_type": "application/json",
    }

    #Funtion to call the model
    def call_model_fn(user_prompt: str):
        resp = client.models.generate_content(
            model=args.model,
            contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
                top_k=gen_cfg["top_k"],
                seed=gen_cfg["seed"],
                response_mime_type=gen_cfg["response_mime_type"],
            ),
        )
        text = (resp.text or "").strip()
        try:
            data = json.loads(text) if text else {}
        except json.JSONDecodeError:
            # Write a row with blanks but preserve GT/HYP so you can spot failures.
            out_f.write(f"{gt}\t{hyp}\t\t0\t\t\tMODEL_RETURNED_INVALID_JSON\n")
            return {}, {}
        usage = {"total_token_count": getattr(resp.usage_metadata, "total_token_count", 0),}
        return data, usage
        
    with open(args.output_tsv, "w", encoding="utf-8") as out_f:
        out_f.write("GT\tHYP\tERROR_TYPE\tWISE_SCORE\tIntentJump\tsum_of_all_costs\tsum_of_GT_severity\tseverity_words\tsentence_semantic_distance\tmismatch_explaination\n")
        for gt, hyp in zip(gt_lines, hyp_lines):
            base_user_prompt = build_user_prompt(gt, hyp)
            parsed, usage = call_model_fn(base_user_prompt)
            num_rows+=1
            
            #Token usage count
            total_token_count += safe_int(usage.get("total_token_count", 0))
            
            if parsed is None:
                num_failures += 1
                out_f.write(f"{gt}\t{hyp}\t\t0\t\t\t\t\t\tMODEL_RETURNED_INVALID_JSON\n")
                continue

            #Parsing and calculation
            IntentJump = safe_int(parsed.get("IntentJump", 0))
            severity_words = parsed.get("severity_words", "")
            sentence_semantic_distance = safe_float(parsed.get("sentence_semantic_distance", 0.0))
            sum_of_all_costs = parsed.get("sum of all costs", 0)            
            computed_sum = compute_gt_severity_sum_from_severity_words(severity_words)
            sum_of_GT_severity = computed_sum
            
            if sum_of_GT_severity != 0:
                SW_SER = round(sum_of_all_costs / sum_of_GT_severity, 4) 
            else:
                SW_SER = 0
                print("Sum of severity of GT words can't be 0")

            #Mismatch analysis
            mismatches = parsed.get("mismatches", []) or []
            # print(mismatches)
            severity_score = 0
            error_type = []
            for mismatch in mismatches:
                s = safe_int(mismatch.get("SEVERITY_SCORE", 0))
                et = mismatch.get("error_type", "")
                if s > severity_score:
                    severity_score = s
                    error_type = [et]
                elif s == severity_score and et:
                    error_type.append(et)
                    
            if not mismatches:
                error_type = [None]
                
            #Formula Implementation    
            sementic_error_score = min(1,SW_SER + (IntentJump * (1 - math.exp(-0.5 * sentence_semantic_distance))))
            wise_score = round((1 - sementic_error_score) * 100, 2)
            out_f.write(f"{gt}\t{hyp}\t{error_type}\t{wise_score}\t{IntentJump}\t{sum_of_all_costs}\t{sum_of_GT_severity}\t{severity_words}\t{sentence_semantic_distance}\t{mismatches}\n")

    # Original prints
    print(f"total_tokens is {total_token_count}")
    print(f"total {num_rows} are processed.")
    return 0

    
if __name__ == "__main__":
    raise SystemExit(main())
