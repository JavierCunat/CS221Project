
#!/usr/bin/env python3
"""
compat_scorer.py
----------------
A tiny, dependency-light baseline for physician compatibility scoring.

Features:
- Structured overlap scores (hospitals, insurances, metro geography, specialty complement)
- Simple text similarity (cosine over bag-of-words)
- Optional: sentence-transformer embedding similarity if installed (see --use-embeddings)

Usage:
    python compat_scorer.py --provider provider.json --specialist specialist.json --research research.txt
    # or run with the built-in example for Araim ↔ Kiel:
    python compat_scorer.py --example araim_kiel

Installation (optional embeddings):
    pip install sentence-transformers
    # first run may download a model: all-MiniLM-L6-v2
"""

import math
import json
import argparse
from collections import Counter
from typing import Dict, List, Any, Tuple

# ---------- Text helpers ----------

def norm(s: str) -> str:
    return (s or "").strip().lower()

def tokenize(text: str) -> List[str]:
    """A simple alnum tokenizer; no external deps."""
    t = [] #list of clean tokens which are words seperated by punc/space/non alphnum
    w = [] #temporary buffer
    for ch in (text or "").lower():
        if ch.isalnum():
            w.append(ch)
        else:
            if w:
                t.append(''.join(w))
                w = []
    if w:
        t.append(''.join(w))
    return [tok for tok in t if tok]

def cosine_bow(a: str, b: str) -> float:
    """Cosine similarity on bag-of-words counts by comparing word frequency vectors"""
    tokenized_text_a = Counter(tokenize(a))
    tokenized_text_b = Counter(tokenize(b))
    if not tokenized_text_a or not tokenized_text_b:
        return 0.0
    dot = sum(tokenized_text_a[k] * tokenized_text_b.get(k, 0) for k in tokenized_text_a) #sum equal freqs
    #euclidean norms, diving by norms normalizes for text length longer notes don't automatically look more similar
    na = math.sqrt(sum(v*v for v in tokenized_text_a.values())) 
    nb = math.sqrt(sum(v*v for v in tokenized_text_b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def jaccard(list_a: List[str], list_b: List[str]) -> float:
    # returns intersection of words a and b over union of words a and b
    A = {norm(x) for x in list_a if x}
    B = {norm(x) for x in list_b if x}
    if not A and not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def overlap_ratio(list_a: List[str], list_b: List[str]) -> float:
    """|A ∩ B| / min(|A|, |B|) to avoid penalizing different breadths."""
    A = [norm(x) for x in list_a if x]
    B = [norm(x) for x in list_b if x]
    if not A or not B:
        return 0.0
    inter = len(set(A) & set(B))
    denom = min(len(set(A)), len(set(B)))
    return inter / denom if denom else 0.0

def extract_city(s: str) -> str:
    if not s:
        return ""
    parts = [p.strip() for p in s.split(",")]
    return parts[0].lower()

def same_metro(city_a: str, city_b: str) -> bool:
    """Minimal metro heuristic; Fresno~Clovis considered same metro."""
    a = extract_city(city_a)
    b = extract_city(city_b)
    metro_groups = [
        {"fresno", "clovis"},
        {"san francisco", "oakland", "berkeley", "redwood city", "palo alto", "menlo park", "burlingame", "san mateo"},
        {"los angeles", "glendale", "pasadena", "santa monica", "culver city", "burbank"},
    ]
    for g in metro_groups:
        if a in g and b in g:
            return True
    return a == b and a != ""

def specialty_complement_score(spec_a: str, spec_b: str) -> float:
    """Heuristic complementarity: cardiology <-> cardiothoracic = 1.0"""
    s1 = norm(spec_a)
    s2 = norm(spec_b)
    if s1 == s2 and s1:
        return 0.6
    cardio = ("cardio" in s1) or ("cardio" in s2)
    thoracic_or_surg = ("thoracic" in s1 or "surgery" in s1) or ("thoracic" in s2 or "surgery" in s2)
    if cardio and thoracic_or_surg:
        return 1.0
    if ("internal" in s1 or "internal" in s2) and thoracic_or_surg:
        return 0.8
    if ("pulmon" in s1 or "pulmon" in s2) and thoracic_or_surg:
        return 0.8
    return 0.7 if s1 or s2 else 0.0

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def label_from_score(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.45:
        return "Moderate"
    return "Low"

# ---------- Optional embeddings ----------

def embedding_similarity(text_a: str, text_b: str) -> Tuple[float, str]:
    """
    If sentence-transformers is available, compute cosine similarity between
    'all-MiniLM-L6-v2' embeddings. If not available, return (0.0, reason).
    """
    try:
        from sentence_transformers import SentenceTransformer, util as st_util
    except Exception as e:
        return 0.0, f"Embeddings disabled (sentence-transformers not installed): {e}"
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        a = model.encode([text_a], normalize_embeddings=True)
        b = model.encode([text_b], normalize_embeddings=True)
        sim = float((a @ b.T)[0][0])
        return sim, "ok"
    except Exception as e:
        return 0.0, f"Embeddings unavailable or model load failed: {e}"

# ---------- Core scoring ----------

DEFAULT_WEIGHTS = {
    "hospitals": 0.30,
    "insurance": 0.25,
    "geography": 0.20,
    "specialty": 0.15,
    "text":      0.10,
}

def build_provider_text(provider: Dict[str, Any]) -> str:
    #if json has content section get from there
    prov_text = provider.get("content","")
    if prov_text:
        return prov_text
    #if not build the content
    rp = provider.get("referring_physician", {})
    prov_text = " ".join(filter(None, [
        rp.get("practice_focus",""),
        " ".join(rp.get("specialties",[])),
        " ".join(provider.get("compatibility_factors", {}).get("referral_priorities", [])),
        " ".join(provider.get("referral_context", {}).get("common_referral_scenarios", [])),
    ]))
    return prov_text

def compute_compatibility(provider: Dict[str, Any],
                          specialist: Dict[str, Any],
                          research_text: str = "",
                          use_embeddings: bool = False,
                          weights: Dict[str, float] = None) -> Dict[str, Any]:
    weights = weights or DEFAULT_WEIGHTS

    # Extract structured fields
    prov_name = provider.get("provider_name") or provider.get("name") or provider.get("referring_physician", {}).get("name", "")
    prov_spec = provider.get("specialty") or provider.get("referring_physician", {}).get("specialty", "")
    prov_loc  = provider.get("location") or provider.get("referring_physician", {}).get("location", "")
    prov_hosp = provider.get("hospital_affiliations") or provider.get("referring_physician", {}).get("hospital_affiliations", [])
    prov_ins  = provider.get("accepted_insurance") or provider.get("referring_physician", {}).get("accepted_insurance", [])
    prov_text = build_provider_text(provider)

    spec_name = specialist.get("name") or specialist.get("provider_name","")
    spec_spec = specialist.get("specialty","")
    spec_loc  = specialist.get("location","")
    spec_hosp = specialist.get("hospital_affiliations",[])
    spec_ins  = specialist.get("accepted_insurance", [])

    # Structured scores
    hosp_overlap = overlap_ratio(prov_hosp, spec_hosp)
    ins_overlap  = overlap_ratio(prov_ins, spec_ins) if prov_ins and spec_ins else 0.0
    geo_score    = 1.0 if same_metro(prov_loc, spec_loc) else (0.0 if not prov_loc or not spec_loc else 0.3)
    spec_comp    = specialty_complement_score(prov_spec, spec_spec)

    # Text similarity on provider (Ahraim by joining its fields) and target providers research.txt
    text_sim_bow = cosine_bow(prov_text, research_text)
    text_sim = text_sim_bow
    emb_note = "bow"
    if use_embeddings:
        emb_sim, note = embedding_similarity(prov_text, research_text)
        emb_note = note
        # Blend embeddings (70%) with bow (30%) if embeddings available
        if note == "ok":
            text_sim = 0.7 * emb_sim + 0.3 * text_sim_bow

    # Weighted sum
    score = (
        weights["hospitals"] * hosp_overlap +
        weights["insurance"] * ins_overlap +
        weights["geography"] * geo_score +
        weights["specialty"] * spec_comp +
        weights["text"]      * text_sim
    )
    score = clamp01(score)

    return {
        "pair": f"{prov_name} ↔ {spec_name}",
        "components": {
            "hospitals_overlap": round(hosp_overlap, 3),
            "insurance_overlap": round(ins_overlap, 3),
            "geography": round(geo_score, 3),
            "specialty_complement": round(spec_comp, 3),
            "text_similarity": round(text_sim, 3),
            "text_similarity_bow": round(text_sim_bow, 3),
            "text_similarity_note": emb_note,
            "weights": weights
        },
        "compatibility_score": round(score, 4),
        "label": label_from_score(score)
    }

# ---------- Example inputs for Araim ↔ Kiel ----------

ARAIM_PROVIDER = {
  "referring_physician": {
    "name": "Dr. Leheb H. Araim",
    "specialty": "cardiothoracic surgeon",
    "location": "Fresno, CA",
    "hospital_affiliations": [
      "Community Regional Medical Center",
      "UCSF Fresno",
      "Clovis Community Medical Center"
    ],
    "specialties": [
      "cardiothoracic surgery",
      "heart surgery",
      "lung surgery",
      "thoracic surgery"
    ],
    "practice_focus": "complex cardiac and thoracic procedures",
    "accepted_insurance": [
      "Medicare",
      "Medi-Cal",
      "Aetna",
      "Anthem Blue Cross",
      "Blue Shield of CA",
      "Cigna",
      "UnitedHealthcare",
      "Health Net",
      "TriCare/TriWest"
    ],
  },
  "compatibility_factors": {
    "referral_priorities": [
      "clinical expertise",
      "geographic proximity",
      "cultural competency"
    ]
  },
  "referral_context": {
    "common_referral_scenarios": [
      "advanced heart failure patients needing surgical evaluation",
      "post-operative cardiac patients needing follow-up care",
      "complex thoracic cases requiring multidisciplinary approach"
    ]
  }
}

KIEL_SPECIALIST = {
    "name": "Dr. Richard G. Kiel",
    "specialty": "Cardiology (Advanced Heart Failure & Transplant)",
    "location": "Clovis, CA",
    "hospital_affiliations": [
        "Community Regional Medical Center",
        "Clovis Community Medical Center"
    ],
    "accepted_insurance": [
        "Medicare",
        "Aetna",
        "Anthem Blue Cross",
        "Blue Shield of CA",
        "Cigna",
        "UnitedHealthcare"
    ]
}

KIEL_RESEARCH = """Dr. Richard G. Kiel Overview
Specialty: Cardiologist with subspecialty in Advanced Heart Failure & Transplant Cardiology (ABIM board-certified).
Training: UCSF Fresno IM residency and chief residency; UCSF Fresno Cardiology fellowship; UCSF Advanced HF & Transplant fellowship.
Current group & hospitals: Co-founder, Advanced Cardiovascular Specialists of Central California (Clovis). Active at Community Regional Medical Center (CRMC) and Clovis Community Medical Center.
Core focus: Advanced heart failure (HFrEF/HFpEF), cardiogenic shock, device-guided HF care (CardioMEMS), transplant/LVAD evaluations with local optimization and co-management.
One-sentence take: Highly compatible for seamless CT surgery co-management.
Key Compatibility: shared institutions (Community Regional, Clovis Community), heart-team model, physical proximity (suites 221 vs 223 at 729 N. Medical Center Dr W).
Insurance: Medicare and major commercial plans (Aetna, Anthem, Blue Shield, Cigna, UHC); not Kaiser.
Referral Activation: pre-op optimization for CABG/valve, post-op HF co-management, cardiogenic shock/MCS coordination, CardioMEMS-enabled recovery pathways.
"""

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Physician compatibility scorer (baseline).")
    parser.add_argument("--provider", type=str, help="Path to provider JSON")
    parser.add_argument("--specialist", type=str, help="Path to specialist JSON")
    parser.add_argument("--research", type=str, help="Path to research text file")
    parser.add_argument("--use-embeddings", action="store_true", help="Use sentence-transformers embeddings if available")
    parser.add_argument("--example", type=str, choices=["araim_kiel"], help="Run built-in example inputs")
    args = parser.parse_args()

    if args.example == "araim_kiel":
        provider = ARAIM_PROVIDER
        specialist = KIEL_SPECIALIST
        research_text = KIEL_RESEARCH
    else:
        if not (args.provider and args.specialist and args.research):
            parser.error("Provide --example araim_kiel OR all of --provider --specialist --research")
        with open(args.provider, "r", encoding="utf-8") as f:
            provider = json.load(f)
        with open(args.specialist, "r", encoding="utf-8") as f:
            specialist = json.load(f)
        with open(args.research, "r", encoding="utf-8") as f:
            research_text = f.read()

    result = compute_compatibility(provider, specialist, research_text, use_embeddings=args.use_embeddings)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
