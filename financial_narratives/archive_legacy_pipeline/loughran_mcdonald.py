"""
Loughran-McDonald Financial Sentiment Lexicon
==============================================
Finance-specific sentiment dictionary based on Loughran & McDonald (2011).
This module provides curated word lists for six key sentiment dimensions
commonly used in financial text analysis.

Reference:
    Loughran, T. & McDonald, B. (2011). When Is a Liability Not a Liability?
    Textual Analysis, Dictionaries, and 10-Ks. Journal of Finance, 66(1), 35-65.

For the full dictionary, download from:
    https://sraf.nd.edu/loughranmcdonald-master-dictionary/

The word lists below are representative subsets sufficient for research-grade
analysis of financial news, central bank communications, and social media.
"""

import re
import collections
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Loughran-McDonald sentiment categories (representative subsets)
# ---------------------------------------------------------------------------

LM_NEGATIVE = {
    # Core negative sentiment in financial contexts
    "abandon", "abdicate", "aberrant", "aberration", "abrupt", "absence",
    "abuse", "abyss", "accident", "accuse", "acquit", "admonish", "adverse",
    "aftermath", "aggravate", "allegation", "annihilate", "anomalous",
    "anxiety", "attrition", "aversion",
    "bail", "bailout", "bankrupt", "bankruptcy", "bear", "bearish", "blame",
    "bleak", "breach", "breakdown", "bubble", "burden",
    "calamity", "capitulate", "catastrophe", "caution", "cease", "closure",
    "collapse", "collateral", "complain", "concern", "contagion",
    "contraction", "correction", "crash", "crisis", "critical", "curtail",
    "damage", "deadlock", "debacle", "debt", "decline", "default", "deficit",
    "deflation", "delinquency", "delinquent", "deplete", "depreciate",
    "depression", "destabilize", "deteriorate", "devalue", "disappoint",
    "disaster", "disclose", "discontinue", "discrepancy", "disinflation",
    "dislocate", "dislocation", "dismal", "dispute", "distort", "distress",
    "downturn", "drag", "drought",
    "erode", "erosion", "escalate", "evict", "exacerbate", "excessive",
    "exhaust", "exigent", "exposure",
    "fail", "failure", "falter", "fear", "flee", "flight", "forbearance",
    "force", "foreclose", "foreclosure", "forfeit", "fraud", "freeze",
    "gridlock", "grim",
    "halt", "hardship", "harm", "harsh", "havoc", "hazard", "headwind",
    "illiquid", "illiquidity", "impair", "impairment", "impediment",
    "implode", "impose", "inability", "inadequate", "insolvent", "instability",
    "insufficient",
    "jeopardize", "junk",
    "lack", "lag", "languish", "layoff", "leak", "liability", "liquidate",
    "liquidation", "litigation", "lose", "loss", "losses",
    "malfunction", "manipulate", "margin", "mismanage", "mislead",
    "mispricing", "miss", "moratorium",
    "neglect", "negative", "nonperforming",
    "obstacle", "offload", "onerous", "outage", "outflow", "overdue",
    "overexposure", "overextend", "overleveraged", "overvalue",
    "panic", "penalty", "peril", "pessimism", "pessimistic", "plummet",
    "plunge", "precarious", "precipitate", "pressure", "problem",
    "prosecute", "protest", "pullback",
    "rattle", "reassess", "recession", "reckless", "recoil", "relinquish",
    "reluctance", "repercussion", "retaliate", "retract", "retreat",
    "retrench", "reversal", "revoke", "risk", "risks", "risky", "run",
    "rupture",
    "sacrifice", "sanction", "scam", "scandal", "scare", "scrutiny",
    "seizure", "selloff", "setback", "severe", "shortage", "shortfall",
    "shrink", "shutdown", "skeptic", "skepticism", "slacken", "slash",
    "slide", "slip", "slow", "slowdown", "slump", "spike", "spiral",
    "squeeze", "stagnant", "stagnation", "stall", "steep", "strain",
    "stress", "stringent", "struggle", "stumble", "subprime",
    "suffer", "surge", "suspect", "suspend", "suspension", "swindle",
    "taper", "tension", "terminate", "threat", "threaten", "tighten",
    "toxic", "trouble", "tumble", "turbulence", "turmoil",
    "unable", "uncertain", "undermine", "underperform", "underwater",
    "unfavorable", "unfortunate", "unpaid",
    "violate", "volatile", "volatility", "vulnerable",
    "warn", "warning", "weaken", "woes", "worsen", "worst", "writedown",
    "writeoff",
}

LM_POSITIVE = {
    "accomplish", "achieve", "advance", "advantage", "affirm", "appreciate",
    "approval", "approve", "assure", "attain",
    "benefit", "bolster", "boom", "boost", "breakout", "breakthrough",
    "broad", "buoyant",
    "certainty", "champion", "climb", "collaborate", "command", "commend",
    "competent", "compliment", "confidence", "confident", "constructive",
    "contributor", "convenient", "conviction", "cooperate", "creative",
    "cushion",
    "decisive", "deliver", "dependable", "desirable", "diligent",
    "distinguish", "dividend", "dominant", "durable", "dynamic",
    "earn", "ease", "efficient", "elevate", "empower", "enable",
    "encourage", "endorse", "enhance", "enjoy", "enrich", "enthusiasm",
    "excel", "exceptional", "excitement", "exclusive", "exemplary", "expand",
    "favorable", "flourish", "fortify", "fruitful", "fulfill", "fund",
    "gain", "generate", "genuine", "good", "grant", "gratify", "great",
    "grow", "growth", "guarantee",
    "healthy", "heighten", "helpful",
    "ideal", "improve", "improvement", "incentive", "increase", "ingenuity",
    "innovate", "innovation", "insightful", "inspire", "integrity",
    "invigorate",
    "landmark", "launch", "leadership", "leverage",
    "milestone", "momentum",
    "notable",
    "opportunity", "optimal", "optimism", "optimistic", "outpace",
    "outperform", "outstanding",
    "peak", "pioneer", "pleasant", "pledge", "plentiful", "positive",
    "premium", "prevail", "proactive", "productive", "proficiency",
    "profit", "profitable", "progress", "prominent", "promising", "prosper",
    "prosperous", "proven", "prudent",
    "rally", "rebound", "reassure", "recover", "recovery", "refine",
    "reform", "reinforce", "reliable", "remarkable", "resilience",
    "resilient", "resolve", "restore", "return", "revive", "reward",
    "robust",
    "safe", "satisfy", "secure", "shareholder", "skill", "smooth",
    "solid", "solution", "sound", "soar", "stable", "stability",
    "steady", "stimulate", "strength", "strengthen", "strong", "succeed",
    "success", "successful", "superior", "support", "surge", "surplus",
    "sustain", "sustainable", "swift",
    "talent", "thrive", "top", "transform", "transparent", "triumph",
    "trust", "upbeat", "upgrade", "upside", "upturn",
    "value", "vertex", "vibrant", "vigor", "vision", "vital",
    "welcome", "yield",
}

LM_UNCERTAINTY = {
    "almost", "ambiguity", "ambiguous", "anticipate", "apparent",
    "approximately", "assume", "assumption",
    "believe", "bet",
    "chance", "conceivable", "conceivably", "conditional", "confuse",
    "contingency", "contingent", "could",
    "depend", "deviate", "doubt", "doubtful",
    "equivocal", "estimate", "exposure",
    "fluctuate", "fluctuation", "forecast",
    "generally", "guess",
    "hypothetical",
    "imprecise", "improbable", "incalculable", "incompatible",
    "inconclusive", "indefinite", "indeterminate", "inexact",
    "instability", "intangible",
    "likelihood", "likely",
    "may", "maybe", "might",
    "nearly", "noncommittal",
    "obscure", "occasional", "odds",
    "pending", "perceive", "perhaps", "plausible", "possible", "possibly",
    "precaution", "predict", "prediction", "preliminary", "presumably",
    "presume", "probabilistic", "probability", "probable", "probably",
    "project", "projection", "prone",
    "random", "rarely", "reckon", "reconsider", "risky", "roughly",
    "rumor",
    "seem", "seldom", "skeptic", "skepticism", "sometime", "somewhat",
    "speculate", "speculation", "sporadic", "suggest", "suppose",
    "susceptible",
    "tend", "tentative", "turbulence",
    "uncertain", "uncertainty", "unclear", "unconfirmed", "undecided",
    "undefined", "undetermined", "uneasy", "unforeseeable", "unforeseen",
    "unknown", "unlikely", "unobservable", "unpredictable", "unproven",
    "unquantifiable", "unresolved", "unsettled", "unspecified", "unsure",
    "untested",
    "vagary", "vague", "variable", "vary", "volatile", "volatility",
    "wonder",
}

LM_LITIGIOUS = {
    "acquit", "adjudicate", "allegation", "allege", "amend", "appeal",
    "arbitrate", "arbitration", "arraign",
    "breach",
    "cease", "claimant", "coerce", "collusion", "compliance", "comply",
    "compel", "confiscate", "consent", "conspiracy", "contempt",
    "contractual", "convict", "counsel", "counterclaim", "court",
    "damages", "decree", "defendant", "defraud", "depose", "deposition",
    "discovery", "dismiss", "docket", "enforce", "enjoin",
    "felony", "fiduciary", "file", "fine", "forfeiture", "fraud",
    "grievance", "guilt",
    "hearing",
    "imprisoned", "indict", "indictment", "infraction",
    "injunction", "inspect", "interlocutory", "interrogatories",
    "judgment", "judicature", "judicial", "jurisdiction", "jury",
    "lawsuit", "legislation", "liabilities", "liability", "libel",
    "litigate", "litigation",
    "malfeasance", "mandate", "misdemeanor", "misconduct",
    "negligence", "noncompliance",
    "oath", "obstruct", "offend", "ordinance",
    "penalize", "penalty", "petition", "plaintiff", "plea", "plead",
    "precedent", "probation", "prohibit", "prosecute", "prosecution",
    "punish", "punitive",
    "recourse", "regulate", "regulation", "regulator", "remedial", "remedy",
    "repeal", "restitution", "restrain", "revoke", "rule", "ruling",
    "sanction", "settlement", "statute", "statutory", "subpoena", "sue",
    "summon",
    "testify", "testimony", "tort", "tribunal",
    "vacate", "verdict", "violate", "violation", "warrant", "witness",
}

LM_CONSTRAINING = {
    "abide", "bound",
    "cap", "ceiling", "circumscribe", "commitment", "compel", "comply",
    "compulsory", "condition", "confine", "constrain", "constraint",
    "contingent", "contractual", "covenant", "curb", "curtail",
    "dictate", "disallow",
    "encumber", "enforce", "entail", "exceed", "explicit",
    "forbid", "force",
    "hamper", "hinder",
    "impose", "impound", "inhibit", "injunction",
    "limit", "limitation",
    "mandate", "mandatory", "maximum", "minimum", "must",
    "necessitate",
    "obligate", "obligation", "oblige", "obstruct",
    "peg", "preclude", "prerequisite", "prescribe", "prohibit",
    "provision",
    "quota",
    "refrain", "require", "requirement", "restrain", "restrict",
    "restriction",
    "shall", "stipulate", "subject",
    "tighten",
    "unable",
}

LM_STRONG_MODAL = {
    "always", "best", "clearly", "definitely", "definitively",
    "highest", "must", "never", "shall", "strongest", "will",
}

LM_WEAK_MODAL = {
    "almost", "apparently", "approximately", "conceivably", "could",
    "depend", "depending", "fairly", "generally", "largely",
    "likely", "mainly", "may", "maybe", "might", "mostly",
    "nearly", "occasionally", "often", "partially", "perhaps",
    "possibly", "potentially", "predominantly", "presumably",
    "probably", "quite", "rather", "reasonably", "roughly",
    "seldom", "seem", "seems", "slightly", "sometimes",
    "somewhat", "suggest", "suggests", "tend", "tends",
    "typically", "unlikely", "usually",
}

# Aggregate dictionary with all categories
LM_CATEGORIES: Dict[str, set] = {
    "negative": LM_NEGATIVE,
    "positive": LM_POSITIVE,
    "uncertainty": LM_UNCERTAINTY,
    "litigious": LM_LITIGIOUS,
    "constraining": LM_CONSTRAINING,
    "strong_modal": LM_STRONG_MODAL,
    "weak_modal": LM_WEAK_MODAL,
}


def compute_lm_densities(texts: List[str]) -> Dict[str, dict]:
    """
    Compute Loughran-McDonald sentiment densities for a list of texts.

    Returns a dict keyed by category name, each containing:
        - density: float (hits / total_words)
        - top_words: list of (word, count) tuples
        - total_hits: int
        - total_words: int
    """
    word_counts = {cat: collections.Counter() for cat in LM_CATEGORIES}
    total_words = 0

    for text in texts:
        words = re.findall(r'\b[a-z]+\b', text.lower())
        total_words += len(words)
        for w in words:
            for cat, lexicon in LM_CATEGORIES.items():
                if w in lexicon:
                    word_counts[cat][w] += 1

    results = {}
    for cat in LM_CATEGORIES:
        hits = sum(word_counts[cat].values())
        results[cat] = {
            "density": hits / total_words if total_words > 0 else 0.0,
            "top_words": word_counts[cat].most_common(10),
            "total_hits": hits,
            "total_words": total_words,
        }

    return results


def compute_lm_tone(texts: List[str]) -> float:
    """
    Compute Loughran-McDonald tone = (positive - negative) / (positive + negative).
    Returns a value in [-1, 1] or 0 if no sentiment words found.
    """
    densities = compute_lm_densities(texts)
    pos = densities["positive"]["total_hits"]
    neg = densities["negative"]["total_hits"]
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0


def compute_modal_strength_ratio(texts: List[str]) -> float:
    """
    Ratio of strong modal words to total modal words.
    Higher values indicate more assertive, definitive language.
    """
    densities = compute_lm_densities(texts)
    strong = densities["strong_modal"]["total_hits"]
    weak = densities["weak_modal"]["total_hits"]
    total = strong + weak
    return strong / total if total > 0 else 0.5


def compare_sources(
    x_texts: List[str],
    wsj_texts: List[str],
    official_texts: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare Loughran-McDonald profiles across sources.
    Returns density for each category for each source.
    """
    sources = {"x": x_texts, "wsj": wsj_texts}
    if official_texts:
        sources["official"] = official_texts

    comparison = {}
    for source_name, texts in sources.items():
        if texts:
            densities = compute_lm_densities(texts)
            comparison[source_name] = {
                cat: info["density"] for cat, info in densities.items()
            }
            comparison[source_name]["tone"] = compute_lm_tone(texts)
            comparison[source_name]["modal_strength"] = compute_modal_strength_ratio(texts)
        else:
            comparison[source_name] = {cat: 0.0 for cat in LM_CATEGORIES}
            comparison[source_name]["tone"] = 0.0
            comparison[source_name]["modal_strength"] = 0.5

    return comparison
