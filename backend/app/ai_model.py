import re
from typing import Dict, List

FOCUS_KEYWORDS: Dict[str, List[str]] = {
    "AI/ML": ["ai", "machine learning", "ml", "deep learning", "neural", "nlp", "computer vision", "classification", "prediction", "model"],
    "Data Analytics": ["data", "analytics", "dashboard", "visualization", "reporting", "insights", "statistical", "metrics"],
    "Web Development": ["web", "frontend", "backend", "api", "react", "ui", "ux", "full stack", "application"],
    "Automation": ["automation", "workflow", "pipeline", "integration", "orchestration", "bot", "scheduler"],
    "Security": ["security", "authentication", "authorization", "encryption", "privacy", "compliance"],
    "IoT": ["iot", "sensor", "hardware", "embedded", "device", "edge"],
    "Healthcare": ["health", "medical", "patient", "clinical", "diagnostic"],
    "Education": ["education", "learning", "student", "course", "training", "assessment"],
    "Finance": ["finance", "banking", "payment", "fraud", "risk", "investment"],
    "Sustainability": ["sustainability", "green", "energy", "environment", "carbon", "ecology"],
}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip().lower()


def _matched_focus_areas(text: str) -> List[str]:
    normalized = _normalize_text(text)
    matched = []
    for area, keywords in FOCUS_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            matched.append(area)
    return matched or ["General Productivity"]


def analyze_project_idea(title: str, domain: str, description: str) -> Dict[str, object]:
    title_text = _normalize_text(title)
    domain_text = _normalize_text(domain)
    description_text = _normalize_text(description)
    combined_text = f"{title_text} {domain_text} {description_text}"

    score = 35

    if any(keyword in combined_text for keyword in ["ai", "machine learning", "ml", "deep learning", "predict", "model", "analytics"]):
        score += 12
    if any(keyword in combined_text for keyword in ["dashboard", "reporting", "insights", "tracking", "analysis"]):
        score += 8
    if any(keyword in combined_text for keyword in ["api", "backend", "frontend", "web", "application", "software", "platform"]):
        score += 10
    if any(keyword in combined_text for keyword in ["automation", "workflow", "pipeline", "integration", "monitoring"]):
        score += 8
    if any(keyword in combined_text for keyword in ["security", "auth", "privacy", "encryption", "compliance"]):
        score += 6
    if len(description_text.split()) >= 25:
        score += 8
    if len(title_text.split()) >= 3:
        score += 4
    if domain_text:
        score += 6

    matched_areas = _matched_focus_areas(combined_text)
    score += min(15, len(matched_areas) * 4)
    score = max(0, min(100, score))

    if score >= 80:
        risk_level = "Low"
        recommendation = "High feasibility: proceed with implementation and validation."
    elif score >= 65:
        risk_level = "Medium"
        recommendation = "Moderate feasibility: refine the scope and validate the core use case before scaling."
    else:
        risk_level = "High"
        recommendation = "Low feasibility: simplify the idea, reduce complexity, and validate the problem before building at scale."

    return {
        "project_title": title.strip(),
        "domain": domain.strip() or "General",
        "feasibility_score": score,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "matched_focus_areas": matched_areas,
    }
