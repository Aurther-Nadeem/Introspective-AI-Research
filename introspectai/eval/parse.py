import json
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class IntrospectionResult:
    detected: bool
    confidence: float
    concept: Optional[str]
    authorship: Optional[str]
    rationale: Optional[str]

def parse_introspection_json(text):
    """
    Parses the JSON output from the model.
    Handles cases where the model outputs multiple JSON objects or text surrounding them.
    Prioritizes a 'detected': True result if found.
    """
    # Find all potential JSON blocks (non-nested for simplicity, or just brace-delimited)
    # This regex matches a brace, then anything that is NOT a brace, or a nested brace block (simple 1-level nesting)
    # But simpler: just find all { ... } blocks non-greedily?
    # Actually, let's just use a simple approach: find all substrings starting with { and ending with }
    # and try to parse them.
    
    candidates = []
    
    # Simple regex for non-nested JSON objects
    # matches { ... } non-greedily
    matches = re.finditer(r"\{.*?\}", text, re.DOTALL)
    
    for match in matches:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            # Validate it has the keys we expect
            if "detected" in data:
                candidates.append(data)
        except json.JSONDecodeError:
            continue
            
    # If no simple matches, try the greedy one as a fallback (for single large objects)
    if not candidates:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                candidates.append(data)
            except:
                pass

    # Decide which candidate to return
    # Strategy: Return the first one that says detected=True. 
    # If none, return the last valid one.
    
    selected = None
    for data in candidates:
        if data.get("detected") is True:
            selected = data
            break
    
    if selected is None and candidates:
        selected = candidates[-1]
        
    if selected:
        return IntrospectionResult(
            detected=selected.get("detected", False),
            confidence=float(selected.get("confidence", 0.0)),
            concept=selected.get("concept"),
            authorship=selected.get("authorship"),
            rationale=selected.get("rationale")
        )
    else:
        # Fallback
        return IntrospectionResult(
            detected=False,
            confidence=0.0,
            concept=None,
            authorship=None,
            rationale=None
        )
