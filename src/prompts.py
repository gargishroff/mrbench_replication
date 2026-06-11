JUDGE_PROMPT = """You are a careful, impartial evaluator of AI tutor responses. \
You will be shown a math-tutoring dialogue. The final student turn contains a \
mistake or confusion. You will then be shown a candidate tutor response that \
is attempting to remediate that mistake. Your job is to grade the candidate \
response on four pedagogical-ability dimensions.

# The four dimensions

1. **Mistake Identification (MI)** - Does the response acknowledge that the \
student made a mistake, even implicitly? Implicit acknowledgement (e.g. "let's \
take another look at that step") counts.

2. **Mistake Location (ML)** - Does the response correctly point to the \
specific part of the student's reasoning that is wrong? It must indicate \
*where* the mistake is, not just *that* there is one.

3. **Providing Guidance (PG)** - Does the response offer correct, relevant \
help toward fixing the mistake (a hint, a leading question, a worked sub-step)? \
Critically, giving the full answer is NOT good guidance - it short-circuits the \
student's reasoning.

4. **Actionability (AC)** - Is it clear from the response what the student \
should do next? A response can be technically correct but vague enough that the \
student has no idea how to proceed; that is *not* actionable.

# The label space

For each dimension, pick exactly one of:
- "Yes" - the response clearly meets the criterion
- "To some extent" - the response partially meets it (vague, hedged, partial)
- "No" - the response does not meet it

# Dialogue

{history}

# Candidate tutor response

{response}

# Your task

Grade the candidate tutor response on the four dimensions. Output a single \
JSON object with the keys "MI", "ML", "PG", "AC" and the corresponding label \
strings. Do not output anything else.

JSON:"""


def make_prompt(history: str, response: str) -> str:
    return JUDGE_PROMPT.format(history=history.strip(), response=response.strip())


def parse_response(raw: str) -> dict[str, str] | None:
    import json as _json
    import re

    valid = {"Yes", "To some extent", "No"}

    raw = raw.strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?|\n?```$", "", raw.strip(), flags=re.MULTILINE).strip()

    match = re.search(r"\{[^{}]*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = _json.loads(match.group(0))
    except _json.JSONDecodeError:
        return None

    out = {}
    for key in ("MI", "ML", "PG", "AC"):
        val = obj.get(key)
        if not isinstance(val, str):
            return None
        
        norm = val.strip().rstrip(".")
        if norm.lower() == "to some extent":
            norm = "To some extent"
        if norm not in valid:
            return None
        out[key] = norm
    return out


if __name__ == "__main__":
    # Demo: building a prompt and show what the parser does
    demo_history = """Tutor: What is 3 times 4?
Student: 3 times 4 is 7, because 3 + 4 = 7."""
    demo_response = "Hmm, let's revisit that. You're adding 3 and 4 there, but the question asks you to multiply. What does multiplication mean again?"
    print(make_prompt(demo_history, demo_response))
    print("\n---\n")
    # And demonstrate parsing
    fake_output = '''{"MI": "Yes", "ML": "Yes", "PG": "To some extent", "AC": "Yes"}'''
    print("Parsed:", parse_response(fake_output))
