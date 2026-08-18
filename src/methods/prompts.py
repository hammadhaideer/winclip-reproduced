"""Compositional Prompt Ensemble for WinCLIP.

Following Jeong et al., CVPR 2023, Section 3.1: "Compositional Prompt Ensemble."
We generate prompts by crossing state words (normal/anomaly) with prompt templates,
yielding ~200 prompts per object class.

The state word lists and template list below are taken directly from the WinCLIP
paper's supplementary material.
"""

from typing import Dict, List


STATE_WORDS_NORMAL: List[str] = [
    "perfect",
    "unblemished",
    "flawless",
    "without flaw",
    "without defect",
    "without damage",
]


STATE_WORDS_ANOMALY: List[str] = [
    "damaged",
    "broken",
    "with flaw",
    "with defect",
    "with damage",
]


TEMPLATES: List[str] = [
    "a cropped photo of a {}.",
    "a cropped photo of the {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of a {}.",
    "a dark photo of the {}.",
    "a jpeg corrupted photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of the {} for visual inspection.",
    "a photo of a {} for visual inspection.",
    "a photo of the {} for anomaly detection.",
    "a photo of a {} for anomaly detection.",
]


def _phrase(state: str, object_name: str) -> str:
    if state.startswith("with") or state.startswith("without"):
        return f"{object_name} {state}"
    else:
        return f"{state} {object_name}"


def build_prompt_ensemble(
    object_name: str,
    state_words_normal: List[str] = None,
    state_words_anomaly: List[str] = None,
    templates: List[str] = None,
) -> Dict[str, List[str]]:
    if state_words_normal is None:
        state_words_normal = STATE_WORDS_NORMAL
    if state_words_anomaly is None:
        state_words_anomaly = STATE_WORDS_ANOMALY
    if templates is None:
        templates = TEMPLATES

    object_clean = object_name.replace("_", " ")

    normal_prompts = []
    for state in state_words_normal:
        phrase = _phrase(state, object_clean)
        for template in templates:
            normal_prompts.append(template.format(phrase))

    anomaly_prompts = []
    for state in state_words_anomaly:
        phrase = _phrase(state, object_clean)
        for template in templates:
            anomaly_prompts.append(template.format(phrase))

    return {
        "normal": normal_prompts,
        "anomaly": anomaly_prompts,
    }


if __name__ == "__main__":
    prompts = build_prompt_ensemble("metal nut")
    print(f"Normal prompts: {len(prompts['normal'])}")
    print(f"Anomaly prompts: {len(prompts['anomaly'])}")
    print()
    print("Sample normal prompts:")
    for p in prompts["normal"][:5]:
        print(f"  - {p}")
    print()
    print("Sample anomaly prompts:")
    for p in prompts["anomaly"][:5]:
        print(f"  - {p}")
