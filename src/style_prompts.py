import json
from typing import Dict, Any

STYLE_GUIDES = {
    "formal": {
        "role": "Professional visual description",
        "definition": "Neutral, objective, factual, polished — like a documentary narrator or museum curator. Describe only observable visual information.",
        "rules": [
            "Use a professional, objective tone.",
            "Write in the third-person.",
            "Use complete, grammatically correct sentences.",
            "Do not use jokes, sarcasm, or slang.",
            "Do not include emojis, hashtags, or exclamation marks.",
            "Do not make assumptions or interpret intentions beyond what is visible.",
            "Describe the overall video clip sequence rather than describing individual frames."
        ],
        "examples": [
            "A small orange tabby kitten walks slowly through the green garden foliage on a dirt path.",
            "Vehicles traverse a multi-lane city boulevard at twilight, flanked by high-rise buildings and autumn trees.",
            "A person wearing a striped apron dices fresh zucchini on a wooden cutting board with a chef's knife.",
            "An aerial shot captures a forested mountain ridge with granite formations and a city skyline on the horizon.",
            "A young woman sits in a modern open-plan office, typing on a keyboard at her desk."
        ]
    },
    "sarcastic": {
        "role": "Dry, deadpan internet sarcasm",
        "definition": "Subtle clever sarcasm remaining factually accurate. Humor comes from ironic observation, not insults or cruelty.",
        "rules": [
            "Maintain strict factual accuracy regarding the scene.",
            "Include exactly one sarcastic or dry observation.",
            "Never invent events, objects, or actions that are not present.",
            "Never insult or mock people in the video; keep humor light and witty.",
            "Do not use profanity, emojis, hashtags, or exclamation marks."
        ],
        "examples": [
            "Another day, another time-lapse of city traffic, as if we have never seen cars traveling on a road before.",
            "The mountains put on a grand display of majestic peaks, while the city below desperately waves for attention through the haze.",
            "A tiny orange kitten paces slowly through a backyard bush, demonstrating that it is clearly the ruler of this patch of grass.",
            "A chef dices zucchini with extreme focus, as if each vegetable slice represents a major milestone in global history.",
            "A person sits at a clean office desk typing, successfully making an email exchange look like a dramatic battle."
        ]
    },
    "humorous_tech": {
        "role": "Technology and programming humor",
        "definition": "Describe the actual scene while making the joke using software, programming, AI, or engineering references that naturally fit the visual content.",
        "rules": [
            "Always describe the visible scene and map it to a tech metaphor.",
            "Use programming, software development, database, hardware, or AI analogies.",
            "Avoid listing random buzzwords; the analogy must fit the visual action logically.",
            "Keep the joke understandable to anyone with basic tech knowledge.",
            "Do NOT force a 'When you...' template; vary sentence structure naturally."
        ],
        "examples": [
            "The orange kitten successfully updated its pathfinding algorithm to navigate the garden obstacles.",
            "The traffic database on this city boulevard is experiencing high concurrent write operations at twilight.",
            "A kitchen developer performs structured data slicing on zucchini arrays with clean execution.",
            "The drone camera successfully initialized its mountain scanning routine while the distant city servers remained idle.",
            "The developer's desktop workspace is currently struggling with physical cable routing issues."
        ]
    },
    "humorous_non_tech": {
        "role": "Relatable observational comedy",
        "definition": "Lighthearted everyday humor, zero technical jargon — like a family-friendly stand-up comedian's quick observation.",
        "rules": [
            "Do not use technical or programming references.",
            "Keep observations playful, natural, and relatable.",
            "Ensure humor is family-friendly and positive.",
            "Never invent events or details not present in the clip.",
            "Base every joke on the visible actions and objects in the scene.",
            "Vary sentence structure naturally; do not lock yourself into a single template format."
        ],
        "examples": [
            "That tiny orange kitten strolls through the backyard like it is personally inspecting the landscaping work.",
            "Thousands of commuters cross paths at this junction, yet everyone successfully avoids eye contact.",
            "Someone is taking zucchini chopping way too seriously for a simple Tuesday night dinner.",
            "A hiker climbs all the way to a remote peak just to realize they can still see their office building on the horizon.",
            "Staring intensely at the monitor as if the spreadsheet will self-assemble if looked at long enough."
        ]
    }
}

def get_style_prompts_system_prompt() -> str:
    """
    Constructs the consolidated style guide system prompt containing explicit instructions,
    rules, and calibration examples for all four tones.
    """
    prompt_parts = []
    
    # 1. Main Role & Structured Output Instructions
    prompt_parts.append(
        "You are an expert AI captioning engine. You are given structured details describing a video scene. "
        "Your task is to generate exactly four distinct captions, each written in a different style tone. "
        "You MUST return your response ONLY as a single valid JSON object. No markdown block formatting, no explanations, no preamble.\n"
        "Output JSON Schema:\n"
        "{\n"
        '  "formal": "string",\n'
        '  "sarcastic": "string",\n'
        '  "humorous_tech": "string",\n'
        '  "humorous_non_tech": "string"\n'
        "}"
    )

    # 2. explicit visual grounding rules
    prompt_parts.append(
        "VISUAL GROUNDING RULES:\n"
        "- Describe ONLY what is literally visible or direct facts of the scene.\n"
        "- Never invent dialogue, character names, specific locations (unless explicitly mentioned in the text context), brands, text on signs, or events outside the clip.\n"
        "- Do not make up emotions or sounds that are not visually or textually supported."
    )

    # 3. explicit style consistency rules
    prompt_parts.append(
        "STYLE & CONSISTENCY RULES:\n"
        "- Every caption must describe the SAME video clip context — only the stylistic presentation changes.\n"
        "- Avoid repeating the exact same sentence structures or copying identical phrases across styles."
    )

    # 4. Length constraints
    prompt_parts.append(
        "LENGTH & FORMAT CONSTRAINTS:\n"
        "- Each caption must be between 12 and 35 words long.\n"
        "- Write exactly 1 to 2 sentences per caption. Keep it concise."
    )

    # 5. Tone Specific Guides and Calibration Examples
    prompt_parts.append("TONE STYLE GUIDES & CALIBRATION EXAMPLES:")
    
    for tone_name, guide in STYLE_GUIDES.items():
        rules_bulleted = "\n".join(f"  - {r}" for r in guide["rules"])
        examples_bulleted = "\n".join(f"  * \"{ex}\"" for ex in guide["examples"])
        
        prompt_parts.append(
            f"STYLE: {tone_name.upper()}\n"
            f"Role: {guide['role']}\n"
            f"Definition: {guide['definition']}\n"
            f"Rules:\n{rules_bulleted}\n"
            f"Calibration Examples:\n{examples_bulleted}\n"
        )

    return "\n\n".join(prompt_parts)
