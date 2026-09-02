"""Global visual modifier bank used by metric-guided refinement (mutate_prompt).

Shared vocabulary built from visual characteristics that appear across all
targets. The evaluation metrics, not manual selection, decide which
modifiers work best for each image.
"""

STYLE_MODIFIERS = [
    # Food / product / still life
    "realistic food photography",
    "commercial drink photography",
    "clean still life product photo",
    "warm studio product photography",

    # Anime / character / fantasy
    "anime fantasy character illustration",
    "digital concept art",
    "painterly illustration",
    "highly detailed character design",

    # Creature / 3D / surreal
    "surreal realistic creature render",
    "highly detailed 3D render",
    "soft studio product render",
    "fantasy animal sculpture",

    # Landscape / seascape
    "realistic tropical landscape",
    "cinematic seascape",
    "dreamlike realistic environment art",
    "detailed digital painting",

    # Sci-fi / space
    "cinematic science fiction concept art",
    "epic space illustration",
    "realistic sci-fi environment",
    "highly detailed digital painting",

    # Cute fantasy creature
    "fantasy creature illustration",
    "colorful digital painting",
    "painterly concept art",
    "cute detailed character design",
]


LIGHTING_MODIFIERS = [
    "warm studio lighting",
    "soft studio lighting",
    "gentle shadows",
    "soft shadows",
    "dramatic low-key lighting",
    "glowing orange light",
    "soft rim light",
    "moody shadows",
    "soft sunset lighting",
    "warm sunlight reflected on the water",
    "gentle atmospheric haze",
    "dramatic cosmic backlighting",
    "cool blue nebula glow",
    "bright horizon rim light",
    "soft magical lighting",
    "colorful glowing ambient light",
    "warm highlights",
]


REFINEMENT_SUFFIXES = [
    "stronger focus on the main subject",
    "cleaner background",
    "more faithful colors",
    "balanced lighting",
    "detailed foreground",
    "sharp focus",
    "high detail",
]

REFINEMENT_REPLACEMENTS = [
    ("soft studio lighting", "warm soft studio lighting"),
    ("soft lighting", "warm soft lighting"),
    ("dramatic lighting", "dramatic cinematic lighting"),
    ("close-up", "tight close-up"),
    ("shallow depth of field", "very shallow depth of field"),
    ("slightly blurred background", "softly blurred background"),
]
