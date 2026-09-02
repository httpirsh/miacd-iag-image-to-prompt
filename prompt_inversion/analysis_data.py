from .targets import safe_stem

# Manual visual analysis for the six provided target images.
# Intentionally descriptive rather than "final prompt" text: candidate
# generation combines these visual ingredients into prompt variants.
ANALYSIS_BY_IMAGE_ID = {
    # 1159_3: anime/cyber fantasy warrior with elemental fire and spirit forms
    "1159_3": {
        "subject": "a serious blond male anime warrior wearing futuristic silver armor, holding a glowing orange fire sword",
        "composition": "centered upper-body character portrait, frontal view, symmetrical heroic pose, character fills most of the frame",
        "background": "dark smoky gray background with large abstract elemental spirit shapes behind the character",
        "style": [
            "anime fantasy character illustration",
            "digital concept art",
            "painterly illustration",
            "highly detailed character design",
        ],
        "lighting": [
            "dramatic low-key lighting",
            "glowing orange light from the sword",
            "soft rim light",
            "moody shadows",
        ],
        "camera": [
            "medium close-up portrait",
            "sharp focus on the face",
            "centered composition",
        ],
        "details": "blond messy hair, intense expression, silver shoulder armor, teal ghostly smoke on the left, orange fire spirit on the right, glowing curved flame blade crossing the lower body",
    },

    # 1159_7: spiky hedgehog-like creature emerging from a wooden cube
    "1159_7": {
        "subject": "a small hedgehog-like creature with very long spiky orange fur emerging from the top of a carved wooden cube",
        "composition": "square centered composition, cube viewed from slightly above, creature face centered near the top front edge",
        "background": "plain warm brown studio background and matching brown floor, minimal environment",
        "style": [
            "surreal realistic creature render",
            "highly detailed 3D render",
            "soft studio product render",
            "fantasy animal sculpture",
        ],
        "lighting": [
            "warm soft studio lighting",
            "gentle shadows",
            "diffuse ambient light",
        ],
        "camera": [
            "close-up",
            "slightly elevated viewpoint",
            "shallow depth of field",
        ],
        "details": "orange spiky fur radiating outward like a sunburst, tiny dark eyes, small snout, pale carved wooden cube with rough vertical grooves and blocky texture",
    },

    # 1159_25: orange juice still life
    "1159_25": {
        "subject": "a clear glass filled with fresh orange juice decorated with orange slices and pineapple pieces",
        "composition": "still life product shot, glass placed slightly right of center, tabletop view with fruit arranged around the glass",
        "background": "warm brown tabletop with a round wooden board on the left and soft neutral background",
        "style": [
            "realistic food photography",
            "commercial drink photography",
            "clean still life product photo",
            "highly detailed realistic render",
        ],
        "lighting": [
            "warm studio lighting",
            "soft shadows",
            "gentle highlights on the glass",
        ],
        "camera": [
            "close-up tabletop shot",
            "slightly elevated camera angle",
            "sharp focus on the glass and fruit",
        ],
        "details": "bright orange slices on the rim and around the table, orange halves in the background, small pineapple cubes scattered on the left foreground, transparent glass rim, saturated orange juice",
    },

    # 1159_29: palm tree at ocean sunset
    "1159_29": {
        "subject": "a tall palm tree standing in shallow ocean water during sunset",
        "composition": "wide tropical seascape, palm tree on the left foreground, horizon across the middle, sun low near the center, distant island on the right",
        "background": "turquoise ocean waves, cloudy pastel sky, distant mountains or island on the horizon",
        "style": [
            "realistic tropical landscape",
            "cinematic seascape",
            "detailed digital painting",
            "dreamlike realistic environment art",
        ],
        "lighting": [
            "soft sunset lighting",
            "warm sunlight reflected on the water",
            "gentle atmospheric haze",
        ],
        "camera": [
            "wide-angle view",
            "low viewpoint near the water",
            "deep depth of field",
        ],
        "details": "large green palm fronds, textured palm trunk, foamy waves around rocks at the base, reflective water surface, blue-green sea, orange sun glow",
    },

    # 7836: astronaut under huge planets in deep space
    "7836": {
        "subject": "a lone astronaut standing on a dark alien planet looking up at enormous planets and a star-filled galaxy",
        "composition": "small astronaut centered near the bottom, huge diagonal planet dominating the upper half, dramatic vast scale",
        "background": "deep outer space filled with stars, blue nebula clouds, massive planets and curved planetary horizon",
        "style": [
            "cinematic science fiction concept art",
            "epic space illustration",
            "realistic sci-fi environment",
            "highly detailed digital painting",
        ],
        "lighting": [
            "dramatic cosmic backlighting",
            "cool blue nebula glow",
            "bright horizon rim light",
            "high contrast shadows",
        ],
        "camera": [
            "wide-angle cinematic shot",
            "low viewpoint behind the astronaut",
            "deep depth of field",
        ],
        "details": "white astronaut suit, tiny human silhouette, cracked rocky alien ground, enormous reddish planet crossing diagonally above, blue star clusters and dark space atmosphere",
    },

    # 9338: colorful fantasy dragon hamster
    "9338": {
        "subject": "a cute small hamster-like fantasy creature with colorful dragon scales, horns and tiny wings",
        "composition": "centered character portrait, creature sitting upright, three-quarter side view facing left, body fills most of the frame",
        "background": "abstract painterly background with swirling colorful smoke and flame-like shapes",
        "style": [
            "fantasy creature illustration",
            "colorful digital painting",
            "painterly concept art",
            "cute detailed character design",
        ],
        "lighting": [
            "soft magical lighting",
            "colorful glowing ambient light",
            "warm highlights",
        ],
        "camera": [
            "close-up portrait",
            "shallow depth of field",
            "sharp focus on the creature's face",
        ],
        "details": "large glossy black eye, orange furry face, white belly, blue and red scales, small horns, tiny wings, rainbow colors, yellow and green swirling flame shapes behind the creature",
    },
}


def build_target_analysis(target_images, vlm_captions, manual_analysis=ANALYSIS_BY_IMAGE_ID):
    """Merge the manual per-target analysis with generated VLM captions."""
    target_analysis = {}
    missing = []

    for path in target_images:
        key = safe_stem(path)
        if key in manual_analysis:
            analysis = manual_analysis[key].copy()
            analysis["vlm_caption"] = vlm_captions.get(key, "")
            target_analysis[key] = analysis
        else:
            missing.append(key)

    if missing:
        raise KeyError(
            "Missing manual analysis for these targets: "
            + ", ".join(missing)
            + ". Make sure ANALYSIS_BY_IMAGE_ID keys match safe_stem(path)."
        )

    return target_analysis
