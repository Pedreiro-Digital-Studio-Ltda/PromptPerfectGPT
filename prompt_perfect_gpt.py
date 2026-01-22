import os
import json
from typing import Dict, Any

from openai import OpenAI


# =========================================================
#  PromptPerfectGPT Node
#  - Generates a "final prompt" using ChatGPT
#  - Returns: (prompt_text, negative_text, debug_json)
# =========================================================

def _safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _build_system_prompt() -> str:
    return (
        "You are a senior Prompt Engineer and Advertising Art Director specialized in Qwen Image 2511.\n"
        "Goal: output a single, production-ready, hyper-realistic image prompt for real-world advertising photography.\n"
        "\n"
        "OUTPUT FORMAT (MANDATORY)\n"
        "- Output MUST be valid JSON with keys: prompt, negative, notes.\n"
        "- prompt: ONE single paragraph, English only, no emojis, no bullet points.\n"
        "- negative: ONE single paragraph, comma-separated negatives.\n"
        "- notes: short, practical, may include chosen conservative defaults.\n"
        "\n"
        "CORE PRINCIPLES\n"
        "1) Hyper-real photographic realism ONLY:\n"
        "- The result must look like a real professional photograph, not CGI, not illustration, not 3D render.\n"
        "- Use photography language: real materials, plausible lighting, correct scale, natural shadows, real optics.\n"
        "\n"
        "2) One image, one moment:\n"
        "- Describe a single scene, single setting, single time.\n"
        "- No multiple scenes, no transitions, no sequences.\n"
        "\n"
        "3) Product identity must be preserved:\n"
        "- Treat the product as immutable.\n"
        "- Do NOT change brand marks, text, label, logo, typography, colors, proportions, surface finish, or design.\n"
        "- Do NOT invent new graphics or alter packaging.\n"
        "- Keep correct real-world scale. No oversized/miniature product unless user explicitly states.\n"
        "- If the user mentions “use reference product / same product”, enforce: same exact product appearance.\n"
        "\n"
        "4) Character identity must be preserved (when characters are provided):\n"
        "- Keep the same person: face structure, age range, skin tone, hairstyle, expression intent.\n"
        "- Natural interaction only; realistic body language; avoid uncanny faces.\n"
        "- Avoid glam/beauty retouch cues; keep authentic skin texture.\n"
        "\n"
        "5) Deterministic composition control:\n"
        "- Must explicitly fix: camera position, framing, lens (mm), perspective, and focus priority.\n"
        "- Must explicitly fix: lighting type (softbox/window/overcast), direction, intensity feel, shadow softness.\n"
        "- Must explicitly fix: environment boundaries (what appears and what does NOT).\n"
        "\n"
        "6) Conservative defaults (do NOT ask questions):\n"
        "- If camera is missing: choose 50mm, eye-level or product-level, tripod, natural perspective.\n"
        "- If lighting is missing: choose soft diffused key + subtle fill, realistic reflections, no harsh stylization.\n"
        "- If composition is missing: product centered/hero, sharp product details, background secondary.\n"
        "- If environment is missing: choose clean, plausible real location consistent with a product shoot.\n"
        "\n"
        "PROMPT CONSTRUCTION (REQUIRED ORDER)\n"
        "Write the prompt as ONE paragraph in this order:\n"
        "(A) Medium & intent: hyper-real professional product/lifestyle photograph.\n"
        "(B) Product: exact product description + key materials + immutable identity constraints.\n"
        "(C) Characters: who, what they are doing, natural interaction with product (if any).\n"
        "(D) Environment: set details, props (only necessary), cleanliness, realism.\n"
        "(E) Lighting: source, direction, softness, reflections, shadow behavior.\n"
        "(F) Camera: lens mm, framing, angle, distance feel, tripod/handheld, focus point.\n"
        "(G) Composition: hierarchy (product first), sharpness targets, background control.\n"
        "(H) Realism controls: true scale, no distortion, no stylization.\n"
        "\n"
        "NEGATIVE PROMPT RULES (MANDATORY)\n"
        "- Always include strong negatives to prevent: CGI/3D/illustration, text/logo changes, label typos,\n"
        "  brand/name changes, extra objects, warped geometry, wrong hands/faces, plastic skin, heavy retouch,\n"
        "  unrealistic reflections, over-sharpening artifacts, blur on product, random props, clutter.\n"
        "- Also block: anime, cartoon, painterly, low-res, noise, watermark, signature, frame, border.\n"
        "\n"
        "STRICTNESS\n"
        "- Use only the user-provided fields. Do not invent brand names, slogans, or readable text.\n"
        "- Do not contradict any constraint. Do not add unnecessary adjectives.\n"
        "- Prioritize repeatability and physical plausibility over creativity.\n"
    )


def _build_user_payload(fields: Dict[str, str]) -> str:
    # Campos esperados (você pode ampliar)
    # style, subject, environment, lighting, camera, product, characters, composition, constraints, model_hint
    return json.dumps(fields, ensure_ascii=False, indent=2)


class PromptPerfectGPT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style": ("STRING", {"default": "photoreal premium product photography", "multiline": False}),
                "subject": ("STRING", {"default": "main subject", "multiline": True}),
                "environment": ("STRING", {"default": "environment / set / context", "multiline": True}),
                "lighting": ("STRING", {"default": "lighting description", "multiline": True}),
                "camera": ("STRING", {"default": "camera + lens + framing", "multiline": True}),
                "composition": ("STRING", {"default": "composition + focus priorities", "multiline": True}),
                "constraints": ("STRING", {"default": "consistency constraints, scale, realism rules", "multiline": True}),
                "negatives_extra": ("STRING", {"default": "extra negatives (optional)", "multiline": True}),
            },
            "optional": {
                "model_hint": ("STRING", {"default": "qwen / flux / wan / sd", "multiline": False}),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "gpt-4.1-mini"}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative", "debug_json")

    FUNCTION = "run"
    CATEGORY = "Prompting"

    def run(
            self,
            style,
            subject,
            environment,
            lighting,
            camera,
            composition,
            constraints,
            negatives_extra,
            model_hint="",
            api_key="",
            model="gpt-4.1-mini",
            temperature=0.2
    ):
        key = _safe_str(api_key) or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not found. Provide api_key input or set env var OPENAI_API_KEY.")

        client = OpenAI(api_key=key)

        fields = {
            "style": _safe_str(style),
            "subject": _safe_str(subject),
            "environment": _safe_str(environment),
            "lighting": _safe_str(lighting),
            "camera": _safe_str(camera),
            "composition": _safe_str(composition),
            "constraints": _safe_str(constraints),
            "model_hint": _safe_str(model_hint),
            "negatives_extra": _safe_str(negatives_extra),
        }

        system_prompt = _build_system_prompt()
        user_payload = _build_user_payload(fields)

        resp = client.chat.completions.create(
            model=model,
            temperature=float(temperature),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
        )

        content = resp.choices[0].message.content.strip()

        # Parse JSON safely
        try:
            data = json.loads(content)
        except Exception:
            # fallback: return raw as prompt, minimal negatives
            return (content, _safe_str(negatives_extra), json.dumps({"raw": content}, ensure_ascii=False))

        prompt = _safe_str(data.get("prompt", ""))
        negative = _safe_str(data.get("negative", ""))
        notes = data.get("notes", "")

        # Merge extra negatives (if any)
        extra = _safe_str(negatives_extra)
        if extra:
            if negative:
                negative = f"{negative}, {extra}"
            else:
                negative = extra

        debug = json.dumps(
            {"fields": fields, "notes": notes, "model_used": model},
            ensure_ascii=False,
            indent=2
        )

        return (prompt, negative, debug)


NODE_CLASS_MAPPINGS = {
    "PromptPerfectGPT": PromptPerfectGPT
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptPerfectGPT": "Prompt Perfect (ChatGPT)"
}
