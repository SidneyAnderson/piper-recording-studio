"""CLI tool to generate phonetically diverse TTS training prompts.

Generates prompt text files compatible with Piper Recording Studio's
expected format (id<tab>text per line). Follows ElevenLabs best practices:
- Sentences between 10-30 words for optimal TTS quality
- Numbers written as words, abbreviations expanded
- Diverse phoneme coverage, prosodic patterns, and topics
- Mix of British and General English phrasing

Usage:
    python3 -m generate_prompts --count 11000 --language en-GB
    python3 -m generate_prompts --count 11000 --language en-GB --output prompts/
"""

import argparse
import hashlib
import random
import sys
from pathlib import Path

from .templates import GENERATORS


def validate_sentence(text: str) -> bool:
    """Check that a sentence meets quality criteria for TTS training."""
    words = text.split()
    # ElevenLabs recommends 10-30 words per sentence
    if len(words) < 6 or len(words) > 35:
        return False
    # Must end with sentence-ending punctuation
    if not text.rstrip().endswith((".", "!", "?")):
        return False
    # No stray special characters
    for char in ["&", "#", "@", "*", "<", ">", "{", "}", "[", "]"]:
        if char in text:
            return False
    return True


def generate_prompts(
    count: int,
    seed: int = 42,
    categories: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Generate unique prompts with category and phoneme diversity.

    Returns list of (id, text) tuples.
    """
    random.seed(seed)

    if categories is None:
        categories = list(GENERATORS.keys())

    prompts = []
    seen = set()
    attempts = 0
    max_attempts = count * 10  # Safety valve

    # Distribute evenly across categories, cycling through them
    cat_cycle = 0
    while len(prompts) < count and attempts < max_attempts:
        category = categories[cat_cycle % len(categories)]
        generator = GENERATORS.get(category)
        if generator is None:
            cat_cycle += 1
            continue

        attempts += 1
        try:
            text = generator()
        except (IndexError, KeyError):
            continue

        if not validate_sentence(text):
            continue

        # Deduplicate by normalized text
        normalized = text.lower().strip()
        if normalized in seen:
            continue
        seen.add(normalized)

        # Generate a stable ID based on content
        prompt_id = f"5{hashlib.md5(text.encode()).hexdigest()[:9]}"
        prompts.append((prompt_id, text))
        cat_cycle += 1

    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Generate phonetically diverse TTS training prompts"
    )
    parser.add_argument(
        "--count", type=int, default=11000,
        help="Number of prompts to generate (default: 11000)"
    )
    parser.add_argument(
        "--language", default="en-GB",
        help="Language code for output directory naming (default: en-GB)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: prompts/<Language>_<code>/)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--filename", default=None,
        help="Output filename without extension (default: auto-generated)"
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Specific categories to generate (default: all)",
        choices=list(GENERATORS.keys()),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts to stdout instead of writing to file"
    )
    args = parser.parse_args()

    print(f"Generating {args.count} prompts (seed={args.seed})...")
    prompts = generate_prompts(
        count=args.count,
        seed=args.seed,
        categories=args.categories,
    )
    print(f"Generated {len(prompts)} unique prompts across {len(GENERATORS)} categories.")

    if len(prompts) < args.count:
        print(f"WARNING: Could only generate {len(prompts)}/{args.count} unique prompts.")
        print("         Increase template variety or reduce count.")

    if args.dry_run:
        for prompt_id, text in prompts[:20]:
            print(f"{prompt_id}\t{text}")
        if len(prompts) > 20:
            print(f"... and {len(prompts) - 20} more")
        return

    # Determine output path
    if args.output:
        out_dir = Path(args.output)
    else:
        # Map language code to directory name
        lang_names = {
            "en-GB": "English (United Kingdom)",
            "en-US": "English (United States)",
        }
        lang_name = lang_names.get(args.language, f"Generated_{args.language}")
        out_dir = Path(__file__).resolve().parent.parent / "prompts" / f"{lang_name}_{args.language}"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    if args.filename:
        filename = f"{args.filename}.txt"
    else:
        start_id = prompts[0][0] if prompts else "0"
        end_id = prompts[-1][0] if prompts else "0"
        filename = f"{start_id}_{end_id}_Generated.txt"

    out_path = out_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        for prompt_id, text in prompts:
            f.write(f"{prompt_id}\t{text}\n")

    print(f"Written to: {out_path}")
    print(f"\nElevenLabs recommended settings for consistent training data:")
    print(f"  Stability:        0.75-0.85 (higher = more consistent)")
    print(f"  Similarity Boost: 0.80-1.00 (higher = closer to voice profile)")
    print(f"  Sample Rate:      24000 Hz")
    print(f"\nNext step: Generate audio via the web UI or CLI:")
    print(f"  python3 -m piper_recording_studio  # then visit /generate")
    print(f"  # or")
    print(f"  python3 -m elevenlabs_generate --voice-id YOUR_ID --language {args.language}")


if __name__ == "__main__":
    main()
