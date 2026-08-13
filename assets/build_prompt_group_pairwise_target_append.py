"""
Script to rebuild a prompt_group file for pairwise attacks that append suffixes to the target response.

This script loads an existing `prompt_group.pth` (or other .pth containing a list of suffix strings),
normalizes suffixes so they are safe to append to a response (guarantee a leading space, preserve markers
like [KEEPER]/[REPLACE]), and writes out a new file `prompt_group_pairwise_target_append.pth`.

Usage (from project root):
    python assets/build_prompt_group_pairwise_target_append.py

You can also provide an alternative input/output path by passing arguments when calling the script
(see variables at top of file).

The generated file is a plain Python-serialized object (torch.save) containing a list of suffix strings.
"""
import torch
from pathlib import Path

# Config - change if you want a different source/target
SRC_PATH = Path('assets/prompt_group.pth')
OUT_PATH = Path('assets/prompt_group_pairwise_target_append.pth')


def normalize_suffix_for_target_append(s: str) -> str:
    """Ensure the suffix is safe to append to a response/target.
    - guarantee a single leading space unless the suffix is empty
    - preserve special tokens like [REPLACE], [KEEPER]
    - strip trailing whitespace
    """
    if s is None:
        return ""
    s = str(s)
    s = s.rstrip()
    if s == "":
        return ""
    # If suffix already starts with punctuation (",", ".", etc.) we still want a space before
    if not s.startswith(' '):
        s = ' ' + s
    return s


def main():
    if not SRC_PATH.exists():
        print(f"Source file {SRC_PATH} not found. Please make sure prompt_group.pth exists in assets/")
        return

    try:
        reference = torch.load(SRC_PATH, map_location='cpu')
    except Exception as e:
        print(f"Failed to load {SRC_PATH}: {e}")
        return

    if not isinstance(reference, (list, tuple)):
        # If it's a dict with 'prompts' or similar, try to handle common shapes
        try:
            # fallback: try to coerce values to list
            reference = list(reference)
        except Exception:
            print("Unrecognized structure in prompt_group file; expected a list of suffix strings.")
            return

    new_list = [normalize_suffix_for_target_append(x) for x in reference]

    try:
        torch.save(new_list, OUT_PATH)
        print(f"Wrote {len(new_list)} suffixes to {OUT_PATH}")
    except Exception as e:
        print(f"Failed to write output file: {e}")


if __name__ == '__main__':
    main()
