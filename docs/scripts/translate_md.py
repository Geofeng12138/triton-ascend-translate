#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the triton-ascend project.
#
"""
PO-based translation system for Chinese Markdown (docs/zh/) to English (docs/en/).

This script follows the gettext PO (Portable Object) translation workflow:
  1. Parse Chinese Markdown into translatable blocks (paragraphs, code blocks, etc.)
  2. Create/update .po files mapping Chinese source -> English translation
  3. Use DeepSeek API to translate NEW or CHANGED blocks only
  4. Rebuild English Markdown from translated PO entries
  5. Save PO files to docs/po/ for version control and future incremental updates

Key benefits over the previous direct-AI approach:
  - Incremental updates: only new/changed blocks are retranslated
  - Translation memory: unchanged blocks keep their existing translations
  - Version control: PO files can be tracked in git for history and review
  - Better diffs: changes are visible at the block level, not file level

Usage:
    # Translate specific files (paths relative to docs/zh/, no prefix needed)
    python translate_md.py --files "quick_start.md,FAQ.md"

    # Translate all changed/new files
    python translate_md.py --all
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

# --- Directory Layout ---
# docs/
#   zh/          Chinese source Markdown files
#   en/          English translated Markdown files (output)
#   po/          PO translation files (version-controlled)
#     zh-to-en/  One .po file per source .md file

ZH_DIR = Path("docs/zh")
EN_DIR = Path("docs/en")
PO_DIR = Path("docs/po") / "zh-to-en"

SYSTEM_PROMPT = ("You are a professional technical documentation translation expert, "
                 "proficient in Chinese-to-English technical document translation.")

# Prompt for translating individual paragraph blocks.
BLOCK_TRANSLATION_PROMPT = """Translate the following Chinese text fragment into English.

This fragment is part of a Markdown technical documentation file. Translate only the visible text content.

Rules:
1. Do NOT wrap the result in markdown code fences or quotes — return ONLY the translated text.
2. Use standard English technical terminology. Keep technical terms accurate and consistent.
3. For proper nouns (person names, company names, product names, contributor names), keep them as-is.
4. If the text contains inline code (`code`), leave the inline code content unchanged.
5. If any sentence is too difficult to translate, keep the original Chinese as-is rather than producing incorrect English.

Text to translate:
{content}"""


# ---------------------------------------------------------------------------
# PO file I/O helpers
# ---------------------------------------------------------------------------


def _escape_po(s: str) -> str:
    """Escape a string for use in a PO msgid/msgstr value (single-line)."""
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    return s


def _write_multiline_po_value(value: str, indent: str = "") -> list[str]:
    """Write a potentially multi-line string as PO-formatted quoted lines.

    In PO format, a string can span multiple lines:
        msgstr ""
        "line1"
        "line2"
    """
    parts = value.split('\n')
    lines = []
    for part in parts:
        lines.append(f'{indent}"{_escape_po(part)}\\n"')
    return lines if lines else [f'{indent}""']


def parse_po_file(filepath: Path) -> dict:
    """
    Parse a .po file and return a dict mapping content_hash -> entry dict.

    Entry dict format:
    {
        "msgid": str,         # Original Chinese text
        "msgstr": str,        # English translation (empty if untranslated)
        "translated": bool,   # Whether msgstr is non-empty
        "type": str,          # "text" or "code"
        "index": int,         # Position in document
        "hash": str,          # MD5 hash of msgid (same as dict key)
    }
    """
    entries = {}
    if not filepath.exists():
        return entries

    raw = filepath.read_text(encoding="utf-8")

    # Split into entry blocks (separated by blank lines)
    # We use a regex that matches from #. (comment) through msgstr "..."
    # for each entry.
    entry_pattern = re.compile(
        r'#\. type:\s*(?P<type>\S+)\s*\n'
        r'#:\s*\S+:(?P<index>\d+)\s*\n'
        r'msgid\s+"(?P<msgid>(?:[^"\\]|\\.)*)"\s*\n'
        r'msgstr\s+"(?P<msgstr>(?:[^"\\]|\\.)*)"\s*',
        re.MULTILINE,
    )

    for match in entry_pattern.finditer(raw):
        msgid_raw = match.group("msgid")
        msgstr_raw = match.group("msgstr")

        # Unescape
        msgid = msgid_raw.replace('\\"', '"').replace('\\\\', '\\')
        msgstr = msgstr_raw.replace('\\"', '"').replace('\\\\', '\\')

        entry_type = match.group("type")
        entry_index = int(match.group("index"))

        content_hash = hashlib.md5(msgid.encode("utf-8")).hexdigest()
        entries[content_hash] = {
            "msgid": msgid,
            "msgstr": msgstr,
            "translated": bool(msgstr and msgstr.strip()),
            "type": entry_type,
            "index": entry_index,
            "hash": content_hash,
        }

    return entries


def write_po_file(filepath: Path, entries: dict, source_file_rel: str):
    """Write entries dict to a .po file, sorted by index."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Sort by index
    sorted_entries = sorted(entries.values(), key=lambda e: e.get("index", 0))

    lines = []
    now_str = time.strftime("%Y-%m-%d %H:%M%z")

    # Header entry
    lines.append(f'# PO translation file for {source_file_rel}\n'
                 f'#\n'
                 f'msgid ""\n'
                 f'msgstr ""\n'
                 f'"Project-Id-Version: triton-ascend-docs\\n"\n'
                 f'"POT-Creation-Date: {now_str}\\n"\n'
                 f'"PO-Revision-Date: {now_str}\\n"\n'
                 f'"Last-Translator: Auto Translation (DeepSeek)\\n"\n'
                 f'"Language-Team: English\\n"\n'
                 f'"Language: en\\n"\n'
                 f'"MIME-Version: 1.0\\n"\n'
                 f'"Content-Type: text/plain; charset=UTF-8\\n"\n'
                 f'"Content-Transfer-Encoding: 8bit\\n"\n')

    for entry in sorted_entries:
        entry_type = entry.get("type", "text")
        entry_index = entry.get("index", 0)
        msgid = entry["msgid"]
        msgstr = entry.get("msgstr", "")

        # Comments
        lines.append(f'#. type: {entry_type}')
        lines.append(f'#: {source_file_rel}:{entry_index}')

        # msgid – must be written as a single-quoted string or multi-line
        if '\n' in msgid:
            lines.append('msgid ""')
            for part in msgid.split('\n'):
                lines.append(f'"{_escape_po(part)}\\n"')
        else:
            lines.append(f'msgid "{_escape_po(msgid)}"')

        # msgstr
        if '\n' in msgstr:
            lines.append('msgstr ""')
            for part in msgstr.split('\n'):
                lines.append(f'"{_escape_po(part)}\\n"')
        else:
            lines.append(f'msgstr "{_escape_po(msgstr)}"')

        lines.append('')  # blank line separator

    text = '\n'.join(lines)
    filepath.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Markdown block splitting
# ---------------------------------------------------------------------------


def _is_heading_line(line: str) -> bool:
    """Check if a line is a Markdown heading."""
    stripped = line.strip()
    return stripped.startswith('#') and (len(stripped) > 1 and stripped[1] in ' #')


def _is_code_fence_line(line: str) -> bool:
    """Check if a line is a Markdown code fence (``` or ~~~)."""
    stripped = line.strip()
    return stripped.startswith('```') or stripped.startswith('~~~')


def split_markdown_blocks(content: str) -> list[dict]:
    """
    Split Markdown content into translatable blocks.

    Block types:
      - "text":  Paragraph or heading text that needs translation
      - "code":  Code block (fenced) – kept as-is, not translated

    Rules:
      - Consecutive non-blank, non-code lines form a single "text" block.
      - Code blocks (``` ... ```) are preserved whole and marked as "code".
      - Blank lines separate blocks but are not themselves blocks.
      - Headings are grouped with their following paragraph into one text block.
    """
    blocks: list[dict] = []
    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Code block ──
        if _is_code_fence_line(line):
            code_lines = [line]
            i += 1
            while i < len(lines) and not _is_code_fence_line(lines[i]):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                code_lines.append(lines[i])  # include closing fence
                i += 1

            blocks.append({
                "content": '\n'.join(code_lines),
                "type": "code",
                "index": len(blocks),
            })
            continue

        # ── Blank line(s) – count them and preserve spacing ──
        if stripped == '':
            blank_count = 0
            while i < len(lines) and lines[i].strip() == '':
                blank_count += 1
                i += 1
            blocks.append({
                "content": '\n' * blank_count,
                "type": "blank",
                "index": len(blocks),
            })
            continue

        # ── Text block – collect until blank line or code fence ──
        text_lines = []
        while i < len(lines):
            cline = lines[i]
            if cline.strip() == '' or _is_code_fence_line(cline):
                break
            text_lines.append(cline)
            i += 1

        text = '\n'.join(text_lines)
        if text.strip():
            blocks.append({
                "content": text,
                "type": "text",
                "index": len(blocks),
            })

    return blocks


def rebuild_markdown(blocks: list[dict], po_entries: dict) -> str:
    """
    Rebuild English Markdown from the original block structure and PO entries.

    For each block:
      - If "code" or "blank": use original content as-is.
      - If "text" and hash found in po_entries with a translation: use msgstr.
      - If "text" but no translation available: fall back to original Chinese.
    """
    result_parts = []

    for block in blocks:
        block_type = block["type"]
        content = block["content"]

        if block_type in ("code", "blank"):
            result_parts.append(content)
            continue

        # text block – look up translation
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        entry = po_entries.get(content_hash)
        if entry and entry.get("translated"):
            result_parts.append(entry["msgstr"])
        else:
            # Fallback: keep original Chinese
            result_parts.append(content)

    return '\n'.join(result_parts)


# ---------------------------------------------------------------------------
# Translator class
# ---------------------------------------------------------------------------


class MarkdownTranslator:
    """PO-based Markdown translator using DeepSeek API."""

    def __init__(self, api_key: str, max_concurrent: int = 5):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        self.max_concurrent = max_concurrent

    def _po_path(self, zh_path: Path) -> Path:
        """Convert a docs/zh/... path to its docs/po/zh-to-en/...po counterpart."""
        rel = zh_path.relative_to(ZH_DIR)
        return PO_DIR / rel.with_suffix('.po')

    def _en_path(self, zh_path: Path) -> Path:
        """Convert a docs/zh/... path to its docs/en/... counterpart."""
        rel = zh_path.relative_to(ZH_DIR)
        # Strip _zh suffix from filename
        if rel.stem.endswith('_zh'):
            rel = rel.parent / (rel.stem[:-3] + rel.suffix)
        return EN_DIR / rel

    async def _translate_single(self, content: str, context: str = "") -> Optional[str]:
        """Translate a single text block via DeepSeek API."""
        prompt = BLOCK_TRANSLATION_PROMPT.format(content=content)
        system = SYSTEM_PROMPT
        if context:
            system = f"{SYSTEM_PROMPT} (File: {context})"

        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8192,
                temperature=0.3,
            )
            text = response.choices[0].message.content
            return text.strip() if text else None
        except Exception as e:
            print(f"API error: {e}")
            return None

    async def translate_file(self, zh_path: Path) -> bool:
        """Translate a single Markdown file using the PO workflow."""
        if not zh_path.exists() or zh_path.suffix != ".md":
            print(f"  Skip: {zh_path} (not found or not .md)")
            return False

        content = zh_path.read_text(encoding="utf-8")
        if not content.strip():
            print(f"  Skip: {zh_path.name} (empty)")
            return False

        en_path = self._en_path(zh_path)
        po_path = self._po_path(zh_path)
        source_rel = str(zh_path.as_posix())

        print(f"  {zh_path.name} → {en_path.name}", end=" ", flush=True)

        # Step 1: Split current Markdown into blocks
        blocks = split_markdown_blocks(content)
        print(f"[{len(blocks)} blocks]", end=" ", flush=True)

        # Step 2: Load existing PO entries
        po_entries = parse_po_file(po_path)

        # Step 3: Merge – build a new entries dict from current blocks,
        #          keeping existing translations where content hasn't changed.
        new_entries: dict[str, dict] = {}
        untranslated_text_blocks = []

        for block in blocks:
            block_hash = hashlib.md5(block["content"].encode("utf-8")).hexdigest()
            existing = po_entries.get(block_hash)

            if existing:
                # Exact match found in PO – keep existing translation
                new_entries[block_hash] = existing
            else:
                # New or changed content – create fresh entry, mark as untranslated
                entry = {
                    "msgid": block["content"],
                    "msgstr": "",
                    "translated": False,
                    "type": block["type"],
                    "index": block["index"],
                    "hash": block_hash,
                }
                new_entries[block_hash] = entry
                if block["type"] == "text":
                    untranslated_text_blocks.append((block_hash, block["content"]))

        kept = sum(1 for e in new_entries.values() if e.get("translated"))
        new_count = len(untranslated_text_blocks)
        print(f"[{kept} kept, {new_count} new]", end=" ", flush=True)

        # Step 4: Translate new/changed text blocks via API
        if untranslated_text_blocks:
            print(f"\n    Translating {len(untranslated_text_blocks)} block(s)...", end=" ", flush=True)
            for idx, (block_hash, text) in enumerate(untranslated_text_blocks):
                translation = await self._translate_single(text, str(zh_path.name))
                if translation:
                    new_entries[block_hash]["msgstr"] = translation
                    new_entries[block_hash]["translated"] = True
                # Brief rate-limit pause between calls
                if idx < len(untranslated_text_blocks) - 1:
                    await asyncio.sleep(0.3)
            print(f" done", end=" ", flush=True)

        # Step 5: Save PO file
        write_po_file(po_path, new_entries, source_rel)

        # Step 6: Rebuild and write English Markdown
        en_path.parent.mkdir(parents=True, exist_ok=True)
        en_content = rebuild_markdown(blocks, new_entries)
        en_path.write_text(en_content, encoding="utf-8")

        print("OK")
        return True

    async def translate_files(self, file_list: list[Path], output_json: str) -> int:
        """Translate a list of files sequentially and write a results JSON."""
        print(f"Translating {len(file_list)} file(s) using PO workflow")

        success_files = []
        for fp in file_list:
            ok = await self.translate_file(fp)
            if ok:
                success_files.append(str(self._en_path(fp)))

        print(f"\nResult: {len(success_files)}/{len(file_list)} translated")

        report = {
            "success_files": success_files,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": len(file_list),
            "success_count": len(success_files),
        }
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Results written to {output_json}")

        return 0 if success_files else 1

    @staticmethod
    def find_changed_zh_files() -> list[Path]:
        """Find docs/zh/ .md files that differ from their docs/en/ counterpart.

        Uses git diff if available, otherwise falls back to comparing zh and en
        file contents directly.
        """
        import subprocess

        # Try git diff first
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=ACMUX", "HEAD", "--", "docs/zh/"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                changed = []
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if line.endswith(".md"):
                        changed.append(Path(line))
                if changed:
                    return changed
        except Exception:
            pass

        # Fallback: compare zh vs en file contents
        def _to_en_path(zh_md: Path) -> Path:
            rel = zh_md.relative_to(Path("docs/zh"))
            if rel.stem.endswith("_zh"):
                rel = rel.parent / (rel.stem[:-3] + rel.suffix)
            return Path("docs/en") / rel

        changed = []
        for md_file in Path("docs/zh").rglob("*.md"):
            en_file = _to_en_path(md_file)
            if not en_file.exists():
                changed.append(md_file)
            elif md_file.read_text(encoding="utf-8") != en_file.read_text(encoding="utf-8"):
                changed.append(md_file)

        return changed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def write_empty_json(output_json: str, reason: str = ""):
    """Write an empty results JSON (used for error / no-work outcomes)."""
    report = {
        "success_files": [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": 0,
        "success_count": 0,
        "note": reason,
    }
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Empty result written to {output_json} (reason: {reason})")


async def async_main():
    parser = argparse.ArgumentParser(description="PO-based Markdown translation: docs/zh/ → docs/en/", )
    parser.add_argument(
        "--files",
        help="Comma-separated file paths (relative to docs/zh/)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Translate all changed/new .md files",
    )
    parser.add_argument(
        "--output-json",
        default=os.getenv("OUTPUT_JSON", "/tmp/translation_results.json"),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DEEPSEEK_API_KEY"),
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent API calls (default: 5)",
    )
    args = parser.parse_args()

    output_json = args.output_json

    # ── Validate API key ──
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        msg = "DEEPSEEK_API_KEY not set"
        print(f"Error: {msg}")
        write_empty_json(output_json, msg)
        return 1

    # ── Resolve file list ──
    if args.files:
        raw_files = [f.strip() for f in args.files.split(",") if f.strip()]
        file_list = []
        for f in raw_files:
            p = Path(f)
            if not p.exists():
                p = Path("docs/zh") / f
            file_list.append(p)
    elif args.all:
        file_list = MarkdownTranslator.find_changed_zh_files()
    else:
        msg = "specify --files or --all"
        print(f"Error: {msg}")
        write_empty_json(output_json, msg)
        return 1

    # ── Validate ──
    valid = [f for f in file_list if f.exists() and f.suffix == ".md"]
    skipped = [str(f) for f in file_list if f not in valid]
    if skipped:
        print(f"Skipped (not found or not .md): {skipped}")

    if not valid:
        print("No valid .md files to translate")
        write_empty_json(output_json, "no valid .md files to translate")
        return 0

    # ── Translate ──
    translator = MarkdownTranslator(
        api_key=api_key,
        max_concurrent=args.max_concurrent,
    )
    return await translator.translate_files(valid, output_json)


if __name__ == "__main__":
    sys.exit(asyncio.run(async_main()))