#!/usr/bin/env python3
"""convert_to_unified_jsonl.py
Scan folders for .json/.jsonl files, normalize varying schemas and
write a unified JSONL with fields: instruction, input, output.

Usage:
  python convert_to_unified_jsonl.py --input_dirs GeoCode-PT GeoCode-Eval --output unified.jsonl

This helps when files have mixed schemas (instruction/output vs conversations etc.).
"""
import argparse
import glob
import json
import os
from typing import Any, Dict, Iterable


def iter_json_items(path: str) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a .json or .jsonl file.
    If .json contains a list, yield each item; if dict, yield it.
    """
    if path.lower().endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    else:
        with open(path, 'r', encoding='utf-8') as fh:
            try:
                obj = json.load(fh)
            except Exception:
                return
            if isinstance(obj, list):
                for item in obj:
                    yield item
            elif isinstance(obj, dict):
                yield obj


def normalize_item(item: Dict[str, Any]) -> Dict[str, str]:
    """Return dict with keys instruction, input, output (strings).
    Heuristics applied for common schemas.
    """
    def get_first(d, keys):
        for k in keys:
            if k in d and d[k] not in (None, ""):
                return d[k]
        return None

    # 1) Direct alpaca-like
    instr = get_first(item, ['instruction', 'prompt', 'question'])
    inp = get_first(item, ['input', 'context']) or ""
    out = get_first(item, ['output', 'response', 'answer', 'completion'])
    if instr is not None or out is not None:
        return {
            'instruction': str(instr or '').strip(),
            'input': str(inp or '').strip(),
            'output': str(out or '').strip()
        }

    # 2) Description/Code mapping
    desc = get_first(item, ['Description', 'description', 'title', 'Title'])
    code = get_first(item, ['Code', 'code'])
    if desc is not None or code is not None:
        return {
            'instruction': str(desc or '').strip(),
            'input': '',
            'output': str(code or '').strip()
        }

    # 3) conversation/chat-style: look for 'conversations' or 'dialog' lists
    conv = get_first(item, ['conversations', 'convo', 'dialogue', 'messages'])
    if isinstance(conv, list):
        # Find last assistant response and preceding user message
        last_user = None
        last_assistant = None
        for m in conv:
            if not isinstance(m, dict):
                continue
            frm = m.get('from') or m.get('role') or m.get('speaker') or ''
            val = m.get('value') or m.get('text') or m.get('message') or ''
            if isinstance(frm, str):
                role = frm.lower()
            else:
                role = ''
            if role in ('user', 'human', 'human_user', 'human:', 'human') or 'user' in role:
                last_user = val
            else:
                # treat anything else as assistant
                last_assistant = val
        return {
            'instruction': str(last_user or '').strip(),
            'input': '',
            'output': str(last_assistant or '').strip()
        }

    # 4) Fallback: try some generic fields
    any_text = get_first(item, ['text', 'body', 'content'])
    if any_text is not None:
        return {'instruction': str(any_text).strip(), 'input': '', 'output': ''}

    # 5) Last resort: dump the whole item as instruction
    return {'instruction': json.dumps(item, ensure_ascii=False)[:1000], 'input': '', 'output': ''}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', nargs='+', default=['data/datasets/GeoCode-PT', 'data/datasets/GeoCode-Eval'], help='Folders to scan')
    parser.add_argument('--output', default='data/unified.jsonl', help='Output JSONL file')
    args = parser.parse_args()

    files = []
    for d in args.input_dirs:
        files.extend(glob.glob(os.path.join(d, '**', '*.json'), recursive=True))
        files.extend(glob.glob(os.path.join(d, '**', '*.jsonl'), recursive=True))

    print(f'Found {len(files)} files')
    counts = {'written': 0, 'skipped': 0}

    with open(args.output, 'w', encoding='utf-8') as outfh:
        for fpath in files:
            for item in iter_json_items(fpath):
                norm = normalize_item(item)
                # optionally skip completely empty
                if not (norm.get('instruction') or norm.get('output')):
                    counts['skipped'] += 1
                    continue
                outfh.write(json.dumps(norm, ensure_ascii=False) + '\n')
                counts['written'] += 1

    print('Done. Written:', counts['written'], 'Skipped:', counts['skipped'])


if __name__ == '__main__':
    main()
