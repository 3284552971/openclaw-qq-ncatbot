#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def extract_pdf_text(path_obj: Path, max_pages: int, max_chars: int):
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError("pdf_not_found")
    if path_obj.suffix.lower() != ".pdf":
        raise ValueError("not_pdf")

    reader = None
    lib_name = ""
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path_obj))
        lib_name = "pypdf"
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            reader = PdfReader(str(path_obj))
            lib_name = "PyPDF2"
        except Exception as exc:
            raise RuntimeError("pdf_library_missing: install pypdf or PyPDF2 in openclaw env") from exc

    pages = getattr(reader, "pages", []) or []
    total_pages = len(pages)
    use_pages = min(max_pages, total_pages)
    chunks = []
    for idx in range(use_pages):
        page = pages[idx]
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if not text:
            continue
        chunks.append(f"\n\n[Page {idx + 1}]\n{text}")
        if sum(len(x) for x in chunks) >= max_chars:
            break

    out = "".join(chunks).strip()
    truncated = len(out) > max_chars
    if truncated:
        out = out[:max_chars]
    return {
        "ok": True,
        "path": str(path_obj.resolve()),
        "library": lib_name,
        "total_pages": total_pages,
        "used_pages": use_pages,
        "truncated": truncated,
        "text": out,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDF.")
    parser.add_argument("--path", required=True, help="PDF file path")
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=40000)
    args = parser.parse_args()

    max_pages = max(1, min(int(args.max_pages), 500))
    max_chars = max(500, min(int(args.max_chars), 2_000_000))
    try:
        result = extract_pdf_text(Path(args.path).expanduser().resolve(), max_pages, max_chars)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
