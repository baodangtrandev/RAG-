from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pdf_to_markdown(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        print(
            "Missing dependency: pypdf. Install it with: pip install pypdf",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    reader = PdfReader(str(pdf_path))
    sections: list[str] = [f"# {pdf_path.stem}"]

    for page_index, page in enumerate(reader.pages, start=1):
        page_text = _clean_text(page.extract_text() or "")
        if not page_text:
            continue
        sections.append(f"## Page {page_index}\n\n{page_text}")

    if len(sections) == 1:
        sections.append("_No extractable text found in this PDF._")

    return "\n\n".join(sections).strip() + "\n"


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    recursive: bool = False,
    overwrite: bool = False,
) -> int:
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory was not found: {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.rglob("*.pdf") if recursive else input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {input_dir}")
        return 0

    converted = 0
    skipped = 0

    for pdf_file in pdf_files:
        out_file = output_dir / f"{pdf_file.stem}.md"
        if out_file.exists() and not overwrite:
            skipped += 1
            print(f"Skipped (already exists): {out_file}")
            continue

        markdown = _pdf_to_markdown(pdf_file)
        out_file.write_text(markdown, encoding="utf-8")
        converted += 1
        print(f"Converted: {pdf_file} -> {out_file}")

    print(f"Finished. Converted: {converted}, Skipped: {skipped}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert all PDF files in a folder to Markdown files."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("documents"),
        help="Input directory containing PDF files (default: documents).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory for Markdown files (default: documents/paper-md).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search for PDF files in subdirectories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Markdown files.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = args.output or args.input / "paper-md"
    return convert_directory(
        input_dir=args.input,
        output_dir=output_dir,
        recursive=args.recursive,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
