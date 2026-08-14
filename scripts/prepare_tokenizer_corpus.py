from pathlib import Path
import json
import re
import sys

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "tokenizer_measurement" / "corpus"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "english_udhr": "https://www.un.org/en/about-us/universal-declaration-of-human-rights",
    "amharic_wikipedia_ethiopia": "https://am.wikipedia.org/wiki/%E1%8A%A2%E1%89%B5%E1%8B%AE%E1%8C%B5%E1%8B%AB",
}


def fetch_main(url: str) -> str:
    response = requests.get(url, timeout=30, headers={"User-Agent": "Attention-tokenizer-measurement/1.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for node in soup(["script", "style", "noscript", "svg"]):
        node.decompose()
    main = soup.find("main") or soup.find(id="mw-content-text") or soup.body
    if main is None:
        raise RuntimeError(f"no main content found for {url}")
    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text + "\n"


def copy_file(name: str, source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    (OUT / name).write_text(text, encoding="utf-8")


def main() -> int:
    (OUT / "english_udhr.txt").write_text(fetch_main(SOURCES["english_udhr"]), encoding="utf-8")
    (OUT / "amharic_wikipedia_ethiopia.txt").write_text(fetch_main(SOURCES["amharic_wikipedia_ethiopia"]), encoding="utf-8")
    english = (OUT / "english_udhr.txt").read_text(encoding="utf-8")
    amharic = (OUT / "amharic_wikipedia_ethiopia.txt").read_text(encoding="utf-8")
    (OUT / "bilingual_udhr.txt").write_text(english + "\n" + amharic, encoding="utf-8")
    copy_file("code_cpp.txt", ROOT / "src" / "transformer_model.cpp")
    copy_file("structured_checkpoint_test.txt", ROOT / "tests" / "test_checkpoint.cpp")
    manifest = {
        "version": "attention.tokenizer_measurement.v1",
        "sources": SOURCES,
        "local_samples": {
            "code_cpp": "src/transformer_model.cpp",
            "structured_checkpoint_test": "tests/test_checkpoint.cpp",
        },
        "note": "Samples are for deterministic representation statistics, not representative language-quality evaluation.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
