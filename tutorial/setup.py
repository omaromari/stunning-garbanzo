# setup.py — verify your environment before starting the tutorial
# Usage:
#   python setup.py      (Windows)
#   python3 setup.py     (macOS / Linux)

import importlib.util
import sys

REQUIRED = [
    ("sentence_transformers", "sentence-transformers"),
    ("langchain",             "langchain"),
    ("pymupdf",               "pymupdf"),
    ("gpt4all",               "gpt4all"),
    ("streamlit",             "streamlit"),
    ("numpy",                 "numpy"),
]

print(f"\nPython {sys.version}\n")
print(f"{'Package':<28} Status")
print("-" * 45)

all_ok = True
for import_name, pip_name in REQUIRED:
    spec = importlib.util.find_spec(import_name)
    if spec is not None:
        print(f"  OK      {import_name}")
    else:
        print(f"  MISSING {import_name:<20}  -->  pip install {pip_name}")
        all_ok = False

print()
if all_ok:
    print("All dependencies found. You are ready to start the tutorial!")
else:
    print("Install the missing packages above, then run this script again.")
print()
