# Module 01: Document Loading & Chunking

## Concept Introduction

The first step in any RAG pipeline is getting your documents into a format a language model can work with. Raw documents — PDFs, Word files, web pages — contain unstructured text that needs to be loaded, cleaned, and cut into manageable pieces before any AI can process them.

**Document loading** is the act of reading a file and extracting its raw text. LangChain provides loader classes for many formats: `PyMuPDFLoader` for PDFs, `TextLoader` for plain text, `Docx2txtLoader` for Word documents. Each loader has the same interface: you instantiate it with a file path and call `.load()`, which returns a list of `Document` objects — one per page or section.

**Text cleaning** removes noise that would confuse downstream steps. PDF text extraction often contains stray newlines mid-sentence (because the PDF renderer inserts a newline at every line break on the page). Replacing `"\n"` with `" "` and collapsing runs of whitespace with a regex are standard pre-processing steps.

**Chunking** (also called *text splitting*) breaks a long document into smaller, overlapping pieces. Language models and embedding models both have limited context windows — you cannot feed an entire medical PDF into a model at once. The `RecursiveCharacterTextSplitter` tries to split on natural boundaries (paragraphs → sentences → words → characters) so that chunks remain semantically coherent. The `chunk_overlap` parameter ensures consecutive chunks share some text, reducing the chance that a relevant sentence gets cut in half at a boundary.

## Key Terms

- **Document loader** — a class that reads a file format and returns `Document` objects
- **`page_content`** — the string attribute on a LangChain `Document` that holds the extracted text
- **chunk** — a short, fixed-size slice of the document text
- **`chunk_size`** — the maximum number of characters per chunk
- **`chunk_overlap`** — how many characters the end of one chunk shares with the start of the next
- **`RecursiveCharacterTextSplitter`** — LangChain splitter that tries delimiters from largest to smallest (`\n\n` → `\n` → space → character)

---

## Exercises

Open `exercise.py`, fill in every `_____`, then run:

```bash
python exercise.py        # Windows
python3 exercise.py       # macOS / Linux
```

---

### Exercise 1 — Instantiate the PDF loader

```python
loader = PyMuPDFLoader(_____)
```

<details>
<summary>Hint</summary>

`PyMuPDFLoader` takes exactly one argument: the path to the PDF file as a string.

</details>

<details>
<summary>Answer</summary>

```python
loader = PyMuPDFLoader("docs/lipitor.pdf")
```

Any valid path string works. The exercise scaffold uses `"docs/lipitor.pdf"` to match the original project.

</details>

---

### Exercise 2 — Extract text from a loaded page

```python
for page in document:
    text += page._____
```

<details>
<summary>Hint</summary>

A LangChain `Document` object stores the page text in an attribute called `page_content`. It is a plain string.

</details>

<details>
<summary>Answer</summary>

```python
for page in document:
    text += page.page_content
```

</details>

---

### Exercise 3 — Replace newline characters

```python
text = text.replace(_____, _____)
```

<details>
<summary>Hint</summary>

PDF extractors insert a literal newline character (`"\n"`) at every visual line break on the page, even inside a sentence. Replace each one with a space.

</details>

<details>
<summary>Answer</summary>

```python
text = text.replace("\n", " ")
```

</details>

---

### Exercise 4 — Collapse multiple spaces

```python
text = re.sub(_____, " ", text)
```

<details>
<summary>Hint</summary>

After replacing newlines you may have runs of multiple spaces. The regex pattern `r'\s+'` matches one or more whitespace characters.

</details>

<details>
<summary>Answer</summary>

```python
text = re.sub(r'\s+', " ", text)
```

</details>

---

### Exercise 5 — Configure the text splitter

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=_____,
    chunk_overlap=_____,
    length_function=len,
    separators=_____,
)
```

<details>
<summary>Hint</summary>

The original project uses chunks of 1 000 characters with 200 characters of overlap. For separators, think: paragraph break first, then line break, then word boundary, then individual characters.

</details>

<details>
<summary>Answer</summary>

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)
```

The splitter tries each separator in order. It only moves to a finer split (e.g., word boundaries) if the coarser split (e.g., paragraph breaks) produces a chunk that is still too large.

</details>

---

### Exercise 6 — Split the text into chunks

```python
chunks = splitter._____(text)
```

<details>
<summary>Hint</summary>

`RecursiveCharacterTextSplitter` has a method that takes a single string and returns a list of strings.

</details>

<details>
<summary>Answer</summary>

```python
chunks = splitter.split_text(text)
```

</details>

---

## Further Reading

- [LangChain Text Splitters docs](https://python.langchain.com/docs/concepts/text_splitters/) — complete reference for all splitter types
- [LangChain Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/) — loaders for PDFs, HTML, CSV, Notion, Google Drive, and more
- [PyMuPDF documentation](https://pymupdf.readthedocs.io/) — the underlying library used by `PyMuPDFLoader`
