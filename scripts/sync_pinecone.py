import os
import uuid
from pathlib import Path
import pinecone
from openai import OpenAI
from tree_sitter import Language, Parser
import tiktoken
import re
from tqdm import tqdm
import yaml
import json

# Constants
MAX_TOKENS = 8192
INDEX_NAME = "repo-chunks"
DIMENSION = 1536
METRIC = "cosine"
BATCH_SIZE = 20
TREE_SITTER_EXTS = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".cs", ".go", ".rb", ".php", ".rs", ".swift", ".kt", ".kts", ".scala", ".cjs", ".mjs", ".sh", ".html", ".css", ".scss", ".xml"}
LANGUAGES = {
    '.py': Language('/path/to/my-languages.so', 'python'),
    '.js': Language('/path/to/my-languages.so', 'javascript'),
    '.jsx': Language('/path/to/my-languages.so', 'javascript'),
    '.cjs': Language('/path/to/my-languages.so', 'javascript'),
    '.ts': Language('/path/to/my-languages.so', 'typescript'),
    '.java': Language('/path/to/my-languages.so', 'java'),
    '.cpp': Language('/path/to/my-languages.so', 'cpp'),
    '.c': Language('/path/to/my-languages.so', 'c'),
    '.cs': Language('/path/to/my-languages.so', 'c_sharp'),
    '.go': Language('/path/to/my-languages.so', 'go'),
    '.rb': Language('/path/to/my-languages.so', 'ruby'),
    '.php': Language('/path/to/my-languages.so', 'php'),
    '.rs': Language('/path/to/my-languages.so', 'rust'),
    '.swift': Language('/path/to/my-languages.so', 'swift'),
    '.html': Language('/path/to/my-languages.so', 'html'),
    '.css': Language('/path/to/my-languages.so', 'css'),
    '.xml': Language('/path/to/my-languages.so', 'xml'),
    '.scss': Language('/path/to/my-languages.so', 'css'),
    '.sh': Language('/path/to/my-languages.so', 'bash'),
}

# Initialize clients
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="us-west1-gcp")
index = pinecone.Index(INDEX_NAME)
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(tokenizer.encode(text, allowed_special="all"))

def re_chunk_if_oversize(sections, max_tokens=MAX_TOKENS):
    final_chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        tokens = count_tokens(section)
        if tokens <= max_tokens:
            final_chunks.append(section)
        else:
            split_points = re.split(r'(?<=[.!?])\s+', section)
            current_chunk = ""
            for part in split_points:
                if count_tokens(current_chunk + " " + part) <= max_tokens:
                    current_chunk += " " + part
                else:
                    final_chunks.append(current_chunk.strip())
                    current_chunk = part
            if current_chunk.strip():
                final_chunks.append(current_chunk.strip())
            really_final = []
            for chunk in final_chunks:
                if count_tokens(chunk) <= max_tokens:
                    really_final.append(chunk)
                else:
                    raw_splits = re.findall(r'.{1,3000}(?:\s+|$)', chunk)
                    really_final.extend([s.strip() for s in raw_splits if s.strip()])
            final_chunks = really_final
    return final_chunks

def chunk_code_tree_sitter(filepath, ext):
    parser = Parser()
    parser.set_language(LANGUAGES[ext])
    try:
        code = Path(filepath).read_text(encoding="utf-8")
        tree = parser.parse(code.encode("utf-8"))
        root = tree.root_node
        chunks = []
        for child in root.children:
            if child.type in ['function_definition', 'class_definition', 'function', 'method_definition', 'element', 'style_rule']:
                snippet = code[child.start_byte:child.end_byte]
                chunks.append(snippet.strip())
        return re_chunk_if_oversize(chunks if chunks else [code.strip()[:1000]])
    except Exception as e:
        return [f"⚠️ Error parsing {filepath}: {e}"]

def chunk_markdown(filepath):
    text = Path(filepath).read_text(encoding="utf-8")
    sections, current = [], []
    for line in text.splitlines():
        if line.startswith("#") and current:
            sections.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return re_chunk_if_oversize(sections)

def chunk_json_yaml(filepath):
    try:
        with open(filepath, 'r', encoding="utf-8") as f:
            data = yaml.safe_load(f) if filepath.suffix in {'.yaml', '.yml'} else json.load(f)
        chunks = []
        def chunk_value(value):
            text = json.dumps(value, indent=2)
            if count_tokens(text) <= MAX_TOKENS:
                chunks.append(text)
            else:
                if isinstance(value, dict):
                    for k, v in value.items():
                        chunk_value({k: v})
                elif isinstance(value, list):
                    for item in value:
                        chunk_value(item)
                else:
                    chunks.append(text[:1000])
        chunk_value(data)
        return re_chunk_if_oversize(chunks)
    except Exception as e:
        return [f"⚠️ Error parsing {filepath}: {e}"]

def chunk_html(filepath):
    return chunk_code_tree_sitter(filepath, '.html')

def chunk_xml(filepath):
    return chunk_code_tree_sitter(filepath, '.xml')

def chunk_css(filepath):
    return chunk_code_tree_sitter(filepath, '.css')

def chunk_ipynb(filepath):
    try:
        with open(filepath, 'r', encoding="utf-8") as f:
            notebook = json.load(f)
        chunks = []
        for i, cell in enumerate(notebook.get('cells', [])):
            source = ''.join(cell.get('source', [])).strip()
            if not source:
                continue
            if cell.get('cell_type') == 'markdown':
                chunks.append(f"[Markdown Cell {i+1}]\n{source}")
            elif cell.get('cell_type') == 'code':
                chunks.append(f"[Code Cell {i+1}]\n{source}")
        return re_chunk_if_oversize(chunks)
    except Exception as e:
        return [f"⚠️ Error parsing {filepath}: {e}"]

def chunk_csv_tsv(filepath):
    try:
        import pandas as pd
        df = pd.read_csv(filepath, sep=None, engine='python', nrows=5, encoding='utf-8', encoding_errors='ignore')
        columns = ", ".join(df.columns)
        preview = df.to_string(index=False)
        return re_chunk_if_oversize([f"Columns: {columns}\nPreview (first 5 rows):\n{preview}"])
    except Exception as e:
        return [f"⚠️ Error parsing {filepath}: {e}"]

def chunk_as_summary(filepath):
    return re_chunk_if_oversize([f"{Path(filepath).suffix.upper()[1:]} file at {filepath}"])

def dispatch_chunking(filepath):
    ext = Path(filepath).suffix.lower()
    if ext in {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".cs", ".go", ".rb", ".php", ".rs", ".swift", ".kt", ".kts", ".scala", ".cjs", ".mjs", ".sh"}:
        return chunk_code_tree_sitter(filepath, ext), True
    elif ext == ".ipynb":
        return chunk_ipynb(filepath), True
    elif ext in {".md", ".mdx", ".txt", ".rst", ".adoc"}:
        return chunk_markdown(filepath), True
    elif ext in {".json", ".yaml", ".yml", ".jsonl", ".webmanifest"}:
        return chunk_json_yaml(filepath), True
    elif ext in {".html", ".htm", ".xhtml"}:
        return chunk_html(filepath), True
    elif ext in {".css", ".scss", ".sass", ".less"}:
        return chunk_css(filepath), True
    elif ext in {".xml", ".xsd", ".xsl"}:
        return chunk_xml(filepath), True
    elif ext in {".csv", ".tsv"}:
        return chunk_csv_tsv(filepath), True
    else:
        return chunk_as_summary(filepath), True

def get_embedding(text):
    try:
        response = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ Failed to embed chunk: {e}")
        return []

def process_file(filepath, status, repo_name):
    chunks, embed = dispatch_chunking(filepath)
    chunk_entries = []
    full_text = Path(filepath).read_text(encoding="utf-8", errors="ignore") if embed else ""
    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        vector = get_embedding(chunk) if embed and count_tokens(chunk) <= MAX_TOKENS else []
        chunk_entry = {
            "id": chunk_id,
            "values": vector,
            "metadata": {
                "repo_name": repo_name,
                "file_path": str(filepath),
                "file_type": Path(filepath).suffix.lstrip('.'),
                "chunk_type": "summary" if not embed else "content",
                "chunk_id": chunk_id,
                "embed": embed,
                "content": chunk,
                "line_range": get_line_range(chunk, full_text),
                "embedded": bool(vector)
            }
        }
        chunk_entries.append(chunk_entry)
    return chunk_entries

def get_line_range(content, full_text):
    if not full_text or not content:
        return "L1-L1"
    try:
        start_idx = full_text.find(content)
        if start_idx == -1:
            return "L1-L1"
        before = full_text[:start_idx]
        start_line = before.count("\n") + 1
        line_count = content.count("\n") + 1
        end_line = start_line + line_count - 1
        return f"L{start_line}-L{end_line}"
    except:
        return "L1-L1"

def main(changed_files_path):
    repo_name = os.getenv("GITHUB_REPOSITORY", "unknown").split("/")[-1]
    to_upsert = []
    to_delete = []

    with open(changed_files_path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=2)
            if not parts:
                continue
            status = parts[0]
            filepath = parts[1] if status != "R" else parts[2]
            old_filepath = parts[1] if status == "R" else None

            if status == "D" or status == "R":
                to_delete.append((filepath, repo_name))
            if status in ["A", "M", "R"]:
                if Path(filepath).exists():
                    chunks = process_file(filepath, status, repo_name)
                    to_upsert.extend(chunks)

    # Delete old embeddings
    for filepath, repo_name in to_delete:
        index.delete(filter={"repo_name": repo_name, "file_path": filepath}, namespace=repo_name)

    # Upsert new embeddings in batches
    for i in tqdm(range(0, len(to_upsert), BATCH_SIZE), desc="Upserting to Pinecone"):
        batch = to_upsert[i:i + BATCH_SIZE]
        if batch:
            index.upsert(vectors=batch, namespace=repo_name)
    print(f"Upserted {len(to_upsert)} chunks to Pinecone for {repo_name}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])