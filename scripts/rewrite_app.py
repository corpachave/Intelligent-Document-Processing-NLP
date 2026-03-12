from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'src' / 'api' / 'app.py'
text = path.read_text(encoding='utf-8')

# Replace function body to ensure temp file is cleaned up
old = """    with tempfile.NamedTemporaryFile(suffix=\".pdf\", delete=False) as tmp:\n        shutil.copyfileobj(file.file, tmp)\n\n    result = extract_entities_from_pdf(tmp.name)\n    return result\n"""
new = """    with tempfile.NamedTemporaryFile(suffix=\".pdf\", delete=False) as tmp:\n        shutil.copyfileobj(file.file, tmp)\n        tmp_path = tmp.name\n\n    try:\n        result = extract_entities_from_pdf(tmp_path)\n        return result\n    finally:\n        try:\n            os.remove(tmp_path)\n        except OSError:\n            pass\n"""

if old not in text:
    raise SystemExit('Old text block not found; not modifying file.')

path.write_text(text.replace(old, new), encoding='utf-8')
print('Updated', path)
