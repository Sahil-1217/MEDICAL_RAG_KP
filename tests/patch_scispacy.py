import pathlib

base = pathlib.Path(r"C:\Users\sahil\AppData\Roaming\Python\Python313\site-packages\en_core_sci_sm")
for p in base.glob("**/*.cfg"):
    text = p.read_text(encoding="utf-8")
    new_text = text.replace('include_static_vectors = "False"', 'include_static_vectors = false')
    new_text = new_text.replace('include_static_vectors = "True"', 'include_static_vectors = true')
    if text != new_text:
        p.write_text(new_text, encoding="utf-8")
        print(f"Patched: {p}")
print("Config patch complete.")
