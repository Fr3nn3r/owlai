# save as extract_imports.py
import os
import ast
import codecs


def try_read_file(filepath, encodings=["utf-8", "latin1", "cp1252"]):
    """Try to read a file with different encodings."""
    for encoding in encodings:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading {filepath}: {str(e)}")
            return None
    return None


project_dir = "."  # or specify your src directory
imports = set()

for root, _, files in os.walk(project_dir):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            content = try_read_file(filepath)

            if content is None:
                print(f"Skipped {filepath} due to encoding issues.")
                continue

            try:
                tree = ast.parse(content, filename=filepath)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            imports.add(n.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split(".")[0])
            except SyntaxError:
                print(f"Skipped {filepath} due to syntax error.")
            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")

with open("used_imports.txt", "w", encoding="utf-8") as out:
    for imp in sorted(imports):
        out.write(f"{imp}\n")

print("Finished. Check used_imports.txt.")
