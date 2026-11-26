import json
from pathlib import Path

ROOT = Path("/Users/oleg/PycharmProjects/Ermakov/Ermakova10SEM/finallab/ErmakovaFinal")
SOURCE = ROOT / "working files/source_reports/SSH_ГА_сдвиг_влево_ошибки_дисперсия_Colab.ipynb"
TARGET = ROOT / "working files/notebook_extracted.md"

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
lines = [f"# Notebook: {SOURCE.name}", ""]
for index, cell in enumerate(notebook.get("cells", []), start=1):
    cell_type = cell.get("cell_type", "unknown")
    lines.extend([f"## Cell {index}: {cell_type}", "", "```python" if cell_type == "code" else "```text"])
    lines.extend("".join(cell.get("source", [])).splitlines())
    lines.extend(["```", ""])
    outputs = cell.get("outputs", [])
    if outputs:
        lines.extend(["### Outputs", "", "```text"])
        for output in outputs:
            if "text" in output:
                lines.extend("".join(output["text"]).splitlines())
            data = output.get("data", {})
            if "text/plain" in data:
                lines.extend("".join(data["text/plain"]).splitlines())
            if "ename" in output:
                lines.append(f"{output.get('ename')}: {output.get('evalue')}")
        lines.extend(["```", ""])
TARGET.write_text("\n".join(lines), encoding="utf-8")
print(f"cells={len(notebook.get('cells', []))}")
print(f"output={TARGET}")
