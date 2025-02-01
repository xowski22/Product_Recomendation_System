import sys
from pathlib import Path

project_root = str(Path(__file__).parent)

if project_root not in sys.path:
    sys.path.append(project_root)


import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8080, reload=True)
