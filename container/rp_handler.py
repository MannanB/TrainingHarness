import os
import sys
import time
import io
import shutil
import zipfile
import importlib
from typing import Tuple, Optional

import runpod
import wandb
import requests  # wandb already depends on this, but we import it explicitly



LOCAL_ROOT = "/tmp/trainingharness"

# Default: your TrainingHarness repo main branch
DEFAULT_ZIP_URL = "https://github.com/MannanB/TrainingHarness/archive/refs/heads/main.zip"


def _extract_root_dir_from_zip(zf: zipfile.ZipFile) -> str:

    # GitHub archives always have everything under a single top-level folder.
    names = zf.namelist()
    top_level_dirs = {name.split("/", 1)[0] for name in names if "/" in name}
    if not top_level_dirs:
        raise RuntimeError("Unexpected zip layout: no top-level directories found.")
    # There should only be one, but we just pick the first sorted for determinism.
    return sorted(top_level_dirs)[0]


def sync_code(zip_url: str) -> Tuple[str, str]:
    os.makedirs(LOCAL_ROOT, exist_ok=True)

    print(f"[bootstrap] Downloading latest repo zip from {zip_url} ...")
    resp = requests.get(zip_url, timeout=30)
    resp.raise_for_status()

    # Clean any previous extraction
    if os.path.isdir(LOCAL_ROOT):
        shutil.rmtree(LOCAL_ROOT, ignore_errors=True)
        os.makedirs(LOCAL_ROOT, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        root_dir_name = _extract_root_dir_from_zip(zf)
        zf.extractall(LOCAL_ROOT)

    root_path = os.path.join(LOCAL_ROOT, root_dir_name)
    if not os.path.isdir(root_path):
        raise RuntimeError(f"[bootstrap] Expected root path not found: {root_path}")

    print(f"[bootstrap] Synced code. Root path: {root_path}")
    return root_path, root_dir_name


def _find_default_package(root_path: str) -> Optional[str]:
    candidates = []
    for entry in os.listdir(root_path):
        full = os.path.join(root_path, entry)
        if (
            os.path.isdir(full)
            and os.path.isfile(os.path.join(full, "__init__.py"))
        ):
            candidates.append(entry)

    candidates.sort()
    if not candidates:
        return None
    return candidates[0]


def _import_module_from_trainingharness(root_path: str, project_name: str):
    # Make container/ the import root so `projects` is importable.
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

    module_path = f"projects.{project_name}"
    print(f"[import] Importing TrainingHarness module: {module_path} from {root_path}")
    module = importlib.import_module(module_path)

    if not hasattr(module, "main"):
        raise RuntimeError(f"[import] Module {module_path} has no `main(run, cfg)` function.")

    return module


def _import_module_from_generic_repo(
    root_path: str, ghentry: Optional[str]
):
    # Make the repo root importable
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

    if ghentry:
        module_path = ghentry
        print(f"[import] Importing custom entrypoint module: {module_path} (root: {root_path})")
    else:
        default_pkg = _find_default_package(root_path)
        if not default_pkg:
            raise RuntimeError(
                "[import] Could not auto-detect default package. "
                "Please provide `github_entrypoint` in the request."
            )
        # This is your '[repo]' auto-detected from the repo structure.
        module_path = default_pkg
        print(
            f"[import] Auto-detected default repo package '{module_path}' "
            f"under {root_path}"
        )

    module = importlib.import_module(module_path)

    if not hasattr(module, "main"):
        raise RuntimeError(f"[import] Module {module_path} has no `main(run, cfg)` function.")

    return module


def handler(event):
    print("Worker Start")
    input_payload = event["input"]

    use_wandb = input_payload.get("wandb_logging", True)
    project_name = input_payload.get("project_name", "default_project")

    # Optional GitHub zip URL & entrypoint
    zip_url = input_payload.get("github_zip_url", DEFAULT_ZIP_URL)
    ghentry = input_payload.get("github_entrypoint")  # None / "" means "auto"

    cfg = input_payload.get("config", {})

    run = None
    run_link = None
    run_id = None

    # --- W&B SETUP ---
    if use_wandb:
        os.environ["WANDB_API_KEY"] = input_payload["W&B_API_KEY"]
        wandb.login()

        wandb_project_name = input_payload.get("wandb_project_name", "default_wandb_project")
        wandb_run_name = input_payload.get("run_name", f"run_{int(time.time())}")
        wandb_user_name = input_payload.get("wandb_user_name", None)

        run = wandb.init(
            project=wandb_project_name,
            entity=wandb_user_name,
            config=cfg,
            name=wandb_run_name,
        )
        run_link = f"https://wandb.ai/{wandb_user_name}/{wandb_project_name}/runs/{run.id}"
        run_id = run.id

    # --- SYNC CODE FROM GITHUB ---
    if os.environ.get("TRAINING_HARNESS_LOCAL_RUN") == "1":
        # Local run: use local code
        root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        root_dir_name = os.path.basename(root_path)
        print(f"[bootstrap] Local run detected. Using local code at {root_path}")
        project_module = _import_module_from_trainingharness(root_path, project_name)
    else:
        root_path, root_dir_name = sync_code(zip_url)

        # --- IMPORT PROJECT / ENTRYPOINT MODULE ---
        if zip_url == DEFAULT_ZIP_URL:
            # Case (a): TrainingHarness default
            #   Use TrainingHarness/projects/<project_name>.py
            #   and import `projects.<project_name>` like before.
            project_module = _import_module_from_trainingharness(root_path, project_name)
        else:
            # Case (b): Custom GitHub zip
            #   If ghentry is None/blank:
            #       find [repo] = top-level package under root, import [repo]
            #   Else:
            #       import [repo].[ghentry] where ghentry is a full module path
            #       relative to repo root (e.g. "mypkg.entrypoint")
            ghentry_clean = ghentry.strip() if isinstance(ghentry, str) else None
            project_module = _import_module_from_generic_repo(root_path, ghentry_clean)

    project_module.main(run, cfg)

    if use_wandb:
        wandb.finish()

    return {"status": "completed", "run_id": run_id, "run_link": run_link}


# Start the Serverless function when the script is run
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
