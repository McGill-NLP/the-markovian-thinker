import glob
import re
from pathlib import Path
from typing import Optional

import fire

STEP_DIR_REGEX = re.compile(r"global_step_(\d+)$")


def _find_hf_checkpoints(base_dir: Path) -> list[tuple[int, Path]]:
    """Find (step, huggingface_dir) pairs for ACTOR under base_dir.

    Expected layout:
      base_dir/global_step_XXXX/actor/huggingface/

    We collect `huggingface` directories whose parent is `actor` and infer
    step from the first ancestor folder matching `global_step_XXXX`.
    """
    results: list[tuple[int, Path]] = []
    if not base_dir.exists():
        return results

    for hf_dir in base_dir.rglob("huggingface"):
        if not hf_dir.is_dir():
            continue
        # Only care about actor checkpoints
        if hf_dir.parent.name != "actor":
            continue
        # Walk up to find the global_step folder
        current = hf_dir
        step: Optional[int] = None
        while current != base_dir and current != current.parent:
            if STEP_DIR_REGEX.search(current.name):
                m = STEP_DIR_REGEX.search(current.name)
                step = int(m.group(1))
                break
            current = current.parent
        if step is None:
            continue
        results.append((step, hf_dir))
    # Deduplicate by (step, path)
    unique: dict[tuple[int, str], Path] = {}
    for s, p in results:
        unique[(s, str(p))] = p
    # Rebuild paths from keys
    canonical: list[tuple[int, Path]] = []
    for (s, p_str), _ in sorted(unique.items(), key=lambda kv: kv[0][0]):
        canonical.append((s, Path(p_str)))
    return canonical


def _list_existing_hf_branches(repo_id: str) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    # Ensure repo exists; create if needed to avoid 404s
    repo_id = api.create_repo(repo_id, repo_type="model", exist_ok=True).repo_id
    refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
    branches = [b.name for b in (refs.branches or [])]
    return branches


def _upload_folder_to_branch(repo_id: str, folder: Path, branch_name: str, commit_message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = api.create_repo(repo_id, repo_type="model", exist_ok=True).repo_id
    api.create_branch(repo_id, branch=branch_name, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        path_in_repo="",
        repo_type="model",
        commit_message=commit_message,
        revision=branch_name,
    )


def upload_missing(
    repo_id: str,
    base_dir: str,
    dry_run: bool = False,
) -> None:
    """Upload any local HF checkpoints that do not yet exist as branches on the Hub.

    - repo_id: HF repo like "org/repo".
    - base_dir: Root directory containing `global_step_XXXX` subfolders.
    - dry_run: Only print actions without uploading.
    """
    base = Path(base_dir).expanduser().resolve()
    items = _find_hf_checkpoints(base)

    existing_branches = set(_list_existing_hf_branches(repo_id))

    to_upload: list[tuple[int, Path, str]] = []
    for step, hf_dir in items:
        branch = f"ckpt-{step:06d}"
        if branch in existing_branches:
            print(f"Skip step {step}: branch {branch} already exists")
            continue
        if not hf_dir.exists():
            print(f"Skip step {step}: missing folder {hf_dir}")
            continue
        to_upload.append((step, hf_dir, branch))

    if not to_upload:
        print("No missing checkpoints to upload.")
        return

    print("Planned uploads:")
    for step, folder, branch in to_upload:
        print(f"  - step {step} -> branch {branch} from {folder}")

    if dry_run:
        print("Dry run enabled; no uploads performed.")
        return

    for step, folder, branch in to_upload:
        msg = f"Checkpoint push for step {step:06d}"
        print(f"Uploading {folder} to {repo_id}@{branch} ...")
        _upload_folder_to_branch(repo_id, folder, branch, msg)
        print(f"Uploaded {folder} to {repo_id}@{branch}")


def main(
    *exp_roots: str,
    repo_id: Optional[str] = None,
    dry_run: bool = False,
):
    """Upload missing HF checkpoints for all matching experiment roots.

    Usage:
      python scripts/upload_missing_hf_checkpoints.py some_pattern_* [--repo_id <org/repo>] [--dry_run True]

    Notes:
      - Each experiment root must contain exactly one seed directory matching *@sd_<seed>
      - Each seed dir is the base dir that contains global_step_XXXX/actor/huggingface/
      - If --repo_id is not provided, each root's repo id defaults to its directory name.
    """
    if not exp_roots:
        raise ValueError("Please provide at least one experiment root or glob pattern")

    # Expand any glob patterns that were not expanded by the shell (e.g., quoted)
    candidate_paths: list[Path] = []
    for token in exp_roots:
        # If token contains wildcard characters and doesn't exist as-is, expand
        p = Path(token).expanduser()
        if any(ch in token for ch in ["*", "?", "["]):
            matches = [Path(m).expanduser() for m in glob.glob(str(p))]
            if matches:
                candidate_paths.extend(matches)
                continue
        candidate_paths.append(p)

    # Normalize, dedupe, ensure existence
    roots: list[Path] = []
    seen: set[str] = set()
    for p in candidate_paths:
        rp = p.resolve()
        if not rp.exists():
            print(f"Skip missing path: {rp}")
            continue
        sp = str(rp)
        if sp in seen:
            continue
        seen.add(sp)
        roots.append(rp)

    if not roots:
        print("No valid experiment roots to process.")
        return

    print("Experiment roots:")
    for r in roots:
        print(f"  - {r}")

    for root in roots:
        # Discover exactly one seed dir
        seed_dirs = sorted([p for p in root.iterdir() if p.is_dir() and "@sd_" in p.name])
        if not seed_dirs:
            print(f"[{root.name}] No seed dirs (*@sd_*) found under {root}. Skipping.")
            continue
        if len(seed_dirs) > 1:
            print(f"[{root.name}] Multiple seed dirs (*@sd_*) found under {root}. Skipping.")
            continue

        repo = repo_id or root.name
        seed_dir = seed_dirs[0]

        print(f"[{root.name}] Repo: {repo}")
        print(f"[{root.name}] Seed directory: {seed_dir}")

        upload_missing(repo_id=repo, base_dir=str(seed_dir), dry_run=dry_run)


if __name__ == "__main__":
    fire.Fire(main)
