#!/usr/bin/env python
"""
Clone a specific branch of a Hugging Face repo into a new repo named
"orig_name__branch" (with branch sanitized), using only huggingface_hub.

Examples:
  - Clone branch "dev" from "org/model" to "org/model__dev":
      python scripts/hf_clone_branch_to_repo.py \
        --source-repo-id org/model \
        --branch dev

  - Clone to a different namespace and make it private:
      python scripts/hf_clone_branch_to_repo.py \
        --source-repo-id org/model \
        --branch feature/x \
        --target-namespace my-username \
        --private

Authentication:
  - Provide a token via --token or set the HF_TOKEN environment variable.
  - The token must have write access to the target namespace/org.

Notes:
  - Uses huggingface_hub.snapshot_download + upload_folder, no git CLI.
  - If the target repo exists, it will be updated (unless --delete-existing is set).
"""

import argparse
import os
import re
import sys
import tempfile
from typing import Optional, Tuple

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError


def parse_repo_id(repo_id: str) -> Tuple[str, str]:
    """Split a repo_id like "owner/name" into (owner, name)."""
    if "/" not in repo_id:
        raise ValueError(f"Invalid repo_id '{repo_id}'. Expected format 'owner/name'.")
    owner, name = repo_id.split("/", 1)
    if not owner or not name:
        raise ValueError(f"Invalid repo_id '{repo_id}'. Expected format 'owner/name'.")
    return owner, name


def sanitize_branch_for_repo_name(branch: str) -> str:
    """Sanitize branch to be safe in a repo name.

    - Replace path separators and spaces with '-'
    - Keep alphanumerics, '-', '_', '.'
    - Replace any other char with '-'
    - Collapse multiple '-' into one
    - Trim leading/trailing '-'
    """
    replaced = branch.replace("/", "-").replace(" ", "-")
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", replaced)
    safe = re.sub(r"-{2,}", "-", safe).strip("-")
    return safe or "branch"


def resolve_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        return cli_token
    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return env_token


def branch_exists(api: HfApi, repo_id: str, branch: str, token: Optional[str], repo_type: str) -> bool:
    try:
        refs = api.list_repo_refs(repo_id=repo_id, repo_type=repo_type, token=token)
    except Exception:
        # Fallback: try snapshot; if it works, branch exists
        try:
            with tempfile.TemporaryDirectory() as tmp:
                snapshot_download(
                    repo_id=repo_id,
                    revision=branch,
                    repo_type=repo_type,
                    local_dir=tmp,
                    local_dir_use_symlinks=False,
                    token=token,
                )
            return True
        except Exception:
            return False

    for b in getattr(refs, "branches", []) or []:
        if getattr(b, "name", None) == branch:
            return True
    return False


def create_or_reset_repo(
    api: HfApi, target_repo_id: str, *, token: Optional[str], repo_type: str, private: bool, delete_existing: bool
) -> None:
    if delete_existing:
        try:
            api.delete_repo(repo_id=target_repo_id, token=token, repo_type=repo_type, missing_ok=True)
        except TypeError:
            # Older huggingface_hub without missing_ok
            try:
                api.delete_repo(repo_id=target_repo_id, token=token, repo_type=repo_type)
            except HfHubHTTPError:
                pass

    # Try modern signature first (repo_id=...)
    try:
        api.create_repo(repo_id=target_repo_id, token=token, repo_type=repo_type, private=private, exist_ok=True)
        return
    except TypeError:
        pass

    # Fallback to legacy signature: name + organization
    owner, name = parse_repo_id(target_repo_id)
    organization = owner  # If owner is the authenticated user, HF will treat this as org and likely fail; user should omit --target-namespace in that case
    try:
        api.create_repo(
            name=name, token=token, repo_type=repo_type, private=private, exist_ok=True, organization=organization
        )
    except TypeError:
        # Last resort: attempt without organization (creates under auth user)
        api.create_repo(name=name, token=token, repo_type=repo_type, private=private, exist_ok=True)


def upload_snapshot_folder(
    api: HfApi, folder_path: str, target_repo_id: str, *, token: Optional[str], repo_type: str, commit_message: str
) -> None:
    # Use upload_folder in chunks; excludes are handled implicitly (no .git in snapshot)
    api.upload_folder(
        repo_id=target_repo_id,
        token=token,
        repo_type=repo_type,
        folder_path=folder_path,
        path_in_repo="",
        commit_message=commit_message,
        allow_patterns=None,
        ignore_patterns=None,
        run_as_future=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone a HF repo branch into a new repo named orig_name__branch")
    parser.add_argument("--source-repo-id", required=True, help="Source repo id in the form 'owner/name'")
    parser.add_argument("--branch", required=True, help="Source branch (revision) to clone")
    parser.add_argument(
        "--target-namespace", default=None, help="Target namespace/owner (default: same as source owner)"
    )
    parser.add_argument(
        "--target-repo-name", default=None, help="Override target repo name (default: orig_name__sanitized_branch)"
    )
    parser.add_argument("--private", action="store_true", help="Create target repo as private")
    parser.add_argument("--delete-existing", action="store_true", help="If target exists, delete before re-creating")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"], help="HF repo type")
    parser.add_argument("--token", default=None, help="HF token (else read HF_TOKEN/HUGGINGFACEHUB_API_TOKEN)")

    args = parser.parse_args()

    token = resolve_token(args.token)
    if not token:
        print(
            "Warning: proceeding without explicit token. Ensure you're authenticated in environment.", file=sys.stderr
        )

    source_owner, source_name = parse_repo_id(args.source_repo_id)
    target_owner = args.target_namespace or source_owner
    branch_safe = sanitize_branch_for_repo_name(args.branch)
    target_name = args.target_repo_name or f"{source_name}__{branch_safe}"
    target_repo_id = f"{target_owner}/{target_name}"

    api = HfApi()

    # Validate branch exists
    if not branch_exists(api, args.source_repo_id, args.branch, token, args.repo_type):
        print(f"Error: Branch '{args.branch}' not found in {args.source_repo_id} ({args.repo_type}).", file=sys.stderr)
        sys.exit(1)

    # Download snapshot of the specified branch
    with tempfile.TemporaryDirectory() as tmpdir:
        local_dir = os.path.join(tmpdir, "snapshot")
        try:
            snapshot_download(
                repo_id=args.source_repo_id,
                revision=args.branch,
                repo_type=args.repo_type,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=token,
            )
        except HfHubHTTPError as e:
            print(f"Error downloading snapshot: {e}", file=sys.stderr)
            sys.exit(1)

        # Create/Reset target repo
        create_or_reset_repo(
            api,
            target_repo_id=target_repo_id,
            token=token,
            repo_type=args.repo_type,
            private=bool(args.private),
            delete_existing=bool(args.delete_existing),
        )

        # Upload folder
        commit_message = f"Import from {args.source_repo_id}@{args.branch}"
        upload_snapshot_folder(
            api,
            folder_path=local_dir,
            target_repo_id=target_repo_id,
            token=token,
            repo_type=args.repo_type,
            commit_message=commit_message,
        )

    web_url = (
        f"https://huggingface.co/{target_repo_id}"
        if args.repo_type == "model"
        else (
            f"https://huggingface.co/datasets/{target_repo_id}"
            if args.repo_type == "dataset"
            else f"https://huggingface.co/spaces/{target_repo_id}"
        )
    )
    print(target_repo_id)
    print(web_url)


if __name__ == "__main__":
    main()
