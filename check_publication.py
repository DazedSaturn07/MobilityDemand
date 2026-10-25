"""Check Git candidates for unexpected files, notebook outputs, and common private data."""

import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    ".gitattributes",
    ".gitignore",
    "README.md",
    "PROBLEM_STATEMENT.md",
    "requirements.txt",
    "run_notebooks.py",
    "check_publication.py",
    "data/hour.csv",
    "data/source.json",
    "mobility/__init__.py",
    "mobility/data.py",
    "mobility/features.py",
    "mobility/evaluation.py",
    "mobility/scoring.py",
    "sql/hourly_summary.sql",
    "notebooks/01_data.ipynb",
    "notebooks/02_model.ipynb",
    "notebooks/03_capacity.ipynb",
    "notebooks/04_review.ipynb",
    "tests/test_project.py",
}


def git_candidates(git_dir=None):
    args = ["git", "-c", "core.excludesFile="]
    if git_dir:
        args += [f"--git-dir={git_dir}", f"--work-tree={ROOT}"]
    result = subprocess.run(
        args + ["ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return {name for name in result.stdout.decode("utf-8").split("\0") if name}


def review(candidates):
    issues = []
    for name in sorted(candidates - EXPECTED):
        issues.append(f"Unexpected publishable file: {name}")
    for name in sorted(EXPECTED - candidates):
        issues.append(f"Required public file missing or ignored: {name}")
    patterns = {
        "local absolute path": r"(?i)[a-z]:[\\/](?:Users|X1|saas)[\\/]",
        "email address": r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        "private key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "access token": r"(?:gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{32,})",
    }
    for name in sorted(candidates & EXPECTED):
        text = (ROOT / name).read_text(encoding="utf-8-sig")
        for label, pattern in patterns.items():
            if re.search(pattern, text):
                issues.append(f"Review {label} in {name}")
        if name.endswith(".ipynb"):
            for cell in json.loads(text)["cells"]:
                if cell["cell_type"] == "code" and (
                    cell["outputs"] or cell["execution_count"] is not None
                ):
                    issues.append(
                        f"Clear source notebook outputs before publication: {name}"
                    )
                    break
    return issues


def main():
    # Before Git initialization, test the actual ignore rules using a temporary
    # empty Git directory. This never initializes or changes the project's Git.
    if (ROOT / ".git").exists():
        candidates = git_candidates()
    else:
        scratch = ROOT / "output" / ".runtime"
        scratch.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=scratch) as temp:
            subprocess.run(
                ["git", "init", "--bare", "--quiet", temp],
                check=True,
                capture_output=True,
            )
            candidates = git_candidates(temp)
    issues = review(candidates)
    if issues:
        print("\n".join(issues))
        raise SystemExit(1)
    print(
        f"Publication check passed: {len(candidates)} expected files; two public Markdown files; no source notebook outputs or matched private-data patterns."
    )
    print(
        "Review the staged diff before every push. Pattern checks cannot guarantee the absence of all sensitive information."
    )


if __name__ == "__main__":
    main()
