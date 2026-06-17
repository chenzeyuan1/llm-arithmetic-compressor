#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/chenzeyuan1/llm-arithmetic-compressor.git"
BRANCH="main"
COMMIT_MESSAGE="${1:-first commit}"

cd "$(dirname "$0")"

if [ ! -d ".git" ]; then
  git init
fi

git add .

if git diff --cached --quiet; then
  echo "No staged changes to commit."
else
  git commit -m "$COMMIT_MESSAGE"
fi

git branch -M "$BRANCH"

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

echo
echo "Pushing to $REPO_URL"
echo "If GitHub asks for a password, paste a GitHub Personal Access Token instead."
echo

git push -u origin "$BRANCH"
