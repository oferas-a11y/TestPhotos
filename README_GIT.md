# Git & Release Workflow

Follow these steps to version the project safely. Do not push without explicit approval.

## 1) Initialize repo locally
```bash
git init
git lfs install
git add .
git commit -m "chore: initialize project with main_app pipeline and docs"
```

## 2) Add remote (when approved)
```bash
git remote add origin <git-remote-url>
git checkout -b main
git push -u origin main
```

## 3) Feature branches
```bash
git checkout -b feat/<short-meaningful-name>
# commit work in small steps
git push -u origin feat/<short-meaningful-name>
```

Open a Pull Request for review. Do not modify verified pipelines unless instructed.

## 4) What is ignored
- `.env` and all secrets
- `main_app_outputs/` (results), except embeddings files are allowed
- Large model files (use LFS)

## 5) Release notes
Document changes in `CHANGELOG.md` and update `CURSOR.md` if dev steps change.


