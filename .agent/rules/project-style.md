---
description: Project style rules - Python venv and English-only requirements
globs: ["**/*"]
alwaysApply: true
---

# Project Style Rules

## 1. Python Virtual Environment

All Python operations must be executed within a virtual environment (venv).

### Requirements

1. **Create venv**: If `venv` directory does not exist, create it first:
   ```bash
   python -m venv venv
   ```

2. **Activate venv**: Before running any Python commands:
   - Windows: `.\venv\Scripts\activate`
   - Linux/macOS: `source venv/bin/activate`

3. **Install dependencies**: All pip commands must run in activated venv:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run scripts**: All Python scripts must run within the venv.

---

## 2. File Language Rule

All generated files must be written in **English only**.

### Exception

README files (e.g., `README.md`, `README_CN.md`) may contain Chinese.

### Requirements

- Code comments: English only
- Variable/function names: English only
- Documentation (except README): English only
- Configuration files: English only
- Commit messages: English only
