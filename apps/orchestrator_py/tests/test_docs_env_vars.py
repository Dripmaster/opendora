import re
from pathlib import Path
from orchestrator.config import AppEnv


def test_env_vars_in_docs():
    """
    Asserts that every environment variable defined in AppEnv
    is present in both README.md and .env.example.
    """
    # 1. Get all env var names from AppEnv
    env_vars = list(AppEnv.model_fields.keys())

    # 2. Read README.md and .env.example
    root_dir = Path(__file__).resolve().parents[3]
    readme_path = root_dir / "README.md"
    env_example_path = root_dir / ".env.example"

    readme_content = readme_path.read_text(encoding="utf-8")
    env_example_content = env_example_path.read_text(encoding="utf-8")

    missing_in_readme = []
    missing_in_env_example = []

    for var in env_vars:
        # Use word boundary to avoid substring false positives
        pattern = rf"\b{var}\b"

        if not re.search(pattern, readme_content):
            missing_in_readme.append(var)

        if not re.search(pattern, env_example_content):
            missing_in_env_example.append(var)

    assert not missing_in_readme, f"Missing in README.md: {missing_in_readme}"
    assert not missing_in_env_example, (
        f"Missing in .env.example: {missing_in_env_example}"
    )
