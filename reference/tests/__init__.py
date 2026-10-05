"""Unit test package for Mantis ADK references and invariants."""


def ensure_reference_deps():
    """Fail closed, with a readable message, when the reference env is absent.

    These suites verify invariants enforced by the ADK runtime; without it
    there is nothing meaningful to test, so a missing environment is an
    error -- never a skip, never a pass.
    """
    try:
        import pydantic  # noqa: F401
        import google.adk  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The Mantis reference test suites require the reference "
            "environment. Run reference/install.sh or `pip install -r "
            "reference/requirements.txt` first. For dependency-light skill "
            "checks, run `python test_skills_integrity.py` at the repo root "
            f"instead. (missing: {exc.name})"
        ) from exc


ensure_reference_deps()
