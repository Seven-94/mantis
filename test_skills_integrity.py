"""Dependency-light integrity checks for the Mantis skills.

Runs in a bare clone with nothing installed:

    python test_skills_integrity.py

Checks the skill surface itself; the reference implementation has its own
suites under reference/, which require reference/install.sh.
"""

import pathlib
import re
import subprocess
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent
SKILLS = sorted(REPO_ROOT.glob("mantis-*/SKILL.md"))
_REF_LINK = re.compile(
    r"\]\(((?:\.\./)?[\w\-./]*references/[\w\-./]+\.md)(?:#[^)]*)?\)"
)


class SkillFrontmatterTests(unittest.TestCase):
    def test_skills_discovered(self):
        self.assertGreater(len(SKILLS), 10)

    def test_frontmatter_declares_name_and_description(self):
        for skill in SKILLS:
            with self.subTest(skill=skill.parent.name):
                lines = skill.read_text(encoding="utf-8").splitlines()
                self.assertEqual(lines[0], "---")
                body = lines[1 : lines[1:].index("---") + 1]
                self.assertTrue(any(l.startswith("name:") for l in body))
                self.assertTrue(any(l.startswith("description:") for l in body))

    def test_frontmatter_name_matches_directory(self):
        for skill in SKILLS:
            with self.subTest(skill=skill.parent.name):
                text = skill.read_text(encoding="utf-8")
                self.assertIn(f"name: {skill.parent.name}", text)


class ReferenceLinkTests(unittest.TestCase):
    def test_reference_links_resolve_and_are_tracked(self):
        for skill in SKILLS:
            for m in _REF_LINK.finditer(skill.read_text(encoding="utf-8")):
                target = (skill.parent / m.group(1)).resolve()
                with self.subTest(skill=skill.parent.name, link=m.group(1)):
                    self.assertTrue(
                        target.is_file(), f"broken link: {m.group(1)}"
                    )
                    rc = subprocess.run(
                        [
                            "git",
                            "ls-files",
                            "--error-unmatch",
                            str(target.relative_to(REPO_ROOT)),
                        ],
                        cwd=REPO_ROOT,
                        capture_output=True,
                    ).returncode
                    self.assertEqual(
                        rc, 0, "target exists but is not git-tracked"
                    )


if __name__ == "__main__":
    unittest.main()
