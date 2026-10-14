"""The requirements files must agree with ``pyproject.toml``.

Two lists of dependencies drift, and the drift is silent: the day someone bumps a version
in one and not the other, the conda user and the pip user are running different libraries
and nothing says so. This repository exists because generic code drifted out of one
repository into three others, so the same failure in miniature is worth a test.

``pyproject.toml`` is the single source of truth. The requirements files are generated
from it by ``tools/sync_requirements.py``, and this test fails when they are stale — which
converts "remember to update both" into a red test.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import sync_requirements  # noqa: E402


@pytest.fixture(scope="module")
def config() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


@pytest.fixture(scope="module")
def rendered(config) -> dict[str, str]:
    return sync_requirements.render(config)


def test_every_generated_file_is_up_to_date(rendered):
    """The whole point. If this fails, run `python tools/sync_requirements.py`."""
    stale = [
        name
        for name, content in rendered.items()
        if not (ROOT / name).exists() or (ROOT / name).read_text(encoding="utf-8") != content
    ]
    assert stale == [], (
        f"{stale} disagree with pyproject.toml. Run: python tools/sync_requirements.py"
    )


def test_the_check_mode_agrees(rendered):
    """`sync_requirements.py --check` is what a CI job runs, so it must not disagree
    with this test about whether the files are current."""
    assert sync_requirements.main(check=True) == 0
    assert rendered  # and it had something to check


def test_every_core_dependency_appears_in_requirements(config):
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for requirement in config["project"]["dependencies"]:
        assert requirement in text, requirement


def test_the_core_file_pulls_in_nothing_optional(config):
    """The core stays small on purpose (D7): a `pip install -r requirements.txt` must
    not drag in darts, transformers or safetycage."""
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines == config["project"]["dependencies"]

    for extra, requirements in config["project"]["optional-dependencies"].items():
        for requirement in requirements:
            package = requirement.split(">")[0].split("=")[0].split("[")[0]
            assert package not in lines, f"{package} from [{extra}] leaked into the core"


def test_the_extras_menu_is_commented_out(config):
    """Uncommented, `pip install -r requirements-optional.txt` would install every
    extra at once — and darts and safetycage have transitive pins that may not
    co-exist. The file is a menu, not an install."""
    text = (ROOT / "requirements-optional.txt").read_text(encoding="utf-8")
    active = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert active == ["-r requirements.txt"]

    # but every extra is present as a commented line, so nothing is silently omitted
    for extra, requirements in config["project"]["optional-dependencies"].items():
        if extra in ("dev", "tutorial"):
            continue  # each has its own file
        for requirement in requirements:
            assert f"# {requirement}" in text, f"{requirement} missing from the menu"


def test_the_dev_file_covers_the_dev_extra(config):
    text = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "-r requirements.txt" in text
    for requirement in config["project"]["optional-dependencies"]["dev"]:
        assert requirement in text, requirement


def test_the_dev_extra_carries_what_the_readme_tells_people_to_run(config):
    """The README says `python -m pytest` and `ruff check src tests tools`, so both have
    to be installed by the thing it tells people to install."""
    dev = " ".join(config["project"]["optional-dependencies"]["dev"])
    assert "pytest" in dev
    assert "ruff" in dev


def test_dev_is_an_extra_so_that_the_documented_command_actually_works(config):
    """The regression this pins, which cost nothing to make and was invisible.

    `dev` used to be a PEP 735 dependency-group. A group is only reachable as
    `pip install --group dev` (pip 25.1+); `pip install -e ".[dev]"` — the command in
    this repo's README *and* in two CI jobs — resolves `[dev]` as an *extra*, finds none,
    prints `WARNING: does not provide the extra 'dev'`, and installs neither pytest nor
    ruff. Both CI jobs then ran `pytest` against whatever happened to be in the image.

    So `dev` must stay an extra for as long as the documentation says `.[dev]`.
    """
    assert "dev" in config["project"]["optional-dependencies"]
    assert "dev" not in config.get("dependency-groups", {})

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    ci = (ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8")
    for name, text in (("README.md", readme), (".gitlab-ci.yml", ci)):
        for extra in _extras_referenced(text):
            assert extra in config["project"]["optional-dependencies"], (
                f"{name} tells people to install the extra '{extra}', "
                f"which pyproject.toml does not define"
            )


def _extras_referenced(text: str) -> set[str]:
    """Every `.[extra]` (or `.[a,b]`) an install command in the text asks for."""
    import re

    found: set[str] = set()
    for group in re.findall(r'(?:pip install[^\n]*?)\.\[([a-zA-Z0-9_,\- ]+)\]', text):
        found.update(part.strip() for part in group.split(",") if part.strip())
    return found


def test_the_tutorial_file_covers_the_tutorial_extra(config):
    text = (ROOT / "requirements-tutorial.txt").read_text(encoding="utf-8")
    for requirement in config["project"]["optional-dependencies"]["tutorial"]:
        assert requirement in text, requirement


def _conda_specs_in(text: str) -> list[str]:
    """The active entries under the conda file's ``dependencies:`` key.

    ``line.lstrip("- ")`` is wrong here and was the first thing I wrote: ``lstrip``
    removes *characters*, not a prefix, so it turns ``-r requirements.txt`` into
    ``r requirements.txt``. ``removeprefix`` is the one that means what it looks like.
    """
    entries = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and not stripped.startswith("- #"):
            entries.append(stripped.removeprefix("- ").strip())
    return entries


def test_the_conda_specs_come_from_pyproject(config):
    """One source of truth. A conda spec and a requirements line that disagree about a
    version is the drift this whole arrangement exists to prevent — so the conda specs
    are generated from the same lists, with the constraints unchanged."""
    text = (ROOT / "environment.yml").read_text(encoding="utf-8")
    specs = _conda_specs_in(text)
    project = config["project"]
    for requirement in [
        *project["dependencies"],
        *project["optional-dependencies"]["dev"],
        *project["optional-dependencies"]["tutorial"],
    ]:
        name = requirement.partition(">")[0].strip()
        expected = sync_requirements.CONDA_NAMES.get(name, name)
        matching = [spec for spec in specs if spec.partition(">")[0] == expected]
        assert matching, f"{expected} is missing from environment.yml"
        assert matching[0] == requirement.replace(name, expected, 1), (
            f"environment.yml pins {matching[0]!r} but pyproject says {requirement!r}"
        )


def test_the_conda_environment_can_run_the_tutorial():
    """`conda env create` must be enough to open docs/tutorial.ipynb.

    It was not, once: Jupyter was a commented suggestion, so the one path aimed squarely
    at someone exploring the code could not run the documentation that explains it.
    """
    specs = _conda_specs_in((ROOT / "environment.yml").read_text(encoding="utf-8"))
    assert any(spec.startswith("jupyter") for spec in specs)


def test_the_conda_environment_carries_the_build_backend():
    """So that `pip install -e . --no-deps --no-build-isolation` needs no network.

    The bug this pins cost a user two failed `conda env create` runs. Building an
    editable install needs the backend (`hatchling`) *and* `editables`, and pip's build
    isolation downloads both from PyPI even when they are already in the environment. On
    a network that truncates PyPI responses that is exactly where it broke:
    `IncompleteRead(14094 bytes read, 815 more expected)` while fetching hatchling.

    With both placed by conda from conda-forge, `--no-build-isolation` makes the build
    entirely local. `--no-build-isolation` cannot go in the conda file itself — it is a
    command-line-only flag, and `PIP_NO_BUILD_ISOLATION` is not honoured by pip 24 — so
    the editable install is a documented second command, and these two packages are what
    make that command offline.
    """
    specs = _conda_specs_in((ROOT / "environment.yml").read_text(encoding="utf-8"))
    names = {spec.partition(">")[0] for spec in specs}
    for tool in sync_requirements.CONDA_BUILD_TOOLS:
        assert tool in names, f"{tool} is needed to build the editable install offline"


def test_the_conda_file_has_no_pip_section():
    """Deliberately. A `pip:` list is flattened into one requirements file and installed
    with `pip install -U -r`, which puts every wheel back on PyPI and lets a single
    truncated download fail the whole environment — the failure this file was changed to
    avoid. conda-forge is a different index reached by a different downloader."""
    # checked against the active YAML, not the prose: the file's own comments explain
    # why there is no pip section, and mention `pip:` while doing so
    active = [
        line.strip()
        for line in (ROOT / "environment.yml").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "- pip:" not in active
    assert "- pip" in active  # but pip itself is present, for the editable install


def test_the_generated_files_say_they_are_generated(rendered):
    """Someone will otherwise edit one by hand and lose the change on the next sync."""
    for name, content in rendered.items():
        head = content.splitlines()[0]
        assert "GENERATED" in head, name
        assert "sync_requirements.py" in content, name


def test_the_test_path_makes_the_package_importable_without_installing(config):
    """The point of `pythonpath = ["src"]`: `pip install -r requirements-dev.txt` then
    `pytest`, with the package never installed. This test is running, so it worked —
    but the setting is easy to delete by accident, so pin it."""
    assert config["tool"]["pytest"]["ini_options"]["pythonpath"] == ["src"]


def test_a_child_interpreter_imports_the_source_tree_not_a_copy():
    """The "explore it in your own environment" path, end to end.

    A child process gets ``src/`` on its ``PYTHONPATH`` (see ``tests/conftest.py``) and
    must import *this* source tree and run an assessment. If it imported an installed
    copy instead, an edit to src/ would not be visible to the four subprocess tests, and
    they would be checking a different library from every other test in the suite.
    """
    from tests.conftest import run_child

    code = (
        "import numpy as np, themis_assess as ta\n"
        "from themis_assess.handlers.sklearn import CallableHandler\n"
        "handler = CallableHandler(lambda X: (X[:, 0] > 0).astype(int), name='rule')\n"
        "data = ta.ArrayAdapter(np.random.default_rng(0).normal(size=(40, 2)),\n"
        "                       feature_names=['a', 'b'], categorical_features=[])\n"
        "actual = (data.values()[:, 0, 0] > 0).astype(int)\n"
        "result = ta.AccuracyEvaluator(handler, metrics=['accuracy']).run(data, actual)\n"
        "print(ta.__file__)\n"
        "print(result['accuracy'])\n"
    )
    completed = run_child(code)
    assert completed.returncode == 0, completed.stderr
    imported_from, accuracy = completed.stdout.strip().splitlines()
    assert Path(imported_from).resolve() == (ROOT / "src" / "themis_assess" / "__init__.py")
    assert float(accuracy) == 1.0


# --------------------------------------------------------------------------
# the CI job that exercises the conda path
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ci() -> dict:
    import yaml

    return yaml.safe_load((ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8"))


def test_ci_actually_creates_the_conda_environment(ci):
    """environment.yml went through three revisions, two of them broken, and every
    breakage was found by a person running the file rather than by a check. This pins the
    job that closes that gap, so it cannot be quietly dropped."""
    jobs = [job for name, job in ci.items() if name != "stages"]
    creating = [
        job
        for job in jobs
        if any("micromamba create" in line for line in job.get("script", []))
    ]
    assert creating, ".gitlab-ci.yml has no job that runs `micromamba create`"
    script = " ".join(creating[0]["script"])
    assert "-f environment.yml" in script, "the job must use the committed env file"
    assert "--no-build-isolation" in script, (
        "the job must exercise the documented editable install, whose whole point is "
        "that --no-build-isolation makes it offline"
    )


def test_the_ci_job_and_the_env_file_agree_on_the_environment_name(ci):
    """``micromamba run -n <name>`` hardcodes the name that ``environment.yml`` declares.
    Rename the environment and the job runs against nothing, which surfaces as a
    confusing 'environment not found' rather than as a rename you forgot to finish."""
    import yaml

    declared = yaml.safe_load((ROOT / "environment.yml").read_text(encoding="utf-8"))["name"]
    jobs = [job for name, job in ci.items() if name != "stages"]
    referenced = {
        line.split("-n ", 1)[1].split()[0]
        for job in jobs
        for line in job.get("script", [])
        if "micromamba run -n " in line
    }
    assert referenced, "no job activates the environment"
    assert referenced == {declared}, (
        f".gitlab-ci.yml runs against {sorted(referenced)} but environment.yml declares "
        f"{declared!r}"
    )
