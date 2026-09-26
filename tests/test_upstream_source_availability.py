"""GitHub source availability must not depend on historical content hashes."""
import io
from pathlib import Path
import tarfile

import pytest

from rift import radar_fields_upstream as radar_fields
from rift import radarsplat_release as radarsplat
from scripts.fetch_radarsplat_reference import materialize


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_radar_fields_accepts_available_source_without_content_checks(tmp_path, monkeypatch, newline):
    monkeypatch.setattr(radar_fields, "REFERENCE_ROOT", tmp_path)
    for name in radar_fields.SOURCE_FILES:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("# GitHub source" + newline).encode())
    radar_fields.verify_sources()
    (tmp_path/radar_fields.SOURCE_FILES[0]).unlink()
    with pytest.raises(RuntimeError, match="needs the upstream source file"):
        radar_fields.verify_sources()


def test_radarsplat_checks_files_but_ignores_historical_digest_values(tmp_path, monkeypatch):
    monkeypatch.setattr(radarsplat, "reference_contract", lambda: {
        "files_sha256": {"model.py": "historical-unused-value"},
        "glm_files_sha256": {"glm.hpp": "historical-unused-value"},
    })
    (tmp_path/"model.py").write_text("# directly fetched source\n")
    assert radarsplat.verify_reference(tmp_path) == tmp_path
    with pytest.raises(RuntimeError, match="Missing GLM source"):
        radarsplat.verify_reference(tmp_path, cuda_dependencies=True)
    glm = tmp_path/"gsplat/cuda/csrc/third_party/glm/glm.hpp"
    glm.parent.mkdir(parents=True)
    glm.write_text("// directly fetched GLM\n")
    assert radarsplat.verify_reference(tmp_path, cuda_dependencies=True) == tmp_path


def archive(content):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as output:
        member = tarfile.TarInfo("upstream-commit/model.py")
        member.size = len(content)
        output.addfile(member, io.BytesIO(content))
    return stream.getvalue()


def test_fetcher_accepts_github_archive_and_preserves_existing_source(tmp_path):
    inventory = {"model.py": "historical-unused-value"}
    materialize(archive(b"print('upstream')\n"), tmp_path, inventory)
    assert (tmp_path/"model.py").read_text() == "print('upstream')\n"
    materialize(archive(b"print('another checkout')\r\n"), tmp_path, inventory)
    assert (tmp_path/"model.py").read_text() == "print('upstream')\n"
