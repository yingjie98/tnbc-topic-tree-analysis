from __future__ import annotations
from pathlib import Path

OBSERVED = "03_hypergraph"
WEIGHTED = "04_weighted"


def artifact_part(part: str, stem: str) -> str:
    if part not in (OBSERVED, WEIGHTED):
        return part
    if stem.startswith(("23a_", "23_图24标记结构_", "Fig3H_结构网络_原始")):
        return OBSERVED
    if stem.startswith("Fig3H_结构网络_"):
        return WEIGHTED
    if stem.startswith(
        (
            "20a1_",
            "22a_",
            "22b_",
            "22c_",
            "22_跨来源结构_结构显示清单",
            "22_探索性质量加权",
            "23_",
            "23b_",
            "23c_",
            "26_",
            "27_",
            "28_S_JCR_IF",
            "Fig4A_",
            "Fig4B_",
            "Fig4C_",
            "Fig4D_",
        )
    ):
        return WEIGHTED
    return part


def artifact_path(path: Path) -> Path:
    path = Path(path)
    parts = list(path.parts)
    index = next(
        (i for (i, value) in enumerate(parts) if value in (OBSERVED, WEIGHTED)), None
    )
    if index is not None:
        parts[index] = artifact_part(parts[index], path.stem)
    return Path(*parts)


class ArtifactDirectory(type(Path())):
    def __truediv__(self, filename):
        return artifact_path(Path(self) / filename)

    def mkdir(self, mode=511, parents=False, exist_ok=False):
        Path(self).mkdir(mode=mode, parents=parents, exist_ok=exist_ok)
        if self.parent.name == OBSERVED and self.name in ("tables", "reports"):
            (Path(self).parent.parent / WEIGHTED / self.name).mkdir(
                mode=mode, parents=True, exist_ok=True
            )


def artifact_directory(path: Path) -> ArtifactDirectory:
    return ArtifactDirectory(path)
