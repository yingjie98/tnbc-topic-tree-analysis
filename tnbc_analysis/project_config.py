from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Calendar:
    observation_start: int = 2001
    observation_end: int = 2025
    cohort_start: int = 2007
    history_start: int = 2007
    snapshot_ends: tuple = (2010, 2015, 2020, 2025)
    forecast_start: int = 2026
    forecast_end: int = 2030
    training_cutoffs: tuple = (2010, 2015, 2020)
    emerging_years: int = 3
    recent_years: int = 5
    structure_start: int | None = None
    structure_end: int | None = None

    @property
    def forecast_years(self):
        return self.forecast_end - self.forecast_start + 1

    @property
    def recent_start(self):
        return self.observation_end - self.recent_years + 1

    @property
    def forecast_slot(self):
        return f"{self.forecast_start % 100:02d}-{self.forecast_end % 100:02d}预测"

    @property
    def history_slots(self):
        starts = (self.history_start, *(v + 1 for v in self.snapshot_ends[:-1]))
        return {
            e: f"{s % 100:02d}-{e % 100:02d}区间"
            for (s, e) in zip(starts, self.snapshot_ends)
        }

    @property
    def snapshot_labels(self):
        return {e: f"{self.history_start}–{e}" for e in self.snapshot_ends}

    @property
    def primary_label(self):
        return f"primary_{self.forecast_years}y"

    @property
    def emerging_label(self):
        return f"emerging_{self.emerging_years}y"

    def key(self, text):
        import re

        replacements = {
            "2001_2025": f"{self.observation_start}_{self.observation_end}",
            "2007_2025": f"{self.history_start}_{self.observation_end}",
            "2021_2025": f"{self.recent_start}_{self.observation_end}",
            "2026_2030": f"{self.forecast_start}_{self.forecast_end}",
            "current_2025": f"current_{self.observation_end}",
            "reaches_2025": f"reaches_{self.observation_end}",
            "to_2025": f"to_{self.observation_end}",
            "before_2025_cutoff": f"before_{self.observation_end}_cutoff",
            "26-30预测": self.forecast_slot,
        }
        return re.sub(
            "|".join(
                (re.escape(k) for k in sorted(replacements, key=len, reverse=True))
            ),
            lambda m: replacements[m[0]],
            text,
        )

    def text(self, text):
        import re

        replacements = {
            "2001": self.observation_start,
            "2007": self.history_start,
            "2010": self.snapshot_ends[0],
            "2015": self.snapshot_ends[min(1, len(self.snapshot_ends) - 1)],
            "2020": self.snapshot_ends[min(2, len(self.snapshot_ends) - 1)],
            "2021": self.recent_start,
            "2025": self.observation_end,
            "2026": self.forecast_start,
            "2030": self.forecast_end,
        }
        return re.sub(
            "(?<!\\d)(?:2001|2007|2010|2015|2020|2021|2025|2026|2030)(?!\\d)",
            lambda m: str(replacements[m[0]]),
            text,
        ).replace("26-30预测", self.forecast_slot)

    def validate(self):
        if not self.observation_start <= self.history_start <= self.observation_end:
            raise ValueError("Invalid observation/history years")
        if (
            tuple(sorted(set(self.snapshot_ends))) != self.snapshot_ends
            or not self.snapshot_ends
            or self.snapshot_ends[-1] != self.observation_end
        ):
            raise ValueError("Snapshot years must increase and end at observation_end")
        if self.snapshot_ends[0] < self.history_start:
            raise ValueError("Snapshot precedes history_start")
        if self.forecast_start != self.observation_end + 1 or self.forecast_years < 1:
            raise ValueError("Forecast must follow observation")
        if self.recent_years < 1 or not 0 < self.emerging_years <= self.forecast_years:
            raise ValueError("Invalid recent/emerging windows")
        if (
            not self.training_cutoffs
            or tuple(sorted(set(self.training_cutoffs))) != self.training_cutoffs
        ):
            raise ValueError("Training cutoffs must increase")
        if any(
            (
                v < self.observation_start
                or v + self.forecast_years > self.observation_end
                for v in self.training_cutoffs
            )
        ):
            raise ValueError("Training labels extend outside observation window")


TIME = Calendar()
PROFILE = {}


def configure(args):
    global TIME, PROFILE
    names = Calendar.__dataclass_fields__
    TIME = Calendar(**{k: getattr(args, k) for k in names if hasattr(args, k)})
    TIME.validate()
    PROFILE = {
        "tree": str(args.tree),
        "papers": str(args.papers),
        "output": str(args.output),
        "model_dir": str(args.model_dir or ""),
    }
