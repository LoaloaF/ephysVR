"""Where the recording sites sit along the shank.

Two ways to say it, one type each -- pick whichever the source spec states:

    Uniform(span=5790)          sites evenly spread over 5790 um, tip to shallow end
    Segments([(1, 42, 90.0),    explicit pitch per run of channels, as design docs
              (42, 43, 970.0),  write them
              (43, 64, 95.0)])

Both resolve to the knobs `BundleConfig` already has (`delta_y`, `delta_y_overrides`) via
`.delta_y_kwargs(n_channels)`, so nothing downstream needs to know which one was used.

SPAN, NOT PITCH, is the primary handle for a uniform design: pitch is span/(n-1), and a
doc that quotes both states the span exactly while rounding the pitch for readability.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Uniform:
    """Sites evenly spaced so the array spans exactly `span` um."""
    span: float

    def delta_y_kwargs(self, n_channels: int) -> dict:
        return {"delta_y": self.span / (n_channels - 1), "delta_y_overrides": None}

    def span_um(self, n_channels: int) -> float:
        return float(self.span)


@dataclass(frozen=True)
class Segments:
    """Per-run pitches, in the notation design docs use.

    Each run is `(start_ch, end_ch, pitch_um)` with 1-INDEXED, INCLUSIVE channel labels,
    e.g. "Ch 1-42: 90 um, Ch 42-43: 970 um, Ch 43-64: 95 um" ->
    [(1, 42, 90.0), (42, 43, 970.0), (43, 64, 95.0)]. Adjacent runs SHARE their boundary
    label (42 appears in both the first and second run above); that shared label is what
    decides which run owns the gap across it, so no extra convention is needed -- run
    (X, Y, P) sets the gaps X..Y-1 to P.

    `rest_pitch` fills any gap no run covers, for docs written as "Ch 1-40: 40 um, rest
    120 um pitch"; without it the runs must cover every gap.
    """
    runs: List[Tuple[int, int, float]] = field(default_factory=list)
    rest_pitch: Optional[float] = None

    def delta_y_kwargs(self, n_channels: int) -> dict:
        #`delta_y` still has to be set even though every gap is overridden: build_bundle
        #seeds the staircase at cy_0 = -delta_y, so leaving it at the class default would
        #rigidly offset the whole array by a number this design never mentions. The first
        #gap's own pitch is the seed that makes Segments and Uniform agree.
        overrides = self.overrides(n_channels)
        return {"delta_y": overrides[1], "delta_y_overrides": overrides}

    def overrides(self, n_channels: int) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for start_ch, end_ch, pitch_um in self.runs:
            for gap in range(start_ch, end_ch):
                out[gap] = pitch_um

        if self.rest_pitch is not None:
            for gap in range(1, n_channels):
                out.setdefault(gap, self.rest_pitch)

        missing = set(range(1, n_channels)) - out.keys()
        if missing:
            raise ValueError(
                f"segments leave gaps {sorted(missing)} unspecified (no rest_pitch given)")
        return out

    def span_um(self, n_channels: int) -> float:
        return float(sum(self.overrides(n_channels).values()))


SiteLayout = (Uniform, Segments)
