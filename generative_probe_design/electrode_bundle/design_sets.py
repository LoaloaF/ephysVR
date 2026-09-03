r"""The probe designs to build, one DesignSpec each. `batch.py` builds every entry.

ALONG THE SHANK -- what each length parameter measures
======================================================
Tip at the LEFT, connector at the RIGHT. Every bar below is measured along the shank axis.
Numbers are design "C1", in um. Parameter names are exactly the DesignSpec fields;
(parenthesised) quantities are derived, not inputs.

        loop           deepest        shallowest      shoulder          first pad
         |               |               |               |                  |
         |---------------|---------------|---------------|------------------|
         |  loop_offset  |  sites.span   |  free fiber   |    (ribbon)      |
         |      445      |     5790      |     2765      |     5406.5       |
         |               |               |               |                  |
         |<------------ fiber_length = 9000 ------------>|                  |
         |               |               |               |                  |
         |<------------------- overall_length = 14406.5 ------------------->|

    loop_offset  +  sites.span  +  (free fiber)  =  fiber_length
        445      +     5790     +      2765      =      9000

                        fiber_length  +  (ribbon)  =  overall_length
                            9000      +   5406.5   =     14406.5

Only `loop_offset`, `sites.span`, `fiber_length` and `overall_length` are inputs. "free
fiber" is whatever is left above the shallowest site; "(ribbon)" is whatever is left above
the shoulder. So lengthening a design's fiber shortens its ribbon by the same amount and
the device still ends at `overall_length` -- which is the point: probes with different
fibers still meet the PCB at the same place. `overall_length=None` pins the ribbon at the
connector's default instead and lets the total float.

ACROSS THE SHANK -- where each channel sits
===========================================
Looking along the axis. All n_channels sit in ONE row across x, alternating sides with the
index growing OUTWARD from the centre: ch0 takes the centre slot (it is the one carrying
the loop), odd channels go to +x, even channels to -x.

              -x  <---------------- x = 0 ----------------->  +x

                   ...  ch4   ch2   ch0 | ch1   ch3   ch5  ...
                         #     #     #  |  #     #     #
                         |<--->|        |
                         delta_x = 24   | ch0 is widened to carry the loop, so its two
                                        | neighbours sit 29 um out, not 24

     solder pads:  [o] [o] ... at x = -975      and  [o] [o] ... at x = +975
                   (pad_row_pitch/2)                 33 per column: 32 wired + 1 REF/GND

THE FIELDS
----------
    n_channels      how many recording sites
    sites           where they sit -- Uniform(span=...) or Segments([(ch, ch, pitch), ...])
    fiber_length    loop -> shoulder
    loop_offset     loop -> deepest site
    overall_length  loop -> first solder pad; None keeps the connector's default ribbon
    magnetic_id     IONP barcode 0-15, or None for no barcode
    doc_id          the source doc's own row name, for provenance

`fiber_length` and `loop_offset` are solved for numerically (lengths.py); `sites` resolves
to delta_y/delta_y_overrides (spacing.py). `magnetic_id` stays None unless the device
really is getting an IONP barcode -- stamping an unplanned ID would be inventing
device-identification data.

THE DESIGNS BELOW
-----------------
The 8 are the design-doc rows that use this generator's connector footprint (2-row round
pads, the doc's "B" family). Rows we cannot build: tetrode layouts, 32-channel devices (the
pad stack here is fixed at 64ch), and the "A"/"C" connector footprints.

Where a doc row states both a span and a pitch, the SPAN is authoritative -- the doc's own
generator takes span as its input and derives pitch = span/(n-1). The three Segments
designs are the exception: their stated pitches sum to a span that disagrees with their
own doc ID (C3 6655 vs 6600, C4 4887.5 vs 4750, H1 6415 vs 6400), unresolved pending the
doc's author, so those keep the stated pitches.
"""
from dataclasses import dataclass
from typing import Optional, Union

from .spacing import Segments, Uniform


@dataclass
class DesignSpec:
    name: str                                 # short handle; names the output files
    n_channels: int
    sites: Union[Uniform, Segments]
    fiber_length: float                        # um, loop -> shoulder
    loop_offset: float                         # um, loop -> deepest site
    #um, loop -> first solder pad, i.e. the whole device. The ribbon (shoulder -> first
    #pad) is whatever is left over, so probes with different fiber lengths still end at the
    #same connector. None = keep the connector's default ribbon and let the total float.
    overall_length: Optional[float] = None
    magnetic_id: Optional[int] = None          # IONP barcode 0-15; None = no barcode
    doc_id: str = ""                           # the source doc's own row name


#Every design in this family ends at the same connector position, so the ribbon absorbs the
#differing fiber lengths instead of the whole device changing length. Value = the longest
#fiber here (H1/H2, 10600) + the connector's default 3806.5 um ribbon, so those two keep
#exactly the default ribbon and every shorter probe gets a longer one.
FAMILY_OVERALL_LENGTH = 14_000


DESIGNS = [
    DesignSpec(name="C1", n_channels=64, sites=Uniform(span=5790),
               fiber_length=9000, loop_offset=445,
               overall_length=FAMILY_OVERALL_LENGTH,
               doc_id="64-lin-B-5790-9000 (C1)"),
    DesignSpec(name="C2", n_channels=64, sites=Uniform(span=5990),
               fiber_length=9840, loop_offset=445,
               overall_length=FAMILY_OVERALL_LENGTH,
               doc_id="64-lin-B-5990-9840 (C2)"),
    DesignSpec(name="C3", n_channels=64,
               sites=Segments([(1, 42, 90.0), (42, 43, 970.0), (43, 64, 95.0)]),
               fiber_length=10350, loop_offset=445,
               overall_length=FAMILY_OVERALL_LENGTH,
               doc_id="64-lin-B-6600-10350 (C3)"),
    DesignSpec(name="C4", n_channels=64,
               sites=Segments([(1, 26, 30.0), (26, 31, 250.0), (31, 64, 87.5)]),
               fiber_length=8440, loop_offset=600,
               overall_length=FAMILY_OVERALL_LENGTH,
               doc_id="64-lin-B-4750-8440 (C4)"),
    DesignSpec(name="H1", n_channels=64,
               sites=Segments([(1, 13, 110.0), (13, 33, 30.0), (33, 42, 355.0),
                               (42, 62, 40.0), (62, 64, 250.0)]),
               fiber_length=10600, loop_offset=340,
               overall_length=FAMILY_OVERALL_LENGTH,
               doc_id="64-lin-B-6400-10600 (H1)"),
    DesignSpec(name="H2", n_channels=64, sites=Uniform(span=5590),
               fiber_length=10600, loop_offset=345,
               overall_length=FAMILY_OVERALL_LENGTH,
               doc_id="64-lin-B-5590-10600 (H2)"),
    DesignSpec(name="980-6500", n_channels=64, sites=Uniform(span=980),
               fiber_length=6500, loop_offset=550,
               overall_length=FAMILY_OVERALL_LENGTH,
               doc_id="64-lin-B-980-6500"),
    DesignSpec(name="3140-8970", n_channels=64, sites=Uniform(span=3140),
               fiber_length=8970, loop_offset=420,
               overall_length=FAMILY_OVERALL_LENGTH,
               doc_id="64-lin-B-3140-8970"),
]
