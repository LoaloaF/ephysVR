# <repo name>

## Device context — read before reasoning about hardware

This repo drives a MaxWell MEA1K CMOS array bonded to custom polyimide electrode devices.

- `docs/device_context.md` — established facts, every line tagged with provenance.
  Read it before reasoning about geometry, routing, bonding, shorts or impedance.
- `docs/device_open_questions.md` — untested hypotheses and open design questions.
  **Not facts.** Read when working on the relevant decision; never cite as established.
- `docs/refs/` — the primary sources the tags in those files point to.

**Two hard rules:**

1. **Polyimide-side geometry varies between devices** — pad shape, pad count, routing,
   shank count, and some devices deliberately short all pads together. `device_context.md`
   §3 has the ranges; no single design is the default.
2. **Terminology is load-bearing.** *MEA1K electrode* = a Pt site on the CMOS array.
   *pad* = a contact site on the polyimide interconnect. *PEDOT electrode* = a brain-facing
   recording site. The papers use these words differently; the code and these docs do not.
