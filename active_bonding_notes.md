## Notes for ACF bonding progress, then recessed bonds, then elecated gold bonds, August-November

#### 27.8, first day
Also designed PCB holder, and Alexie optimized soldering. 

Got wafer. Tried for the first time. Took one device with normal ecoflex press, after quite high pressure started to show shorts, so works in principle. Used ALu + silver paint. detached.. midway, could not deliver ext signal. Used MEA24, over both pillar heights. But as i know shorts are first step, then external signal, then low impedance. And since it needed quite a bit not too promsing in this config. But this used small 35 um pads, so probably not "bending down". Should try again with larger ones: 61um 

Also then tried with soldered silver wire to a new device, with large pads, 61 um. Was hard to align. Used MEA22, with ACF as before. Tightend quite a bit, actually too much. Main issue: MEA22 bad signal, first though ACF + holder shorts somewhere, but chip did not recover so far... Maybe a copper wire pressed to hard on chip, and destryoed something? issue is that this film never lays just on active area. Should insulate Xpress top, and measure shorts in normal device, i am sure Titanium cap is grounded right?  could it be aligned better? And held in place? Assembly was also hard PI + wire constatnly moved, wire to stiff. Disassembled and saw that pressure was high, clear copper 50um imprints (nice and small actually) shown in interconnect PI, it ripped though, concerning. Well that 61 um leaves only a tiny PI gap, not that stable. 

Asked Fernando what could have happened with chip, what components are risked to be shorted. Should def check shorts myself, but on which chip? Basically ACF + chip should not impair anything, and i rmeember it always did actually... I even used insulation tape before when trying to attach wire with silver paint, it alwaays had this 20KHz signal no top. i could use 4984 well chip actually! has it's own press (old)

So, plan for today 28.: CHeck ich tit cap is shorted to GND. See if 4984 + ACF + ecoflex press affects signal, as pressure increases. Try tradiitional as well: attach a better wire to practisce for later, bond to MEA24. Check if you can hold on X press + capton tape for testing. . Further questions: can the AFC film be laser cut to exact shape? can it be held in position? and made flat?
---------


#### 28.8, 2nd day
Friday, with Asude. Farah's defense, long Friday evening. 

So first finding, titatnium holder is not grounded. Alexei said once it's in the headcap that whole thing is grounded, but is it? Yes it connects with the bone screws, but not with the Titatnum wire that is used for recording GND. Should ask Peter about it... 

In any case, if this piece of the entire holder connectes with the ACF, everything gets shorted to everything: bc remember that PI film gets shorted to one terminal. And ACF might contact Ref and Gnd ring electrodes on chip. I did check with 4984 well device and ecoflex press + ACF and found that even with higher pressure recording stays OK, even when ACF is contacting ring electrodes. Very good. Saw the same for MEA24 (using this one now for ACF tests): with higher pressure recording stayed ok, when using ecoflex between X press and ACF. Thought i saw contact once between Ref and Gnd, but cound't reporduce, prob not real. Signal was quite stable now. 

Putting film between them was again not easy. wire is much heavier than PI film, helped to use holders again, held at cable. First attempt slipped out again under press, not enough pressure. But pressure increases steeply now. At one point, screws have resistance. I need to be careful now, bc MEA1K22 broke with too high pressure I think. 

After retightening, no good connection unfortuantely. Signal again noisey, although holder not shorted, everything might be shorted to Gnd and Ref, which is not the normal setting to do the external signal stim. Should check noise level in real recordings: 2026
-08-28_21.22_4thTry_ACF_J2_61Pads_VRef15_PT2/ Checked it. It's messy, looks very different then old ones, something is off here... Without ACF film, we also short many things, eg if one electrode shifts onto ground ringnode, everything shouod be shorted to in this case. But less likely that this shift happens hm.. But yeah i remember recordings where this shifts happened and external signal 10mV was still ok as far as i remember. Left that evening doing shortcut stim, should check that too. Looked initially like not a lot connected, indicating that maybe the Film is deteriorated in same way, used it a lot now many times, including with breaking pressure on MEA22. So that's a good thing to try: simply with a new film go again, and check if the current one created imprints on PI film. Another theory is that these gold pillars are too high, uninutitive but who knows. I only ever tested on MEA22, which has much lower pillars than 24 and 23.

#### A few connections on top, but real?
![alt text](image.png)

#### More shorts, as would be expected by a film on the Chip
![alt text](image-1.png)

#### But impedance is not good, not real connection there it seems. Also not connection to GND ring (short)
![alt text](image-2.png)

#### REcorded impedance where shorts were present, but only a single one was low, all other high, hmmm
![alt text](image-3.png)

So. What do do with this. Disassemble carefully, checking how compressed the ACF was exactyl. THen check PI film and if it has imprints, hopefully not. If not, maybe bc of EcoFlex? I would simply retry with a new piece of AFC for now. As a side project, i could also try to pressure bond without film a 61um pad device to MEA23. OUt of curiosity. 
-------

Just want to note one optimistic thing: we saw this work in sprincple in ACF try10 right, even with the periodic mismatch where some spring pad region was conencted and then the next wasnt again, because of pitch mismatch. I saw it work! Actually this used real electrodes i remember now... so they were no shorted, maybe that's an imporant detail. i could go back to the spring pad devices with solder pads put in saline for optimization . hmm


#### 29.8, 3rd day
Saturday, wrote the suff above, measured impedance from home, wasn't good for 4th try. Went to lab in the evening, disassembled, and found devices was rupurted after 2nd row, therefore only connectivity in first row. ACF also not well placed. Then tried with 2 more. cut a smaller fresh piece of film so it doesn't touch rings. worked ok. but failed to press down twice. then got rid of EcoFlex and put capton tape instead. Finally, with a device that wasn't picked up well (initial ruptures) i soldered copper mesh, in hope it hold better, but stell maeh. but with lower profile of X press could align better, still not perfect but better: called 7th try. Signal quality over that whole evening was better, nothing too crazy. but with high pressure (PT2 on that Try) it showed similar noise pattern to before, but lcukily reversable. was quite scared. At PT1 was good, but also no connection. Only with PT2 some noise connections. Measured imp, was like 50% good, 50% bad.... not good. then disassmbled and found film had ruptured again at some location. Analysis:  the sheer force of the ACF destroyed the PI film! I think what happens is that the ACF wires hinge into the tiny recess of the bonding pad. THen as force increases the copper creates a sheer force on the film, and on the gold pillars. Gold can take it, but PI can’t as it is now. I tested this before designing the new film and also saw the the spring pads get ruptured, but i thought it's becuase they are generally fragile. I tried on flat piece of PI film, 3um, and that held up to high pressure. So th

Therefore followup: under mic using glass plate, compress ACF on PI flim, electrodes and no electrodes, maybe flip, and see if it damages things. Then practically for making connection, a crazy idea: MEA-PI-AFC change order. use ACF wires as pointy pressure to press down large metal pads. maybe this works.... could try first without ACF, with high pressure maybe also not terrible. I also want to make a transparent press, and i probably need to cut one screwdriver. Or maybe print one. 

Went in the evening. Starting by taking an old interconnection, and pressing ACF on it, later other way wround, ACF at bottom, interconnect on it. then pressed with glass slide under mic, I thought i can reproduce the sheer force, then rupture. But nothing happened. I could also not get imprints of ACF wires onto the pads, which i saw before on first try (promising). Was the film so dirty and used up maybe? Or pressure with hands much lower than with X press? was good to see how CF deforms though. it's non unfirom over it's surface, not like one uniform sheer in one direciton. 

Then i decided to try a new thing: just bond a large pad device with the old method: CMOS-PI-EcoFlex-Xpress. struggled with sodering... Did assembly almost without microscope, held flim with holder, then a drop of IPA, then press with thin EcoFlex 00-30. CHecked with external signal - very strong response! many many connected, also still shorted with liquid i assume. Started measuring impedance and saw ohmic connections, exciting. Then wanted to connect external current measurement to identify return current path. In this process soldered wire came off.... very frustrating, hacky on chip soldering of wire again, managed to get connection again on half the device (right side) 

![alt text](image-4.png)
![alt text](image-5.png)

Measured impedance over night. And oh boi, it looked good! patchy, but it seemed like one pad had a mix of high land low imp electrodes on them, so selecting a low imp one is feasable. The large pads may have helped? Would be good to know if smaller ones can work as well:) But not all good. there is a big area that was high imp, and it was associated with the last StimUnits in the sequence, indicating that something happened towards the end of the measurment. But not a clean cut StimUnit 29 already more then expected high imp electrodes, then 30 with even more, but still some low imp ones, so wire didn't come off i think. then 31 almost all bad. Then checkd again in the morning... and it didn't look good. Big deteriation on most sites, but enirely high imp, so wire is still on i think. What happened? Pressure is somewhat high i'd say. Hm should go and measure connectivity again, and try increasing pressure, measure with distance-measurment tool. Ok very interseting. In the measurement over night, the high imp electrodes at 29,30,31 stimunits, they have a phase shift! of about 50 degree's so they are really floating above the CMOS ? like very closeely giving capacitive connection? Perhaps something like this can happen as IPA evaporates! Ant now in the morning everything has evaporated! now most is bad. Hmm but in the new one in the morning, not all went to this high phase shift, more then before, actually also some no longer have this phase shift. Uh or there was still IPA in those electrodes in the evening? and the deterioaration this morning has a idfferent source? Hm.

![alt text](image-6.png)

![alt text](image-7.png)

Check next:  see what pins we have, 0.5mm ? assemble test jig with transparent press, no pcb in holder, visible pressing preocess, check ohms. Need to see imprints again, so strange. Model ACF to PI film in 3d. Buy rubber rings to replace springs. Deliver external 10mV sine to see what's going on with implant. Could even map single mea pillar gemoerty to it's impedance/ match on PI film/ how it affects it.



#### 30.8, 4th day
Just wrote up what's on top, didn't make a new assembly

#### 31.8, 5th day, Tuesday noACFTry2

Went to the lab in the morning, wanted to check the connectivity of the Sunday assembly. It was basically completely disconnected: 

![alt text](image-8.png)

I think the cable came on dueing handling... i disassembled it and don't remember spotting anything fishy. I think before disassmbly i tried to diagnose the noise in the external signal analysis. Took a screen shot too i think [put it]. But even with all grounding etc ccoulnd't fix. But overall RMS wasn't too bad i think. So i ignored it. Next devices didn't have it. It acutally turned out that i useed 00-10 EcoFlex on this assembly... didn't realize it. Perhaps the press got very close to the chip? And that somehow interacted with the signal? Who knows. EcoFlex showed big indent after disassmbly. don't remember looking at electrodes. 
So i could not retighten and check again unfortunately... 

SO i did a new assembly, this time with smaller pads, and placed on top part of MEA24 chip, which has lower gold pillars (!). THis time no IPA, just pressure. Used EcoFlex 00-30 as i should, again 1mm, so rather thin. Initial pressure showed some connection, which improved with more pressure (not well measured, spacers are not accurate, need torque control, ordered o rings, to remove springs, they are annoying), together with impedance again being good. I also tried the acrylmic press, but although i could see through, it was hard to learn something from it. The acrylic cracked, had to glue, a bit messy. could create imprintes on Film, though not unform, not in the center, also broke 1-2 pads, but not ruptured as we saw before. Also saw sheering [insert picture]
![alt text](image-15.png)
![alt text](image-16.png)

But more importantly, the new device kep impedance quite well, still deterioration, that has not slowed down yet (24 h have passed). We are at about 10% i would say. 

PT3:
![alt text](image-9.png)

THen after24h, and the after retightenting:
![alt text](image-10.png)
![alt text](image-11.png)

Timeline:
First TIghtenting:
![alt text](image-12.png)

THen 24h deterioaraiton, but at least slowing down: 
![alt text](image-13.png)

Then retightening improves again, plus some shift i suspect (also quuite a few got worse) (+small clibration issue, but not so bad)
![alt text](image-14.png)

Wednesday: How to move on? Again, increase pressure again to see how signal evovles, can imp recover back to good values? I ordered EcoFlex 50, maybe better. But i also consider dropping EcoFlex, just kapton tape, and see what happens. or use PDMS (stiff), but can't make mold right now without 3d printer... Bigger pads also better i think. I think i will assemble MEA23 today using big pad dummy device, without ecoflex, if pad, can still try again with ecoflex. 
Also, discuss el design. and talk to alexei.


#### 2. Sept, 6th day noACFnoEcoFlex

First, retightened noACFTry2 device, and impedance improved again see picture above. Reassuring. Perhaps it now won't deteriorate? Will check again tmr (thursday)

Then i tried with MEA1K23 + a 61um pad device (very dense) and withoutecoFlex, just Capton tape. Strange pattern after 2nd pressure increase: That doesn't seem to work
![alt text](image-17.png)
![alt text](image-18.png)![alt text](image-19.png)


All of them actually low impedance through....But was alreedy very tighy, so i cancled that try and added EcoFlex 00-30

This gave the best result yet! SO many low imp connections!!

![alt text](image-20.png)

Very funny pattern with these diagonals. I comapred with pillar hieght profile and it matches! high pillars --> good conenction, as we saw before. ANd high islands show this directly. also right side less pressure then left

But again deteriation aligned with pressure over time:
![alt text](image-22.png)
![alt text](image-23.png)


-------

#### 3. Sept, 7th day — [PRELIM, inferred from data — Simon to fill in]

I think i just remeaserd noACFTry4 no that day, the first time using MEA23 + J2 soldered device.
2026-09-03_08.46_noACFTry4_J2_61umP_PT2_Imp

Need to check, but i think saw first signs of pressure-decay related deterioaration of impedance.
#### 4. Sept, 8th day — PDMS press (MEA24) + new 2x8x56umP device (MEA23)

Two things in parallel.

**PDMS press, MEA24, 2x61umP+35umP device.** The stiff-press idea from the 31.8 notes. It failed
outright: PT1 connectivity had *zero* electrodes above 0.5, and the PT2 impedance run gave median
1.7 MΩ with 0.8 % below 200 kΩ. Nothing bonded at all. (Also left four 0-byte aborted Imp dirs at
16.52–16.55.) So PDMS as used here is worse than EcoFlex, not better — presumably too stiff to
conform to the pillar-height variation that the diagonal pattern on 2 Sept showed matters.

**New 2x8x56umP device on MEA23, PT1.** First bond at 09.55/09.59. PT1 gave 0.27 of electrodes
above 0.5 connectivity (0.17 above 0.8) and 0.475 below 100 kΩ — a moderate start. Median phase
was 49.6°, i.e. strongly capacitive, so at PT1 most of this is proximity/liquid coupling, not
metal-to-metal contact.

Also retightened noACFTry2 on MEA24 and measured it at +3d (07.52).

#### 7. Sept — [PRELIM] left sitting, +3d readout

Single recording, 22.43, the 2x8x56umP device at PT1 after 3 days untouched. It got clearly worse:
median 110 kΩ → 1.04 MΩ, frac <100 kΩ 0.475 → 0.368, and median phase fell 49.6° → 10.4°. The
phase drop is interesting — whatever capacitive coupling was there at PT1 has dried out or
settled, and it did not convert into resistive contact.

#### 8. Sept, 9th day — tightening series (MEA23) + 100 kΩ reference bond (MEA24)

**Morning, MEA23 2x8x56umP, after the group meeting.** Tightening series PT2→PT4. Connectivity
over all 26 395 electrodes:

| step | frac >0.5 | frac >0.8 | note |
|---|---|---|---|
| PT1 (4 Sept) | 0.27 | 0.17 | |
| PT2 (11.41) | 0.50 | **0.41** | "very good, but left side off" |
| PT3 (11.59) | 0.56 | 0.24 | retighten left, slightly bad on right |
| PT4 (12.09) | 0.54 | 0.26 | tiny tightening of bottom right screw, little worse |
| PT4 (12.30) | 0.60 | 0.17 | added more ringer |

This answers the "little worse, why?" in the notebook. The >0.5 count keeps going up while the
>0.8 count collapses — so past PT2 I am not making new real contacts, I am raising the weak-
connection floor. The ringer step makes it unmistakable: the median connectivity over *all*
26 395 electrodes goes to 0.77, which cannot be bonding (the device only has ~900 pads, covering
at most ~9000 electrodes at 56 µm over a 17.5 µm pitch). That is a global liquid shunt.
**Conclusion: stop using frac>0.5 as the yield metric for this device — it is dominated by ringer.
PT2 was the actual best mechanical state.** Impedance at PT4 does recover vs. the +3d state
(median 58 kΩ, 0.565 below 100 kΩ, phase 33.5°), so retightening helps, but the connectivity map
is overstating it.

Shorts: TileShorts at 02.16 (still PT1), then SingleShorts at 16.20 (aborted) and 19.46 (the real
run, 3039 files). Note both SingleShorts dirs are *named* PT1 but were recorded after the PT4
tightening — the PT in those names is wrong.

**Afternoon, MEA24, `test_100k` — this is the good idea of the week.** A known 100 kΩ resistor as
the load instead of a PEDOT site, on a 35 µm-pad device. That decouples "is the bond good" from
"what is the electrode impedance": a genuinely bonded pad *must* read ≈100 kΩ at ≈0° phase, and
anything higher is a bad bond. No more arguing about whether 4 kΩ is a great electrode or a shunt.

It validates cleanly — median 104–106 kΩ at 0.1–0.3° phase at every pressure level, so the
measurement chain is accurate. And the yield rises monotonically with pressure:

| step | frac reading <150 kΩ | median |
|---|---|---|
| PT2 (14.35) | 0.67 | 105.7 kΩ |
| PT3 (15.25, right tightened) | 0.81 | 103.8 kΩ |
| PT4 (16.44, left tightened) | **0.82** | 104.7 kΩ |

82 % of measured sites correctly reading the reference resistor is the cleanest yield number in
the whole ACF2 series, and unlike every other number above it cannot be inflated by a liquid
shunt. Two follow-ups suggest themselves: (1) re-run the 100 kΩ reference on the *ringer* device
to calibrate how much of that 0.60 connectivity is shunt; (2) leave the 100 kΩ bond sitting for
24 h — if it *also* deteriorates, the deterioration is mechanical/bonding, and if it does not,
then the deterioration seen everywhere else is electrochemical at the PEDOT interface, which
would be a completely different problem to fix.

Note the 8 Sept `test_100k` recordings are on **MEA24**, not MEA23 (the notebook had them under
MEA23), and the two connectivity runs are PT1 and PT2 (both were labelled PT4).

-------

### Appendix: recording-naming issues found 8 Sept (data audit)

Mislabelled at record time, worth remembering when reading old dirs:
- `2026-09-04_07.52_noACFTry2_J2_61umP_PT3_retightened_Imp` — says `61umP`, is the **36 µm** device.
- `2026-09-08_16.20.34` and `2026-09-08_19.46.49_2x8x56umP_**PT1**_SingleShorts` — recorded after
  the PT4 tightening; should be PT4.
- `2026-08-29_21.31.30_5thTry_ACF_J2_36umP_TileShorts_**PT4**_2mm` — 2 mm was PT2 (PT3 was 1.9 mm),
  so PT4-at-2mm is inconsistent.
- `2026-08-29_22.54`/`22.59_7thTry...PT1`/`PT2` and `2026-08-30_21.00`–`21.22_noACF...PT1` are
  impedance recordings saved **without the `_Imp` suffix**.
- Notes say the noACF device was disassembled on the 31st, but there is a `2026-09-01_08.43`
  connectivity recording and a `2026-08-31_17.56` Imp on it. Notes are probably a day off.

**Two measurement sweeps hidden in one directory — do NOT delete, and re-check the analysis.**
`2026-09-03_08.46_noACFTry4_J2_61umP_PT2_Imp` contains **two complete sweeps** of the same 7218
configs: the canonical `*.raw.h5` set written 08:46–~11:00, and a second `*_0.raw.h5` set written
11:22–19:28 the same day (the recorder appended `_0` because the files already existed). So there
is an extra, entirely unanalysed noACFTry4 timepoint (~+3 h after the 08.46 readout, ~+17 h after
the PT2 bond) sitting in there — 283 GB of real data, not duplicates.

Its `processed/all_impedance.csv` has only 7218 rows, so the processing used just one of the two
sweeps — almost certainly the 08:46 one, since it iterates the config `.csv` files and derives
`<base>.raw.h5`. Meanwhile `_get_hdf5_fnames_from_dir()` in `analyze_extSine_connectivity.py`
globs on `fname.endswith('raw.h5')`, which *would* pick up both sets and interleave them. So:
- the "+14 h" number quoted above is one sweep only, and is fine;
- but anything that goes through the `endswith('raw.h5')` path on this directory is mixing two
  timepoints. Worth splitting the `_0` set into its own dir with a proper `+Nh` name.

No implant dir has been created for any of the ACF2 / noACF / 2x8x56 work — newest
`implant_devices/` entry is still `260715_*`. Everything since is headstage-recordings only.


Future (11.9)
Assembled with 00-50 EcoFlelx yesterday (ZNZ) for the first time, and i think i see earlier connection with screw pressure (test2). I also tried open, move interconnect close test with MEA24 test1 (first consitent nt good results using soldered PCB), but that tidn't work, i think the EcoFLex becomes super thinwith / during pressing, repressing hits a different part of ecoflex --> not good contact. This was called test3, will now reassembled with fresh00-50 EcoFlex ...annoying... shorted to ring i think... no 1KHz... tried again with new alignment, same EcoFlex00-50 ... ffs again nothing, wrong side i think -.-.... still nothing! film was very scrabled at this point to be fair... annyoing. Will need to do one before i will put the real one.



----

22.10.26

Ok, sorry looks like i didn't write for a very long time. Basic timeline was that during ZNZ week (around 7.9.26) we tried to solder PCBs to interconnects, and failed. In that same week, i did many assemblies, up to test9 i think. This turned into bonding figure. ALways used MEA1K24 and 23. 23 has this patterned elvation, 24 doesn't bond on top, and not as well on general. But overall i could get very good bonding results, 6 samples 83% average. Then greece holiday. When i came back i think i did only 1 attempt, were soldering failed? Don't quite remember if i really did? Must have been late September or early October using 24 or 23. Actually 24 is still assembled i think, so 23 if anything. Then didn't do anythink for a bit, postproc new batch of chips (seee below). Then wanted to check goldplating, a few shorts unfortuatnely, Did test10 i think. But soldeirng was very bad, inconsitent results. Then yesterday i did test11, which is not tightented yet, but looked good, again with MEA1K27
Different behavior from before, did the 35 minutes make a difference? High imp, good connectivity results i have not seen before, looked like capacitipve bond at mid pressure. only high pressure fixed it for the most part. Need to profile. This should be moved to a new ipynb file , it's a new batch now. 

I also worked quite  abit on trying to understand the calibration better using current meter from alexei and well_devices/8249 . good for the most part, but there is about a 15% error... Need to look into this


On batch processing of MEAs. Since these had 1mm bigger PCB we needed new Tit holders. This time tolerances were crazy, hard to place, only later found good method with filing. I broke on chip by crashing in wirebonding while screwing: MEA1K26. MEA1K25 i did together with ALexei (first) Placement in holder was ok, but epoxy failed in strange way: bubbles, and too high and unstable, ALexei put 2ml. He tried with dremel to release and broker everything. Realized i need to take more responsibility. I did the other 7, one (as i said above) broke by wirebond crash - but 6 looked good. Updated also Goldplating form. On 10th of October we did goldplating for MEA1K27, 35 minutes. Lots of bubbles... constant brush needed... But surface looked good! i tried to clean before, but it was still dirty... i needed brush to fix it, and that course one left stain-like scrubs - bad. don't use again. cleaned with IPA and acetone. But there were a few shorts this time - unusual. I think we should do 30 minutes again. There were like gold crumps that could be pushed away with the brush - perhaps they fell out of solution? In anycase. will goldplate MEA28-32 soon:)

BTW here i figured out the calibration of DAC units for good, publ. ready figure. But proba should check with real device once as well to compare this 15% too low voltage result.

30.10.26 

New devices done fabricated that Friday. ALso electrodes a few days before that. Did a bonding test that evening called test12. i used an old recessedPad, J2 wafer device with 36umP on top, and bottom a new eelevated gold  device with large 56um pads (10um gap). Had issues measureing them bc the flexPCB cable wasn't properly attached... Fixed this only on Sunday (1.11.26). But already ran short analysis showing that elevted gold pads connect well and before recessed pads on top. But there can be shorts on x axis see below. But biggest issue was that there where only 3 islands that had connection with external signal and impedance. One island was shorted. It's not clear whaat happened. Perhaps via etching connecting metal1 to metal2 was not good enough. This enables better interpretation of shorts though: since they are really not connected (shorted to each other by distal wire) if they show up in sequential tile stim, they are really conencted on via chip surface (gold bumps). Therefore island 3 is likely also shorted, and only one via is good there. Top impedance was mediocre too, right pressure tild didn't help. My impression of MEA1K27 is that not only left and right side are lower in gold, but top side also bonds less well then bottom side. ToDo try to check if you can spot issues on the devices by eye. I will likely bond another device from this wafer and see. Maybe in the center its' better? 

![alt text](image-24.png)