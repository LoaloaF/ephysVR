# 3 major domains: 
1. measuring impedance reliably
2. logistics, devices
3. mechanics of bonding

---

## ① • Large initial progress with external current measurements, but some questions remain:

* **Q.** Chip 24 doesn't deliver good stable currents anymore. Did it degrade?
    * $\rightarrow$ calibrate again using many amplifiers, saw trend there <font color="red">--> did it with 2, and still didn't look too good. SC was ok.But havene't seen good data from it</font> I think it recovered for some reason. 


* **Q.** When chips stim units drift, or DAC code is off, not only current is lower than what chip says (common), also amplifier gives stereotypical amplitudes. Only potentiostat *[James]* completely takes MEA out of equation.
    * This remains poorly understood. [SOLUTION: <font color="blue">TODO</font> CHECK CIRCUIT, amplifier made for this? voltage divider like aminhan drew for INTAN?] I don't see such things in calibration settings, there also, shift in DAC have no effects on amplifiers. I know from bond2 that when DAC not centered, chip delivers less current than what it should, but also the amplifier gives stereotypical amplitudes, in the case below, very high for example, although real current low. When DAC is shifted, imp 1 and 2 match (evidance for "real" impedance measurement) ![alt text](image-1.png)
    ![alt text](image.png)
* **Q.** Is small current mode stable? Less variant than large current? 

    * $\rightarrow$ On MEA23 4_4 shank, calibrate carefully, measure with external confirmation. Compare. <font color="gree">--> looked good, not less variable, but less risky i think</font>
    * <font color="blue">TODO</font> stress test single shank, check stability first, then shift DAC, measure if DC offset is there, sweep. Do final confirmation that this can kill electrode (LC, shift DAC, check if still low imp, continue)

* **Q.** <font color="blue">TODO</font> Sidequest: Can calibration be sped up with more continuous DAC shifting? Also record current externally to confirm lack of bias. Generally understand this better, Alexei adjusted bias. ALSO <font color="blue">TODO</font> See if you can spped up DAC shifting procedure, measure real current AND measure phase shift stuff

* /resolved **Q.** Unresolved data from MEA 16 23: external signal on shunted electrodes has large cross impact. $\rightarrow$ check charts on Shank 1-4 MEA23, measure. 

* /resolved **Q.** Shorts remain strange. Reanalyze, & try current vs voltage mode & current to GND versus not (what I always did so far)

* Shorts: last step needed here is to compare "expected shorts" with real ones. Fun task for some evening <font color="blue">TODO</font>

* NEW: How to configure chip for Potentiostat measurements. Check goldplating script, understand circuit. Did thos on the weekend, led to also understading amplifier circuit better, together with Claude and PhD thesis. Fed the old MEA software script to claudecode. Tried to connect MEA electrode to stim buffer as current sink, but it rather clipped it? Saw this in vivo before, when stim unit connected, no more signal on amplifier. Because stim unit is holding it at steady potential?  Could try this again with reference not in bath, maybe that changes things? Connect Ref-Gnd.

* Switched quite a bit to external current approach, related to potetiostat too, need to understand circuit for this too. Did this first on Chip24 over weekend, and then also the high pressure one, MEA23. To look at this again, would check 23 data, have ext current all current single and, voltage. Need to repeat singles measurement. ALso have imp for this one.

* Other big interest remains: difference bewteen calibration setting and real setting: Go back to 4983 and gimp image. Then solder 1M resistor to new devices and try to reproduce. Does current leak to other MEA electrodes? 

* Sped up calibration a lot, and resolved some strange things that i did before, would say that i understand parasitic current better now.

* Digged alot into current path again, not fully resolved still, but helpful to some degree, multimeter analogy, shunt R, capcitance.

* Did a bond with soldered device, all shorted, and found interseting patterns. WOuld have expected more shorts bewteen pads, but just the usual pattern. Saw how single pads have very low impedance when touching GND ring. Could tighten scrwewns more and check how imp develops. Overall, impedance surprisingly high... ALthough just soldered cable at long end, measured fron interconnect to solderpad next day and is in low KOhm range.  Impedance was 200K for many, how could this be? One last interesting thing i did, connect 2 shorted pads, one one one DAC, one on another that served as current sink. And there, amplitude was lower, as if the shank+solder is not a low imp current path?
---

## ② • LOGISTICS

* **3+1+1** $\rightarrow$ PEDOT:PSS on 2x $\rightarrow$ Daniel/Baran $\rightarrow$ needed for testing + re-etch 

    * $\Rightarrow$ this week Baran <font color="green">done</font> Batch6: only one 8sh device left, but this batch 6 was bad according to Shubham. 
* **3+1** $\rightarrow$ re-etch on 1x first, then 1.7 try again
    * $\Rightarrow$  <font color="green">good</font> Wafer7 fixed, check PEDOT:PSS under mic. 8 and 9 need to go thrugh the same.  
* **connector masks** $\rightarrow$ Netherlands company
    * $\Rightarrow$ needed next week <font color="green">dane, fabricated:)</font>
* **connector PCBs** $\rightarrow$ Needed in 2 weeks, Alexei [push] Ask how long againagaignagin, now James <font color="green">submitted</font>
* **Depth/profile measur.** $\rightarrow$ <font color="green">mt:)</font> Cleanroom+Alexei / Houmam / Irchel / Eminhan

* **MEA headstage 16 PCB** <font color="green">submitted</font> $\rightarrow$ Alexei $\rightarrow$ China ||$\rightarrow$ Fernando $\rightarrow$ Alexei 

*James: New interconnect with simpler pads? Electrodes needed, plus geometry...

##  BONDING Grrr
* EcoFlex 00-10 , made thin film by luck, cut by hand, good, should have smae for 00-30, but there, have some left I think? Or even from Acylic mold. Yes have nice and thin one.
* Attack angle: understand geometry: confocal images of new bonding, maybe even of MEA23, 4_4 device? Can you spot the short to GND? Can you see the super high pressure?
* Then there is MEA24, with 6 shanks down on old MEA24 gold pillars (which were maeh before) and top, scaped off gold 8 shank device. Pressure profileing with impedance + shorts needs to be done here. 
* OR just f* bond a new one to MEA23 with high pressure and see what happens...

- bonding: gold irregular, show mic images. Need profilemeter, Alexei will get intro in mjuly, new chips with Pt in already potential new path

- will just try again and again, new devices

- connector stuff 

- new Ecoflex

* puh... shorts are again highly pressent in new 14shank 22 bond, this is getting very consitent. we are pinching the traces that are routed out from the interconnect. Maybe onto the passivation bump? Or in other case to GND. Would need profilemeter to confirm. +mic images.

* one idea was to measure Chip23 one last time, confirm the low impedance. And then setup current mode stim from external device? similar to Poteiostat, just indepdendent. 

* onther idea was to disassemble 24 board, scrape other half (left) and goldplate again higher? or lower?

* other paths are plasma cleaning? and conductive films



## Bonding history of each chip (redundant with each chips notebook).

### MEA24:

* Early May: I think this was the **Bond1**, 260501_MEA1K24_S1688pad14shankB5. This was a wafer from Shubham, hist first or second? Put silk and everything, Had hope. Did a lot of shortcut stim on tiles before, recorded 50Hz voltage patterns, kept dry for long. Pressure high, but not very high if i remember correctly. Imp not good. Especially metal2 not good, later found reason for it. And deterioarted over time. Measured this like crazy, so much, tried to understand it. Single shank 13 and 14. Found suspicious shorts that now i would say are connected to GND ringnode. Deterioaration was linked to to then silver electrode, and zeroDACcode drifting/ being off. Wrote DAC drift code. Or cases where LC mode + amplitude 30, super high currents. In the end (22nd-31st of May) got current sense circuit to work with ALexei. FOund that tiny DAC shifts influence measurements a lot. Imp can have constant value offset (good case), fully match (perfect) or be completely off (no match)  ![alt text](image-2.png) ![alt text](image-3.png) 

* 10.6. - 16.6 Extremly unsure what this is, called **Bond3** in recordings dir, looks like i realigned this a bunch of times. Looks very wet in voltage map. No idea what electrodes these are, must be wafer 5 again?. Did i try to recover those? In ultrasonic cleaner? Didn't work , remember that ![alt text](image-4.png).

* 17.6 New bond, now called **Bond4**, last one till now. **scraped gold off of top part** of chip to check if lower gold can help (it didn't). Connected the wrong cable with this one -.- one day lost, next day, found capctitive connection for the first time, after doing it, but for the half with pillars. Single MEA electrodes connected. No silk on those electrodes. Tried to do proper depth prifle. Used interconnects with some error, Shubham said this wafer (6) is not that good, but Baran put PEDOT. With high pressure, found some electrodes also connceted on scraped side. But, unfortuatnely, over all pressures, imp stayed high... Used 00-10 for the first time, thought needs lower pressure, but not really i think. Clean surface is the key to an early bond. Probably not electrodes, rigth? SHould have checked better.. Also struggled wi    th StimUnit calibration, had to manuallt shift by 20 DAC units (SC) that aligned external and internal current somewhat, but still results looked quite bad. These amplifiers are sooo off? Only Unit14 had bimodal reasonable things, suggesting there is a good connection?  ![alt text](image-5.png)![alt text](image-6.png) 
![alt text](image-8.png)![alt text](image-9.png)
![alt text](image-7.png)

* Chip is sitting there - could check under mic?


![alt text](image-25.png)



## MEA22:

* There was exploration with chip before as i can see test bonds mid april, for example testBond2 and testBond4 ShubhamWafer1, it looked quite good! 
![alt text](image-14.png)
![alt text](image-13.png)
![alt text](image-15.png)

#### Bond2
* I think i started with this one in a serious way. First real attempt with Shubham's devices. Called it **Bond2**. ShubhamW3, mid april, looked very promising. Here i still used copper wire through... copper deterioartion. And shorts issue. Wanted to hook this one, but then got demotivated by shorts.  Could recheck Still was using SC back then, discovered LC later, and did frequency checks.  Measured this one like hell. I started using Gallium, which never fully worked, i actually never fully looked at it if i rmember now... Gallium then harded and shanks ripped... lost one ofter the other like this, in the end also tried silver paint which looked much better. Eventually, i also imaged this with Anna, and i spotted gold on the ring electrodes, suggesting potential for shorts.

![alt text](image-11.png)
![alt text](image-10.png) 

#### Bond5
* So this is the most recent one, late June, so long break for MEA22 (misnamed it before as MEA23, now fixed). High pressure using EcoFlex00-10, disappointed. Had Silk, but overall bad impdance, but also measurements with stimUnits quite unstable... Now Firday, should check this evening... ANd confirmation from last bond (bond2) there seems to be an issue with the gold there, again patchy, high pressure needed. pickup was kinda messy and accidentally scaped polyimide cable once...
![alt text](image-18.png)
![alt text](image-16.png)
![alt text](image-17.png)

Problems... there are again signicant shorts, and now very clearly organized the way they are routed out.
![alt text](image-23.png)
![alt text](image-24.png)




## MEA23

* Interesting stuff. I think here goldplating randomly stopped early... so pillars were low. i tried a single no press bond early/mid april: 2026-04-09_14.44.38_SC_noPressBond_/ Then alexei replated some amount. i think 1.5 um. Perhaps this one has quite high pillars. Needs this profilemeter ffs.... Because then I did a testbond I think shortly after the presentation I gave, where I picked up 4 shanks with wafer, and 4 with silver (right, 5,6,7,8). Did a pressure gradient, and saw how imp improved radically from 1.9 to 1.7. This one is complicated though. Because the silver shanks influced the voltage map readouts, they dropped when measured together. Alfter many measurementes, i figured shorts to GND ring of a few electrodes could short everything to GND, current didn't flow through external sense circuit (solution) when the 4 silver were in there. Here, i also reanalyzed shorts properly for this and foound reasonable pattern where true shorts (pads that are connected by design) have quite a bit higher amplitude than others. But still they shared quite a bit of signal, like 20% - 30%  still not that little tbh... When using manual stimulaiton, saw them everywhere....
![alt text](image-21.png)
![alt text](image-20.png)
![alt text](image-19.png)
![alt text](image-22.png)
On Tuesday, 30.June, went back to this device, in parallel to disassmbley of MEA1K24 (half gold chip) redid imp measurements, and did external current stimulation. 





I sometimes have these moments where i see it, imagine it works/ What a f banger.... how the f is it still not out... also fatih mentioned scopp agina...

Unknown unknown vibe is large, how to reduce? do more research, be more in the loop with outputs, takl to people. this reducines this.

foundational qustion: when impedance is high wtf is the actualy connection? such a simple quesiton, that is unanswered! Microscope imagaes are a clue, profilemeter, impedance patterns over pressure
Do the bonding with the new decices that short everything, or with Alu+Etsu flim


laminar flow hood

plasma

etsu

profilemeter, better goldplating 

those are my bets on mechanistic side...

WHO CAN HELP

clean chip for alexei as good as you can, as if bonding it. 

Could plasma activate surface of interconnect PI film, and put ShinEtsu on top? Would it connect? I think so.

Fernando's elevated PT electrodes? promising

TODAY: 
MEASURE
1. measure singles ext current on MEA1k23.Look at the data, compare iwiht parallel current. Or switch to MEA22, also can connect with stim buffers over night. Maybe iddeal: rebond MEA22 + 100K resistor soldered? but all shorted.
2. Check cur sense circuit using Oscilloscope on well device. Show Alexei if bad. Would be nice to have for debugging MEA23.

MAKE
3. Give Alexei Chip 24, potentially plate 1 more microns on top. Give him Chip 23 as well. The best one. And then rebond with new batch7 electrodes with careful pressure gradient. Best bet... 
4. Get mic data. How does confocal image look like of MEA24? Shorts where we expect? impedance unfortuantely never properly recorded on this one i think, but could still try to combine all these three together to get abetter idea. Hinge on hypothesis: whats happening during a bad bond? small contact? ruptured flex pads? poked?
5. try etsu film without pressure? Could disasselbe MEA22, useless, no path forward for this device i think...

Get yellow bottle + check 3print filament


New theory: why is current always so low : maybe electrodes (just a few) get shorted to GND ringnode, then most of current would flow there. I am looking at these ext current stim right now, and it's not that variable. Feeling is there are parasitic current paths. Always . NOt that high that oscilloscope is beeping, but kohm range is enough: A few 50KOhm PEDOT ones in parallel to the ringnode. this is where the current flows, or have you measured on MEA23 proper currents before? Yes eactually when we remove the Alushanks, it was basically matching, right? or just + some offset (as i have seen so often now. And also recently, always used super high pressure. Check that again. How much does this exaplin? Cur sense not see everything, sure, but what else?)









First: Shubhams wafer: shorts onw wires? Show god
       Also, confirm etching, looks like metal 2 was etched?

Second: New wafer had thin polyimide to pickup, and still shows metal 2 worse than metal one... Etching works ok? Never confirmed right, ALexei will chceck.
New wafer design: check if ok

---

## Impedance log addendum — pulled from `compare_impedance.ipynb` (extracted 2026-08-25, notebook itself deprecated: its plotting functions were promoted to `mea1k_modules/mea1k_visualizations.py` / `mea1k_utils.py`, only this narrative was unique)

**MEA24:**
* Drift check, both halfs, then just shank14 - pretty stable.
* Drift check top half - bad from the beginning, especally metal 2 very bad.
* Shank14, zero current DAC code sensitivity ±20 ±5 ±1, +2 - also deterioaration again... from one day to the next, prob silver plating, or huge 20DAC shift? DC Current...?
* Shank13, very low imp metal 2, thought shorted to ring node GND, but cound't confirm. Again signs of deterioaration, although not DAC if i remember, so maybe silver plated.
* Shank13, systematic shift of DAC ± LC SC, shwowing effect.
* Shank13, external current sensing at callibr DAC (off) and then +3 --> improves it. When DAC off, true current is lower than what chip says.
* Evening, switch to LC, very good calibr, pushed from 2.0 to 1.8 1.6. amplifier off... especially later with LC, although there was a current (which previously meant there is linearity bewteen imp1 and imp2, doesn't hold here, for LC).

**MEA23** - shankd 1-4 PEDOT shanks, 5-8 shorted with silver and Alu - flagged "GOATED TRY" in the notebook:
* 1.7mm pressure (2 measurements that slightly differed, lots of high imp), but center tend to be better.
* Now with high pressure, much better, but plus phase anomalies, they disappeared next day, but pressure must have been really super high, with EcoFlex 00-10 maybe it doesnt get their? hardness needed?
* +2d Imp on silver shanks 4-8: no GND still had low imp, actually even lower than with cable, and phase is stereotypical.
* +7d Imp on real shanks 1-4: no GND huge imp, at 25kOhm with Pt counter GND in solution, 25k lower for some reason... but not shorted it looks like.
* Shorts are real on 1-4 shanks, and alu shanks are really shorted to GND, few 100 Ohm connection to GND.
* +10d meausurements, confirmed that Shank 4-8 is shorted to GND. Measurements somwhat make sense. Good stable impedance on shank 1-4 with smalll and large current mode.
* Before palnning to disseemble, wanted to check one lsat time. Remvoing shank 7 got rid of shoort, good, after 1+ month.
* 4 days later, still connected (extenreal voltage) but impedance looks bad still, SC mode, didn't like how calibration looked thorugh.

**MEA22** (in the notebook mislabeled at the time as "previously MEA23") - first proper full analysis: phase shift checks, also did gallium and silver for the first time.