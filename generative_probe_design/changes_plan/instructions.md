Plan to change the current generation of probes.

1. Change the hook design. Save the old (current ) hook in a suitable format in a different file. then create a new file with a new hook spec. the hook should look like the image new_hook.png. With one small change the 6 claps on the side should be wider. right now they are 2um, but they should be e5um in the new design.

2. change the electrode shape. Right now it's a square with rounded edges. make it compeltely round and smooth the polyimide as in the image: new_circular_electrode.png 

3. We need to integrate the IONP patterns. it's not yet settles how we do it. generate a random design, should one design spec have different IONP patterns etc. 

4. Small one: get rid of lollipop side hook. while doing this, also fix the etching surrounding. right now the hook is partly etched away, and in other regions this distance is unecessarily big. 

5. Mario already had something lkie this called roof. we want something similar, but probably a nonlinear folding funciton where the length of fibers & where they start doesn't just change at some regular step interval. But a v1 can use a linear adjustment here like the roof.  For  v2, check the image bundling_v2_better_roof.png. THis is the nonlinear version. it's based on this github repo: https://github.com/Neurotechnology-at-ETH-Zurich/electrode2geometry/
Extract the important part from it. it could even be a hardcoded adjustment specifically for the 64 channels we have as well. 

6. Big one: we want to not have a normal recording electrode on the central fiber with the hook, but instead expose contacts bewteen the hook and last reocrding el. these wide reocrding sites will be used as a Ref electrode. This unfortuantely has deep triclkle down effects up to the PCB. because the whole routing needs to fleip where (hopefully possible?) In that the bottom two contact pads right now are not being used, bc they are the Ref and Gnd pads of the  flexPCB. to route them to the center everything needs to be flipped. but that's just an inversion in channel ordering, so can be recovered afterwards. in anycase, how those Ref istes exactly get exposed needs to be defined in some spec as well, buut will be shared across all probes.