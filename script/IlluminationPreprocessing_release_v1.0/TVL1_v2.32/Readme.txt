Copyright belongs to the original authors: http://www.caam.rice.edu/~optimization/L1/2007/09/software_08.html

Quick start:

>> compile_display
>> Graph_anisoTV_L2_v2
>> Graph_anisoTV_L1_v2

Files:

- compile_display.m

Compile source files into binary code that displays information.
 
- compile_no_display.m

Compile source files into binary code that does not display information.

- compile_display_debug.m

Compile source files in a debug mode into binary code that displays information.

- Graph_anisoTV_L1_v2_consistent_weights.m / Graph_anisoTV_L2_v2_consistent_weights.m

These two files use the new neighbhor weights described in Rice CAAM TR07-09 that are better than those used before in the sense they better approximate the true anisotropic TV.