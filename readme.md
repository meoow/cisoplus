# CISO Plus
## Converting PSP ISO file to compressed CSO format

The code is mainly based on two project:
[phyber/ciso](https://github.com/phyber/ciso)
[barneygale/iso9660](https://github.com/barneygale/iso9660)

But, I made some modifications and added extra features of what [CisoPlus](http://cisoplus.pspgen.com/) has:

1. Threshold: Only the blocks of which compression ratio is below the specific threshold will be compressed.  
2. Do not compress multimedia files: Leave PMF and AT3 files alone.  
3. Vaccum files for system upgrading: Fill these files with blank data (the iso file is untouched), so it can save even more space.
