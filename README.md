Python implementation of the ensemble for land use change detection using  satellite images. The ensemble is based on three algorithm for detecting changes. These algorithms are chosen to be fundamentally unique to each other. Sample input and validation files are provided. 

These codes are written with pixelwise validation data (provided in the sample input files) in mind. Codes for processing full image stacks (eg., an image stack covering a big patch of land) are available upon request. 

Python codes were scaled using python multiprocessing (across cores) and pyparallel (across nodes). Parallel codes are also available on request.
 
Main file is poly\_1D.py.

bfast\_ps.pdf, landtrendr\_ps.pdf, and ewmacd\_ps.pdf provide the pseudocodes for the algorithms.

