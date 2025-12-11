# rrs-SDP-pigments
Translating Kramer Rrs SDP method from MATLAB to python. Original GitHub can be found at https://github.com/sashajane19/Rrs_pigments

Run generate_coefficients in main.py to generate SDP coefficients. Coefficients are saved to an excel sheet. Rrs and coincident SSS and SST are required to generate coefficients. Once coefficients are generated, run run_sdp() in main.py to apply coefficients.

To apply coefficients, run run_sdp() in main.py. This method requires a DataFrame of Rrs spectra (400-700nm, 1nm resolution) and corresponding SSS and SST values. 

To run SDP on PACE L2 data, run sdp_from_pace() in main.py. The file path to a PACE L2 AOP file is required, as well as SMAP SSS and GHRSST SST files. 

Comment out/uncomment code chunks in Kramer_hyperRrs.py to run GSM optimization serially or in-parallel using Ray. 

Papers where methods were used:

Kramer, S.J., D.A. Siegel, S. Maritorena, D. Catlett (2022). Modeling surface ocean phytoplankton pigments from hyperspectral remote sensing reflectance on global scales. Remote Sensing of Environment, 270, 1-14, https://doi.org/10.1016/j.rse.2021.112879.

Kramer, S.J., S. Maritorena, I. Cetinić, P.J. Werdell, D.A. Siegel (2024). Phytoplankton communities quantified from hyperspectral ocean reflectance correspond to pigment-based communities. Optics Express, 32(20), 1-16. https://doi.org/10.1364/OE.529906.

rrsModelTrain.py is adapted from Dylan Catlett. The repository can be found at https://github.com/dcat4/bioOptix_and_PFTs
