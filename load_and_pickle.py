from __future__ import division, print_function

from data import load_epi_ad_data

betas, factors = load_epi_ad_data(verbose=True)

print(betas.shape, factors.shape)
