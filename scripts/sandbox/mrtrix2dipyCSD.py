# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:33:13 2014

@author: dchen
"""
import numpy as np
import nibabel as nib
from dipy.data import get_sphere
from dipy.reconst.peaks import *
from dipy.reconst.shm import *

def getPeaksFromMrtrix(filename, maskfile='', harmonic_order=8, normalize_peaks=True, npeaks=2):
    cdata = nib.load(filename).get_data()
    sphere = get_sphere('symmetric724')
    data = sh_to_sf(cdata, sphere, harmonic_order, basis_type='mrtrix')

    if maskfile:
        mask = nib.load(maskfile).get_data()

    shape = data.shape[:-1]
    peak_dirs = np.zeros(shape+(npeaks,3))
    peak_values = np.zeros(shape+(npeaks,))
    peak_indices = np.zeros(shape+(npeaks,), dtype='int')

    sphere = get_sphere('symmetric724')
    global_max=-10000
    for idx in np.ndindex(shape):
        if mask is not None:
            if not mask[idx]:
                continue
        odf = data[idx]
        direct,pk,ind = peak_directions(odf, sphere, relative_peak_threshold=.5, min_separation_angle=25)

        # Calculate peak metrics
        # this is taken directly from peaks_from_model
        # https://github.com/nipy/dipy/blob/master/dipy/reconst/peaks.py
        if pk.shape[0] != 0:
            global_max = max(global_max, pk[0])

            n = min(npeaks, pk.shape[0])
            qa_array[idx][:n] = pk[:n] - odf.min()

            peak_dirs[idx][:n] = direction[:n]
            peak_indices[idx][:n] = ind[:n]
            peak_values[idx][:n] = pk[:n]

            if normalize_peaks:
                peak_values[idx][:n] /= pk[0]
                peak_dirs[idx] *= peak_values[idx][:, None]

#        n = min(npeaks, pk.shape[0])
#        peak_indices[idx][:n] = ind[:n]
#        peak_values[idx][:n] = pk[:n]
#        peak_dirs[idx][:n] = direct[:n]

    return peak_dir, peak_values, peak_indices

