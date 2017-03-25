#!/usr/bin/env python

import nibabel as nib
import numpy as np
import sys
import multiprocessing
import cPickle
import dipy.core.gradients as dgrad
from dipy.data import get_sphere
import dipy.io as dio
from dipy.reconst.peaks import peaks_from_model
from dipy.segment.mask import median_otsu, applymask


def sh_to_ap(coeffs_mtx, norm=0.000001, gtab=None):

    dim = coeffs_mtx.shape[:3]    
    n_coeffs = coeffs_mtx.shape[3]

    total = dim[0]*dim[1]*dim[2]

    done = 0

    L = 2    # start at L=2
    sum_n = 1

    def single_L_ap(coeffs_mtx, L=2, power=2):
        n_start = 1
        n_L = 2*L+1
        for l in xrange(2,L,2):
            n_l = 2*l+1
            #sum_n start at index 1
            n_start += n_l
        n_stop = n_start + n_L
        print 'func:',L, n_L, n_start, n_stop
        c = np.power(coeffs_mtx[...,n_start:n_stop], power)
        ap_i = np.mean(c, axis=-1)
        #ap_i = np.multiply(ap_i, 1.0/n_L)
        return ap_i

    ap = np.zeros(dim)
    print 'AP shape:',ap.shape
    ap_low = np.zeros(dim)

    while sum_n < n_coeffs:
        n_L = 2*L+1
        ap_i = single_L_ap(coeffs_mtx, L)
        ap = np.add(ap_i, ap)
        sum_n += n_L
        L+=2

    log_ap = np.ma.log(ap/norm)
    # zero all <0 values
    log_ap[log_ap<0] = 0
    return log_ap

def peaks_from_nifti(fdwi, fbvec=None, fbval=None, mask=None):

    if '.' not in fdwi:
        fbase = fdwi
        fdwi = fdwi+".nii.gz"
        if not fbval:
            fbval = fbase+".bval"
        if not fbvec:
            fbvec = fbase+".bvec"
    print fdwi
    img = nib.load(fdwi)
    data = img.get_data()
    zooms = img.get_header().get_zooms()[:3]
    affine = img.get_affine()
    bval, bvec = dio.read_bvals_bvecs(fbval, fbvec)
    gtab = dgrad.gradient_table(bval, bvec)


    if not mask:
        print 'generate mask'
        maskdata, mask = median_otsu(data, 3, 1, False, vol_idx=range(10, 50), dilate=2)

    else:
        mask_img = nib.load(mask)
        mask = mask_img.get_data()

        from dipy.segment.mask import applymask
        maskdata = applymask(data, mask)

    print maskdata.shape, mask.shape


    from dipy.reconst.shm import QballModel, CsaOdfModel
    model = QballModel(gtab, 6)

    sphere = get_sphere('symmetric724')

    print "fit Qball peaks"
    proc_num = multiprocessing.cpu_count()-1
    print "peaks_from_model using core# =" + str(proc_num)

    peaks = peaks_from_model(model=model, data=maskdata, relative_peak_threshold=.5,
                            min_separation_angle=25,
        sphere=sphere, mask=mask, parallel=True, nbr_processes=proc_num)

    return peaks


def main():

    from optparse import OptionParser
    parser = OptionParser(usage="Usage: %prog [options] <input_file>")
    parser.add_option("-e", "--bvec", dest="bvec", help="Bvec file in FSL format")
    parser.add_option("-a", "--bval", dest="bval", help="Bval file in FSL format")

    parser.add_option("-o", "--out", dest="out", help="Output filename")
    parser.add_option("-p", "--is_dipy_peaks", default=False, action="store_true", dest="is_dipy_peaks", help="Input is a pickle of dipy csd peaks")
    parser.add_option("-r", "--ref", dest="ref", help="Reference image for affine info")
    parser.add_option("-m", "--mask", dest="mask", help="Mask image")
    parser.add_option("-s", "--is_sim", action="store_true", dest="is_sim", help="Run map on simulation")

    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
        sys.exit(2)

    input = args[0]

    if options.is_dipy_peaks:
        img = nib.load(options.ref)
        with open(input, 'rb') as fp:
            sh_coeffs = cPickle.load(fp).shm_coeff

    else:
        img = nib.load(input)            
        fbase = input.split('.')[0]
        peaks = peaks_from_nifti(fbase, fbvec=options.bvec, fbval=options.bval, mask=options.mask)
        with open(fbase+'_qball_peaks.dipy', 'wb') as fout:
            cPickle.dump(peaks, fout, -1)

        sh_coeffs = peaks.shm_coeff

    affine = img.get_affine()
    ap_data = sh_to_ap(sh_coeffs)
    newimg = nib.Nifti1Image(ap_data, affine)

    if options.out:
        output_file = options.out
    else:
        output_file = input.split('.')[0]+'_AP.nii.gz'
    nib.save(newimg, output_file)


if __name__ == '__main__':
    main()

