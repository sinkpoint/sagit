from gts import exec_cmd
from gts.scripts.sh_to_apm import peaks_from_nifti, sh_to_ap
import nibabel as nib    
import cPickle
from os import path
import os
from glob import glob

def per_subj_apm(self, subject, **kwargs):
    """
        generate anisotropic power image per subject
    """

    c = self.config
    subj = subject.name
    c.preprocessed_path
    orig_path = c.orig_path

    mypath = path.join(subject.dwi_path, subj)
    if subject.dwi_autodetect_folder:
        mypath = glob(path.join(mypath,subject.dwi_autodetect_folder))[0]    
    dwi_path = path.join(mypath, 'nifti')

    print '--------------------------------run AP %s ---------------------------------' % subj

    # the bet mask should already be generated
    dwi_file = path.join(dwi_path, 'DWI_CORRECTED.nii.gz')
    bvec_file = path.join(dwi_path, 'DWI_CORRECTED.bvec')
    bval_file = path.join(dwi_path, 'DWI.bval')

    bet_mask = path.join(c.preprocessed_path, '%s_MDWI_bet_mask.nii.gz' % subj)

    img = nib.load(bet_mask)
    affine = img.get_affine()

    peaks_file = path.join(c.preprocessed_path, subj+'_qball_peaks.dipy')

    if not path.isfile(peaks_file):
        peaks = peaks_from_nifti(dwi_file, fbvec=bvec_file, fbval=bval_file, mask=bet_mask)
        with open(peaks_file, 'wb') as fout:
            cPickle.dump(peaks, fout, -1)
    else:
        with open(peaks_file, 'rb') as fin:
            peaks = cPickle.load(fin)

    coeffs = peaks.shm_coeff
    # AP
    ap_data = sh_to_ap(coeffs)
    newimg = nib.Nifti1Image(ap_data, affine)

    ap_file = path.join(c.preprocessed_path, subj+'_AP_bet.nii.gz')
    nib.save(newimg, ap_file) 

    # GFA
    gfa_data = peaks.gfa
    newimg = nib.Nifti1Image(gfa_data, affine)

    ap_file = path.join(c.preprocessed_path, subj+'_GFA_bet.nii.gz')
    nib.save(newimg, ap_file)
