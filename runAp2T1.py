#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:43:13 2014

@author: dchen
"""

import sys
import os
from os import path
from glob import glob
import json
import argparse
from gts.groupTractStats import *
from gts.gtsutils import exec_cmd

conf_file = sys.argv[1]
gts =  GroupTractStats(conf_file)

conf_name = os.path.basename(conf_file).split('.')[0]
root_path = os.getcwd()    

def getApPerSubj(subj, c):    
    print '--------------------------------run AP---------------------------------'

    orig_path = c.orig_path
    dwi_path = c.dwi_base_path
    root_path = c.root_path
    dwi_file = subj+'.nii.gz'
    t1_file = subj+'_T1.nii.gz'
    os.chdir(orig_path)

    if not path.isfile(t1_file):
        t1_source = path.join(c.T1_processed_path, '%s_antsT1.nii.gz' % subj)
        cmd = 'ln -s %s %s' % (t1_source, t1_file)
        exec_cmd(cmd)

    if not path.isfile(dwi_file):
        # get dwi files
        subj_path = path.join(dwi_path, subj)
        dwi_folder = glob(subj_path+"/*DTI*60*")[0]
        subj_data_source = path.join(dwi_folder,'nifti')

        dwi_source = path.join(subj_data_source, 'Motion_Corrected_DWI_nobet.nii.gz')
        bvec_source = path.join(subj_data_source, 'newdirs.dat')        
        bval_source = path.join(subj_data_source, subj+'.bval')        
        nhdr_source = path.join(subj_data_source, 'DWI_CORRECTED.nhdr')

        cmd = 'ln -s %s %s' % (dwi_source, dwi_file)
        exec_cmd(cmd)
        cmd = 'ln -s %s %s' % (bvec_source, subj+'.bvec')
        exec_cmd(cmd)
        cmd = 'ln -s %s %s' % (bval_source, subj+'.bval')
        exec_cmd(cmd)

        cmd="slicerTensorScalar.sh -i %s -p %s_ -d ." % (nhdr_source, subj)
        exec_cmd(cmd, truncate=True)

    cmd="fslmaths %s -Tmean %s" % (dwi_file, subj+'_MDWI')
    exec_cmd(cmd)    

    # get b0 image
    cmd="fslroi %s %s_B0 0 1" % (dwi_file, subj)
    exec_cmd(cmd)
    # get dwi average only, excluding b0
    cmd="fslroi %s %s_temp 1 -1" % (dwi_file, subj)    
    exec_cmd(cmd)        
    cmd="fslmaths %s_temp -Tmean %s_ADWI" % (subj, subj)    
    exec_cmd(cmd)   
    cmd="rm %s_temp" % (subj)    

    bet='0.2'
    cmd='bet %s_T1 %s_T1_bet -f %s ' % (subj, subj, bet)
    exec_cmd(cmd)

    # generate mask based on dwi
    cmd='bet %s %s_bet -m -n -f %s' % (dwi_file, subj, bet)
    exec_cmd(cmd)
    
    bet_mask = '%s_erod_bet_mask.nii.gz' % subj

    # erode the mask to remove fringe skull intensities in FA
    cmd='fslmaths %s_bet_mask.nii.gz -kernel box -ero %s' % (subj, bet_mask)
    exec_cmd(cmd)


    # apply mask to MDWI
    cmd='fslmaths %s_MDWI -mul %s %s_MDWI_bet' % (subj, bet_mask, subj)
    exec_cmd(cmd)

    # apply mask to MDWI
    cmd='fslmaths %s_ADWI -mul %s %s_ADWI_bet' % (subj, bet_mask, subj)
    exec_cmd(cmd)

    cmd='fslmaths %s_B0 -mul %s %s_B0_bet' % (subj, bet_mask, subj)
    exec_cmd(cmd)    


    # apply mask to FA
    cmd='fslmaths %s_FA -mul %s %s_FA_bet' % (subj, bet_mask, subj)
    exec_cmd(cmd)    

    if 1: 

        import nibabel as nib    

        # generate ap mask
        from sh_to_apm import peaks_from_nifti, sh_to_ap
        import cPickle

        img = nib.load(bet_mask)
        affine = img.get_affine()

        peaks_file = subj+'_qball_peaks.dipy'    
        # if not path.isfile(peaks_file):
        peaks = peaks_from_nifti(subj, mask=bet_mask)
        with open(peaks_file, 'wb') as fout:
            cPickle.dump(peaks, fout, -1)
        # else:
        #     with open(peaks_file, 'rb') as fin:
        #         peaks = cPickle.load(fin)

        coeffs = peaks.shm_coeff
        # AP
        ap_data = sh_to_ap(coeffs, norm=1)
        newimg = nib.Nifti1Image(ap_data, affine)

        ap_file = subj+'_AP3_bet.nii.gz'
        nib.save(newimg, ap_file)

        # AP real
        ap_data = sh_to_ap(coeffs)
        newimg = nib.Nifti1Image(ap_data, affine)

        ap_file = subj+'_AP4_bet.nii.gz'
        nib.save(newimg, ap_file)        

        # GFA
        gfa_data = peaks.gfa
        newimg = nib.Nifti1Image(gfa_data, affine)

        ap_file = subj+'_GFA_bet.nii.gz'
        nib.save(newimg, ap_file)


    bet_files = glob('*bet*')
    for i in bet_files:
        cmd='mv %s %s' % (i, c.preprocessed_path)
        exec_cmd(cmd)



gts.runPerSubject(getApPerSubj)
# gts.preprocessDWI()
# gts.runAntsDwiToT1(bet='0.1')
# gts.projectRoiT1TemplateToSingle()
# gts.projectRoiT1ToDwi()

# tracts = gts.seedIndividualTracts(labels=[2],recompute=False,overwrite=True)





