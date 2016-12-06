def tractsDipy(subj, roi_base, exclude_base, label, recompute=False, overwrite=False, label_exclude=10,label_str="", label_include=11, config=None):
    """ TODO: Finish this function
    """
    from dipy.reconst.peaks import peaks_from_model
    from dipy.tracking import metrics as tm
    from dipy.segment.quickbundles import QuickBundles
    from dipy.io.pickles import save_pickle
    from dipy.data import get_data
    from dipy.viz import fvtk

    c = config
    origin = os.getcwd()
    os.chdir('dwi')
    basename=subj+'_cor'
    if not os.path.isfile(basename+'.nii.gz'):
        cmd = 'ln -s Motion_Corrected_DWI_nobet.nii.gz %s.nii.gz' % (basename)
        exec_cmd(cmd)
        cmd = 'ln -s newdirs.dat %s.bvec' % (basename)
        exec_cmd(cmd)
        cmd = 'ln -s %s.bval %s.bval' %(subj, basename)
        exec_cmd(cmd)

    peaks_file = basename+'.peaks'
    if not os.path.isfile(peaks_file):
        peaks = gtsdipy.compCsdPeaks(basename, peaks_file, 'dwibet_mask.nii.gz')
    else:
        peaks = gtsdipy.loadPeaks(peaks_file)

    os.chdir(origin)

    roi_file = roi_base+'.nii.gz'
    output = "eudx_%d.dpy" % label
    gtsdipy.runStream(peaks, roi_file, roi_label=label, output_file=output)

    os.chdir('..')
    return output


def tracts_to_density(ref_file, tract_file):
    import tractDensityMap as tdm 
    tdm.tracts_to_density(ref_file, tract_file)
