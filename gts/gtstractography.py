import os
import vtk
from gtsutils import exec_cmd
import gtsdipy

def tractsXst(subj, roi_base, exclude_base, label=1, label_exclude=10, overwrite=False, label_str="", label_include=11, dwi_file='', config=None, gen_only=False, params='', method_label='xst'):
    c = config
    roiBase = roi_base
    filterImageBase = exclude_base

    print filterImageBase

    # TODO: make skipping tractography seeding toggleable

    label_name_base = method_label+'_'+label_str
    if label_name_base == "":
        label_name_base="%s_%d"%(roi_base,label)


    tract_output_base = 'xst/%s'%(label_name_base)
    out_file = tract_output_base+'.vtk'
    if not os.path.isfile(out_file) or overwrite:
        #cmd='ten2fiber %s %d 0.15 0.1 0.8 10 0.8 dti.nhdr %s.nhdr' % (dwi_file, label, roiBase)
        #cmd='ten2fiber test.nhdr 1 0.2 0.1 0.8 10 0.1 dti.nhdr test_label.nhdr'
        cmd='seedTend2Points.py -i %s.nhdr -n 10 -r -m %s' % (roiBase, roiBase)
        exec_cmd(cmd)

        cmd="tend2 fiber -i %s -dwi -wspo -ns seeds/%s%d.txt -o %s -ap -v 2 -t 2evec0 -k cubic:0.0,0.5 -n rk4 " % (dwi_file, roiBase,label,out_file)
        if params=='':
            cmd += "-stop aniso:cl2,0.15 frac:0.1 radius:0.8 minlen:10 -step 0.1"
        else:
            cmd += params

        exec_cmd(cmd)

    #convert vtk to vtp files to filter
    vreader = vtk.vtkPolyDataReader()
    vreader.SetFileName(out_file)
    vreader.Update()
    polydata = vreader.GetOutput()

    vwriter = vtk.vtkXMLPolyDataWriter()
    vwriter.SetFileName(tract_output_base+'.vtp')
    vwriter.SetInput(polydata)
    vwriter.Update()
    vwriter.Write()

    import shutil
    filtered_temp = 'xst/%s_filtered.vtp' % label_name_base
    shutil.copy2(tract_output_base+'.vtp', filtered_temp)    

    if label_include > 0:
        cmd='slicerFilterFibers.sh --pass %d %s %s %s' % (label_include, filterImageBase+'.nhdr', filtered_temp, filtered_temp)
        exec_cmd(cmd)

    if label_exclude > 0:
        cmd='slicerFilterFibers.sh --nopass %d %s %s %s' % (label_exclude, filterImageBase+'.nhdr', filtered_temp, filtered_temp)
        exec_cmd(cmd)

    # convert vtp back to vtk to load in dipy
    # TODO: load vtp directly in VTK renderer instead;
    # or consider using webgl

    vreader = vtk.vtkXMLPolyDataReader()
    vreader.SetFileName(filtered_temp)
    vreader.Update()
    polydata = vreader.GetOutput()

    vwriter = vtk.vtkPolyDataWriter()
    output_file = 'xst/%s_filtered.vtk' % label_name_base
    vwriter.SetFileName(output_file)
    vwriter.SetInput(polydata)
    vwriter.Write()

    return output_file

def tractsMrtrix(subj, roi_base, exclude_base, label, recompute=False, overwrite=False, label_exclude=10,label_str="", label_include=11, config=None, params='', method_label='cst'):
    c = config
    roiBase = roi_base
    filterBase = exclude_base

    #exec_cmd('rm dwi/CSD8.mif')
    if not os.path.isfile('dwi/CSD8.mif') or recompute is True:
        os.chdir('dwi')
        # calculate CSD estimation
        cmd = 'rm dwi.mif; mrconvert Motion_Corrected_DWI_nobet.nii.gz dwi.mif'
        exec_cmd(cmd)
        cmd = 'mrtrix_grad.py -v newdirs.dat -a %s.bval -o dwi.grad' % (subj)
        exec_cmd(cmd)
        cmd = 'mrtrix_compcsd.sh -g dwi.grad dwi.mif'
        exec_cmd(cmd)
        os.chdir('..')

    base = label_str 
    if label_str == "":
        base="%s_%d"%(roi_base, label)

    fiber_basename = '%s_%s' % (method_label, base)
    fiber_output = '%s.tck' % (fiber_basename)

    output = '%s.vtk' % fiber_basename    
    output2 = '%s_filtered.vtk' % fiber_basename    

    if not os.path.isfile(fiber_output) or overwrite is True:

        # # clear old tracks
        # cmd = 'rm cst*;rm exclude*.nii.gz;rm exclude*.mif'
        # exec_cmd(cmd)
        
        #filter exclusive mask            
        exclude_param = ""
        if (label_exclude > 0):
            exclude_name = "exclude_%s" % (base)


            cmd = 'fslmaths %s -thr %d -uthr %d %s' % (filterBase, label_exclude, label_exclude, exclude_name)
            exec_cmd(cmd)
            cmd = 'rm %s.mif; mrconvert %s.nii.gz -datatype Bit %s.mif' % (exclude_name,exclude_name,exclude_name)
            exec_cmd(cmd)

            exclude_param = " -exclude %s.mif " % exclude_name


        # filter inclusive mask
        include_param = ""
        if (label_include > 0):
            include_name = "include_%s" % (base)            

            cmd = 'fslmaths %s -thr %d -uthr %d %s' % (filterBase, label_include, label_include, include_name)
            exec_cmd(cmd)            
            cmd = 'rm %s.mif; mrconvert %s.nii.gz -datatype Bit %s.mif' % (include_name,include_name,include_name)
            exec_cmd(cmd)

            include_param = " -include %s.mif " % include_name


        # convert seed mask
        seedthr_name = 'seed_'+base
        cmd = 'fslmaths %s -thr %d -uthr %d %s' % (roiBase, label, label, seedthr_name)
        exec_cmd(cmd)
        cmd = 'rm %s.mif;' % (seedthr_name)
        exec_cmd(cmd)
        cmd='mrconvert %s.nii.gz %s.mif; rm %s.nii.gz' % (seedthr_name,seedthr_name,seedthr_name)            
        exec_cmd(cmd)


        # tune seeding parameters
        if params == '':
            streamparam = "-algorithm iFOD2 -step 0.3 -angle 60 -minlength 10 -cutoff 0.15 -initcutoff 0.15 -force"
        else:
            streamparam = params

        streamparam += " -seed_image %s.mif " % (seedthr_name)

        output_tck = '%s.tck' % fiber_basename
        cmd = 'tckgen  %s  dwi/CSD8.mif  %s' % (streamparam, output_tck)
        exec_cmd(cmd)



        output_tck = '%s_filtered.tck' % fiber_basename
        cmd = 'tckgen  %s %s %s dwi/CSD8.mif %s ' % (streamparam, exclude_param, include_param, output_tck)
        exec_cmd(cmd)


        cmd = 'tracks2vtk %s.tck %s.vtk' % (fiber_basename, fiber_basename)
        exec_cmd(cmd)
        cmd = 'tracks2vtk %s_filtered.tck %s_filtered.vtk' % (fiber_basename, fiber_basename)
        exec_cmd(cmd)



        cmd = 'copyTensors.py -t dti.nhdr -f %s.vtk -o %s ' % (fiber_basename, output)
        #exec_cmd(cmd)

        cmd = 'copyTensors.py -t dti.nhdr -f %s_filtered.vtk -o %s ' % (fiber_basename, output2)
        #exec_cmd(cmd)

    #exec_cmd(cmd)

    return output2

def tractsSlicer(subj, roi_base, exclude_base, label, recompute=False, overwrite=False, label_exclude=10,label_str="", label_include=11, config=None, params='', method_label='slicer'):
    c = config

    roi_file = roi_base+'.nhdr'
    output_base = method_label+'_'+label_str
    unfiltered_file=output_base+'.vtp'

    if not os.path.isfile(unfiltered_file) or overwrite is True:
        if params=='':
            params = '--stoppingvalue 0.2 --minimumlength 5 --clthreshold 0.2 --randomgrid --seedspacing 0.5 ' 

        #cmd = 'slicerTractography.sh --inputroi %s  --label %s %s dti.nhdr %s' % (roi_file, label, params, unfiltered_file)
        cmd = 'slicerTractography.sh --label %s %s dti.nhdr  %s  %s' % (label, params, roi_file, unfiltered_file)
        exec_cmd(cmd)

    import shutil
    fitered_temp = '%s_filtered.vtp' % output_base
    shutil.copy2(unfiltered_file, fitered_temp)

    if label_include > 0:
        cmd='slicerFilterFibers.sh --pass %d %s %s %s' % (label_include, exclude_base+'.nhdr', fitered_temp, fitered_temp)
        exec_cmd(cmd)

    if label_exclude > 0:
        cmd='slicerFilterFibers.sh --nopass %d %s %s %s' % (label_exclude, exclude_base+'.nhdr', fitered_temp, fitered_temp)
        exec_cmd(cmd)    

    vreader = vtk.vtkXMLPolyDataReader()
    vreader.SetFileName(fitered_temp)
    vreader.Update()
    polydata = vreader.GetOutput()

    vwriter = vtk.vtkPolyDataWriter()
    output_file = '%s_filtered.vtk' % output_base
    vwriter.SetFileName(output_file)
    vwriter.SetInput(polydata)
    vwriter.Write()

    return output_file        




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
