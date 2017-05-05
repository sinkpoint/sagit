#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:43:13 2014

@author: dchen
"""

import json
import numpy as np
import os
import shutil
from functools import partial
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from glob import glob
from os import chdir
from os import getcwd
from os import mkdir
from os import path

import nibabel as nib

from gtsutils import exec_cmd
#from nibabel import trackvis as tv
#from dtproc import *
from gtsconfig import GtsConfig
from pynrrd import *

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import gc
import gts.meas.imagescore as imgs
import time
import vtk

from gts.maps import tract_density as tdm

from tractography import TractographyMethod
from ordered_set import OrderedSet


def getDwiRoiName(config,subj,basename, ref=""):
    c = config    
    name=c.prefix+'_'+subj+'_'+basename
    if ref!="":
        name+='_'+ref
    return name

def getGtsFilename(config, *args, **kwargs):
    c = config
    if not 'prefix' in kwargs:
        kwargs['prefix'] = True
    if not 'ext' in kwargs:
        kwargs['ext'] = True

    if kwargs['prefix']:
        name=c.prefix
    else:
        name=args[0]
        args = args[1:]
    for i in args:
        if i.strip():
            name+='_'+i
    if kwargs['ext']:
        name += c.imgext

    return name


class GroupTractStats:
    def __init__(self, conf_file=''):

        self.config = GtsConfig(conf_file)

        c = self.config
        path_queue = [
            c.preprocessed_path,
            c.processed_path,
            c.tractography_path,
            c.ind_roi_path
        ]
        for p in path_queue:
            if not path.isdir(p):
                mkdir(p)

    def _g(self, *args, **kwargs):
        return getGtsFilename(self.config, *args, **kwargs)

    def _d(self, subj, basename, ref=""):
        return getGtsFilename(self.config, subj, basename, ref, ext=False)

    def filterSubjects(self, names_list):
        filtered = []
        for s in self.config.subjects_pool:
            if s.name in names_list:
                print '>','filtered for',s.name
                filtered.append(s)
        self.config.subjects = filtered
        print self.config.subjects

    def _run_pool_wrapper(self, (func, subj), **kwargs):
        func(self, subj, **kwargs)

    def runPerSubject(self, func, **kwargs):
        if 'single_thread' in kwargs and kwargs['single_thread'] is True:
            for subj in self.config.subjects:
                func(self, subj, **kwargs)
        else:
            poolargs = [ (func, subj) for subj in self.config.subjects]
            print '# runPerSubj args',poolargs
            pool = ThreadPool()
            pool.map(partial(self._run_pool_wrapper, **kwargs), poolargs)
            pool.close()
            pool.join()

    def run(self, func, **kwargs):
        func(self, **kwargs)

    def projectRoiT1TemplateToSingle(self):
        """ Project averaged T1 ROI to indidivual T1 space
        """

        c = self.config
        imgext = c.imgext
        template_rois= c.template_def
        ind_roi_path = c.ind_roi_path

        for subject in c.subjects:
            subj = subject.name
            print '''
            ========================================================================
            ╔═╗┬─┐┌─┐ ┬┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌┬┐┌─┐┬  ┌─┐┌┬┐┌─┐  ╔╦╗┌─┐  ╔═╗┬┌┐┌┌─┐┬  ┌─┐
            ╠═╝├┬┘│ │ │├┤ │   │    ║ ├┤ │││├─┘│  ├─┤ │ ├┤    ║ │ │  ╚═╗│││││ ┬│  ├┤ 
            ╩  ┴└─└─┘└┘└─┘└─┘ ┴    ╩ └─┘┴ ┴┴  ┴─┘┴ ┴ ┴ └─┘   ╩ └─┘  ╚═╝┴┘└┘└─┘┴─┘└─┘
            ========================================================================
            {}
            '''.format(subj)


            for rname, filename in template_rois.iteritems():
                if rname == 'reference':
                    continue

                basename = filename.split('.')[0]
                roi = path.join( c.template_roi_path,basename)
                roiName = path.basename(roi).rsplit('.')[0]
                inName = roi+imgext
                outName = self._g(subj, roiName)
                output = path.join( ind_roi_path,outName)

                self.ants_apply_transform_average_to_t1(subj, inName, output)

    # project individual ROI from T1 space to DWI space
    def projectRoiT1ToDwi(self):
        c = self.config
        orig_path = c.orig_path
        imgext = c.imgext
        prefix = c.prefix
        affix = c.affix
        T1_path = c.T1_path
        processed_path = c.processed_path
        ind_roi_path = c.ind_roi_path

        rois = []
        for k in c.template_def:
            if k=='reference':
                continue
            rois.append(c.template_def[k].split('.')[0])



        self.projectImageT1ToDwi(ind_roi_path, rois, reference='dwi')
        self.projectImageT1ToDwi(ind_roi_path, rois, reference='t1')

        for subject in c.subjects:
            subj = subject.name
            fspath = c.default_freesurfer_path
            if subject.group in c.group_paths:
                try:
                    grppath = c.group_paths[subject.group]['freesurfer']
                    fspath = grppath if grppath else fspath
                except KeyError:
                    pass

            fspath = path.join(fspath, subj, 'mri')
            fs_t1_file = path.join( c.preprocessed_path , self._g(subj,'fs','T1', prefix=False) )
            fs_aseg_file = path.join( c.preprocessed_path , self._g(subj,'fs','aseg', prefix=False))
            fs2fsl_mat = path.join( c.preprocessed_path , self._g(subj,'fs','fs2fsl','mat.txt', prefix=False, ext=False))
            subj_aseg = path.join( c.preprocessed_path , self._g(subj,'aseg', prefix=False))
            subj_t1 = path.join( T1_path, self._g(subj, prefix=False))
            dwi_aseg = path.join( c.preprocessed_path,self._g(subj, 'aseg', 'dwi', prefix=False))

            print '#',fs2fsl_mat
            if not path.isfile(fs2fsl_mat):
                cmd = 'mri_convert %s %s' % (path.join(fspath,'T1.mgz'), fs_t1_file)
                exec_cmd(cmd)

                cmd = 'mri_convert %s %s' % (path.join(fspath,'aparc+aseg.mgz'), fs_aseg_file)
                exec_cmd(cmd)

                cmd = 'flirt -in %s -ref %s -omat %s' % (fs_t1_file, subj_t1, fs2fsl_mat)
                exec_cmd(cmd)

                cmd = 'flirt -in %s -ref %s -interp nearestneighbour -applyxfm -init %s -out %s' % (fs_aseg_file, subj_t1, fs2fsl_mat, subj_aseg)
                exec_cmd(cmd)

            self.ants_apply_transform_t1_to_dwi(subj, subj_aseg, dwi_aseg, auto_naming=False)


    def projectImageT1ToDwi(self, input_path, name_base, output_path='', **kwargs):
        """ Apply Ants deformation from T1 space to Dwi space for each individual

            Args:
                input_path: The input path to find the images.
                name_base: The filename basis used to find the images, final name is derived as prefix+subject+'_'+name_base+ext.
                output_path: Where to put the output file, default is the processed path.
                invserse: Inverse the direction of deformation, so becomes Dwi to T1 space instead.
                reslice: default True; true: use DWI spacing; false: use T1 spacing
                interp: Interpolation method used for deformation. This must match the antsApplyTransforms command. Default is 'NearestNeighbor'
        """
        c = self.config
        imgext = c.imgext
        prefix = c.prefix
        affix = c.affix
        processed_path = c.processed_path

        if output_path=='':
            output_path = processed_path

        for subject in c.subjects:
            subj = subject.name
            print '''
            ==========================================================
            ╔═╗┬─┐┌─┐ ┬┌─┐┌─┐┌┬┐  ╦┌┬┐┌─┐┌─┐┌─┐  ╔╦╗  ╔╦╗┌─┐  ╔╦╗╦ ╦╦
            ╠═╝├┬┘│ │ │├┤ │   │   ║│││├─┤│ ┬├┤    ║    ║ │ │   ║║║║║║
            ╩  ┴└─└─┘└┘└─┘└─┘ ┴   ╩┴ ┴┴ ┴└─┘└─┘   ╩    ╩ └─┘  ═╩╝╚╩╝╩
            ==========================================================
            {}
            '''.format(subj)

            for name in name_base:
                inName = path.join( input_path,self._g(subj, name))
                outName = self._d(subj, name)
                output = path.join( output_path,outName)


                self.ants_apply_transform_t1_to_dwi(subj, inName, output, **kwargs)

                #dir_roi = path.join( path.join( subject_dti_path,subj,outName))
                #print 'copy %s to %s' %( output, dir_roi )
                #shutil.copyfile(output, dir_roi)

    """
        The the transformation arguments for ANTS, depending on the direction of the transformation
    """
    def get_ants_t1_to_dwi_transforms(self, subj, inverse=False, reference='dwi'):
        c = self.config
        orig_path = c.orig_path
        processed_path = c.processed_path

        subj_pref = self._g(subj, ext=False)


        ref = '%s_FA.nii.gz' % subj
        ref = path.join( orig_path,ref)

        if reference=='t1':
            ref = '%s.nii.gz' % subj
            ref = path.join(c.T1_path, ref)


        invWarpAffix = 'InverseWarp.nii.gz'
        warpAffix = 'Warp.nii.gz'
        affAffix = 'Affine.txt'

        if len(glob(subj_pref+affAffix)) == 0:
            # can't find affine.txt, assume it's mat
            invWarpAffix = '1InverseWarp.nii.gz'
            warpAffix = '1Warp.nii.gz'
            affAffix = '0GenericAffine.mat'

        subj_invWarp = path.join( processed_path,subj_pref+invWarpAffix)
        subj_warp = path.join( processed_path,subj_pref+warpAffix)
        subj_affine = path.join( processed_path,subj_pref+affAffix)

        if not inverse:
            trans = "-t [%s,1] -t %s" % (subj_affine, subj_invWarp)
        else:
            trans = "-t %s -t %s" % (subj_warp, subj_affine)

        return [trans, ref]

    def ants_apply_transform_t1_to_dwi(self, subj, input, output, inverse=False, reference='dwi', interp='NearestNeighbor', just_transform_param=False, auto_naming=True):

        c = self.config
        orig_path = c.orig_path
        imgext = c.imgext
        prefix = c.prefix
        affix = c.affix
        processed_path = c.processed_path

        subj_pref = self._g(subj, ext=False)


        ref = '%s_FA.nii.gz' % subj
        ref = path.join( orig_path,ref)
        if reference=='t1':
            ref = '%s.nii.gz' % subj
            ref = path.join( c.T1_path,ref)

        if auto_naming:
            output = output+'_'+reference+imgext
        #print '######t1_to_dwi#######',output


        invWarpAffix = 'InverseWarp.nii.gz'
        warpAffix = 'Warp.nii.gz'
        affAffix = 'Affine.txt'

        if len(glob(subj_pref+affAffix)) == 0:
            # can't find affine.txt, assume it's mat
            invWarpAffix = '1InverseWarp.nii.gz'
            warpAffix = '1Warp.nii.gz'
            affAffix = '0GenericAffine.mat'

        subj_invWarp = path.join( processed_path,subj_pref+invWarpAffix)
        subj_warp = path.join( processed_path,subj_pref+warpAffix)
        subj_affine = path.join( processed_path,subj_pref+affAffix)

        cmd = cmd="antsApplyTransforms -d 3 -i %s -o %s -r %s -n %s " % (input, output, ref, interp)
        if not inverse:
            trans = "-t [%s,1] -t %s" % (subj_affine, subj_invWarp)
        else:
            trans = "-t %s -t %s" % (subj_warp, subj_affine)

        if just_transform_param:
            return trans

        cmd += trans
        exec_cmd(cmd)

    def get_ants_t1_to_avg_transforms(self, subj, inverse=False, reference='t1'):
        c = self.config
        T1_path = c.T1_path
        processed_path = c.T1_processed_path

        #subj_pref = self._g(subj,ext=False)
        subj_pref = c.prefix+'_'+subj

        ref = '%s.nii.gz' % subj
        if reference=='average':
            ref = c.group_template_file
        ref = path.join( T1_path,ref)

        group_prefix = c.group_prefix
        invWarpAffix = group_prefix+'InverseWarp.nii.gz'
        warpAffix = group_prefix+'Warp.nii.gz'
        affAffix = group_prefix+'Affine.txt'

        if len(glob(path.join(processed_path,subj_pref+affAffix))) == 0:
            # can't find affine.txt, assume it's mat
            invWarpAffix = '1InverseWarp.nii.gz'
            warpAffix = '1Warp.nii.gz'
            affAffix = '0GenericAffine.mat'

        subj_invWarp = path.join( processed_path,subj_pref+invWarpAffix)
        subj_warp = path.join( processed_path,subj_pref+warpAffix)
        subj_affine = path.join( processed_path,subj_pref+affAffix)

        if not inverse:
            trans = "-t [%s,1] -t %s" % (subj_affine, subj_invWarp)
        else:
            trans = "-t %s -t %s" % (subj_warp, subj_affine)
            
        return [trans, ref]        

    def ants_apply_transform_average_to_t1(self, subj, input, output, reference='t1', interp='NearestNeighbor', inverse=False,just_transform_param=False):

        c = self.config
        T1_path = c.T1_path
        imgext = c.imgext
        prefix = c.prefix
        affix = c.affix
        processed_path = c.T1_processed_path

        #subj_pref = self._g(subj,ext=False)
        subj_pref = c.prefix+'_'+subj

        ref = '%s.nii.gz' % subj
        if reference=='average':
            ref = c.group_template_file
        ref = path.join( T1_path,ref)

        group_prefix = c.group_prefix
        invWarpAffix = group_prefix+'InverseWarp.nii.gz'
        warpAffix = group_prefix+'Warp.nii.gz'
        affAffix = group_prefix+'Affine.txt'

        detect_path = path.join(processed_path,subj_pref+affAffix)
        print '>>',detect_path
        if len(glob(detect_path)) == 0:
            # can't find affine.txt, assume it's mat
            invWarpAffix = '1InverseWarp.nii.gz'
            warpAffix = '1Warp.nii.gz'
            affAffix = '0GenericAffine.mat'

        subj_invWarp = path.join( processed_path,subj_pref+invWarpAffix)
        subj_warp = path.join( processed_path,subj_pref+warpAffix)
        subj_affine = path.join( processed_path,subj_pref+affAffix)

        #cmd="WarpImageMultiTransform 3 %s %s -R %s --use-NN -i %s %s" % (inName, output, subj_FA, subj_affine, subj_invWarp)
        cmd = cmd="antsApplyTransforms -d 3 -i %s -o %s -r %s -n %s " % (input, output, ref, interp)
        if not inverse:
            trans = "-t [%s,1] -t %s" % (subj_affine, subj_invWarp)
        else:
            trans = "-t %s -t %s" % (subj_warp, subj_affine)

        if just_transform_param:
            return trans

        cmd += trans
        exec_cmd(cmd)

    def preprocessDWI(self):
        c = self.config
        root_path = getcwd()
        for subject in c.subjects:
            subj = subject.name
            dwi_path = c.default_dwi_path
            dwi_autodetect_folder = c.dwi_autodetect_folder
            

            if subject.group in c.group_paths:
                try:
                    grouppaths = c.group_paths[subject.group]
                    grpdwipath = grouppaths['dwi']
                    dwi_path = grpdwipath if grpdwipath else dwi_path

                    dwi_autodetect_folder = grouppaths['dwi_autodetect_folder']
                except KeyError or TypeError:
                    pass


            print '''
            =======================================
            ╔═╗┬─┐┌─┐┌─┐┬─┐┌─┐┌─┐┌─┐┌─┐┌─┐  ╔╦╗╦ ╦╦
            ╠═╝├┬┘├┤ ├─┘├┬┘│ ││  ├┤ └─┐└─┐   ║║║║║║
            ╩  ┴└─└─┘┴  ┴└─└─┘└─┘└─┘└─┘└─┘  ═╩╝╚╩╝╩
            =======================================
            '''
            subj_path = path.join(dwi_path,subj)
            chdir(subj_path)
            if dwi_autodetect_folder:
                dwi_folder = glob(dwi_autodetect_folder)[0]
            else:
                dwi_folder = ''
            
            nifti_path = path.join(dwi_folder,'nifti')
            if not path.exists(nifti_path):
                exec_cmd('eddycor.py -s {dwi_autodetect_folder} {subj_path}'.format(**locals()))

            chdir(nifti_path)
            print getcwd()

            cmd="fslmaths DWI_CORRECTED.nii.gz -Tmean %s" % (path.join(c.orig_path,subj+'_MDWI'))
            exec_cmd(cmd)

            cmd="slicerTensorScalar.sh -i DWI_CORRECTED.nhdr -p %s_ -d %s" % (subj,c.orig_path)
            exec_cmd(cmd, truncate=True)

            chdir(root_path)


        chdir(root_path)

    def skullStrip(self, bet='0.1'):
        self.runAntsDwiToT1(bet)

    def runAntsDwiToT1(self,bet='0.1'):
        """ Use ANTS to register individual DWI space to T1 space.
            For now registers average dwi to T1.

            Args:
                bet: The -f parameter value for FSL bet command. Default is 0.1.
        """
        print '''
        ==================================
        ╔═╗┌┐┌┌┬┐┌─┐  ╔╦╗╦ ╦╦  ╔╦╗┌─┐  ╔╦╗
        ╠═╣│││ │ └─┐   ║║║║║║   ║ │ │   ║ 
        ╩ ╩┘└┘ ┴ └─┘  ═╩╝╚╩╝╩   ╩ └─┘   ╩ 
        ==================================
        '''

        c = self.config
        preprocessed_path = c.preprocessed_path
        root_path = getcwd()
        orig_path = c.orig_path
        t1_path = c.T1_path

        betfiles = ""
        #_SIMULATE = True

        for subject in c.subjects:
            subj = subject.name
            chdir(orig_path)
            print getcwd()

            cmd='bet2 %s %s_T1_bet -f %s ' % (path.join(t1_path,subj), subj, bet)
            exec_cmd(cmd)

            cmd='bet2 %s_MDWI %s_MDWI_bet -m -f %s' % (subj, subj, bet)
            exec_cmd(cmd)

            bet_files = glob('*bet*')
            for i in bet_files:
                cmd='mv %s %s' % (i, preprocessed_path)
                exec_cmd(cmd)

            betfiles += "%s_T1_bet.nii.gz " % subj
            betfiles += "%s_MDWI_bet.nii.gz " % subj
            chdir(root_path)

        chdir(preprocessed_path)
        cmd='slicesdir '+betfiles
        exec_cmd(cmd)

        chdir(root_path)
        antsparam = c.subjects_file
        if c.manual_subjects:
            antsparam = " ".join([s.name for s in c.subjects])

        #cmd='./research/runAntsFAToT1.sh %s' % antsparam
        #exec_cmd(cmd)


    def seedIndividualTracts(self, labels=[1],recompute=False,overwrite=False, reorganize_paths=False, run_unfiltered=True):

        print '''
        ========================================================
        ┌─┐┌─┐┌─┐┌┬┐╦┌┐┌┌┬┐┬┬  ┬┬┌┬┐┬ ┬┌─┐┬ ╔╦╗┬─┐┌─┐┌─┐┌┬┐┌─┐
        └─┐├┤ ├┤  ││║│││ │││└┐┌┘│ │││ │├─┤│  ║ ├┬┘├─┤│   │ └─┐
        └─┘└─┘└─┘─┴┘╩┘└┘─┴┘┴ └┘ ┴─┴┘└─┘┴ ┴┴─┘╩ ┴└─┴ ┴└─┘ ┴ └─┘
        ========================================================
        '''      
        c = self.config
        origin=getcwd()


        orig_path = c.orig_path
        imgext = c.imgext
        prefix = c.prefix
        affix = c.affix
        T1_processed_path = c.T1_processed_path
        processed_path = c.processed_path
        ind_roi_path = c.ind_roi_path
        tractography_path = c.tractography_path_full

        if not path.isdir(tractography_path):
            mkdir(tractography_path)

        res = []

        # start a new time log header
        TRACT_TIME_LOG = c.tract_time_log_file
        with open(TRACT_TIME_LOG, 'w') as fp:
            fp.write('subject,method,roi,filter,time\n')        


        for subject in c.subjects:
            subj = subject.name
            tractography_path = subject.tractography_path

            print '-----------------------------------------------------------------'
            print subj
            subjdir = path.join( tractography_path,subj)
            subjRes = { 'name' : subj }

            dwi_path = c.default_dwi_path
            dwi_autodetect_folder = c.dwi_autodetect_folder
            if subject.group in c.group_paths:
                try:
                    grouppaths = c.group_paths[subject.group]
                    grpdwipath = grouppaths['dwi']
                    dwi_path = grpdwipath if grpdwipath else dwi_path       

                    dwi_autodetect_folder = grouppaths['dwi_autodetect_folder']     
                except KeyError:
                    pass


            if not path.isdir(subjdir) or reorganize_paths:
                ## setup subject tractography directory

                try:
                    mkdir(subjdir)
                except OSError:
                    #file already exists
                    pass


                subj_source_path = path.join(dwi_path,subj)
                chdir(subj_source_path)

                
                if dwi_autodetect_folder:
                    dwi_folder = glob(dwi_autodetect_folder)[0]
                else:
                    dwi_folder = ''
                chdir(path.join(dwi_folder,'nifti'))

                subj_dwi_source_path_abs = getcwd()
                chdir(orig_path)

                cmd = 'ln -s %s %s' % (subj_dwi_source_path_abs, path.join(subjdir, 'dwi'))
                exec_cmd(cmd)


                print 'copy DWI file to '+subjdir

                # copy dwi_corrected to tractography/subj dir
                # correct this to RAS, because xst doesn't understand LPS 

                dwi_file = 'DWI_CORRECTED.nhdr'
                slicer_comp_file = path.join(subjdir,dwi_file)

                cmd = 'slicerDwiFileConvert.sh %s %s' % (path.join(subj_dwi_source_path_abs, dwi_file), slicer_comp_file)
                exec_cmd(cmd)

                # following is delegated to xst tractography class
                # This is required as XST cannot correct read space directions other than RAS                             
                # ras_corrected_file = subj+'_dwi_ras.nhdr'
                # reader = NrrdReader()
                # header, b = reader.load(slicer_comp_file)
                # # once slicer reads the file, it will convert the file it's a space that it understands                
                # header.correctSpaceRas() # convert spacing to RAS, this is needed for xst, else geometry will be inverted.
                # writer = NrrdWriter()
                # writer.write(header, path.join(subjdir,ras_corrected_file))
                

                # generate Tensor, FA, RD, AD, MD maps
                cmd="slicerTensorScalar.sh -i %s -p %s_ -d %s" % (slicer_comp_file, subj,subjdir)
                exec_cmd(cmd)


                # copy t1s over
                t1i = self._g(subj, 'invDeformed', 'DWS')
                t1o = self._g(subj, 'T1', 'dwi', prefix=False)
                shutil.copyfile(path.join(processed_path,t1i),path.join( subjdir,t1o))

                t1i = self._g(subj, 'invDeformed', 'T1')
                t1o = self._g(subj, 'T1', prefix=False)
                shutil.copyfile(path.join(processed_path,t1i),path.join( subjdir,t1o))

                # copy over MDWI files, this should be in preprocessing step
                t1i = self._g(subj, 'MDWI', prefix=False)
                t1o = t1i
                shutil.copyfile(path.join(orig_path,t1i),path.join( subjdir,t1o))

                # copy freesurfer segmentation to tractography folder
                dwi_aseg = self._g(subj, 'aseg', 'dwi', prefix=False)
                shutil.copyfile(path.join(c.preprocessed_path,dwi_aseg),path.join( subjdir,dwi_aseg))                

            ############# Deal with ROIs 

            '''
            pseudo code

            for roi object in roi definitions
                roi_obj.generate(subject)
            for seeds
                seeds seed the relevant 
            '''

            chdir(subjdir)
            print 'copy ROI file to '+subjdir

            # copy t1 projected roi files to individual tractography folder    

            # templates = c.template_def
            # for tname, tfile in templates.iteritems():
            #     if tname == 'reference':
            #         # skip the reference anatomy image
            #         continue
            #     [TODO] copy template files to DWI space without explicit roi usage

            roi_set = set()
            rois = c.rois_def
            for k, roi in rois.iteritems():
                if roi.type == 'from_template':
                    roi_file = roi.get_filename(subj)
                    
                    if roi_file not in roi_set:
                        roi_set.add(roi_file)
                        
                        print 'copy',roi_file,processed_path,subjdir
                        shutil.copyfile(path.join(processed_path,roi_file), path.join(subjdir,roi_file))
                        roi_filebase = roi_file.split('.')[0]

                        cmd="slicerFileConvert.sh %s %s " % (roi_file, roi_filebase+'.nhdr');
                        exec_cmd(cmd, display=False)

                        roi_file = roi.get_filename(subj, ref='t1')
                        print 'copy',roi_file,processed_path,subjdir
                        shutil.copyfile(path.join(processed_path,roi_file), path.join(subjdir,roi_file))


            ############ Pass on methods list and generate tracts

            stream_map = {}
            fiber_name = ''
            method_name = ''

                            
            seeds = c.seeds_def
            for k,seed_map in seeds.iteritems():
                label_str = k
                seed_map['name'] = k

                ## determine the methods to run for this seed_def
                methods_queue = []

                try:
                    seed_methods = seed_map["methods"]
                    for s in seed_methods:
                        methods_queue.append(c.tract_method[s])
                except KeyError:
                    methods_queue = c.tract_method.values()

                for imethod in methods_queue:
                    method_label  = imethod['label']


                    print '\n-- PERFORM %s' % method_label
                    method = TractographyMethod.factory(subj, seed_map, imethod, c)                       


                    start_time = time.time()
                    fiber_name = method.run(filter=True, recompute=recompute)
                    stop_time = time.time()
                    elapsed_time = stop_time - start_time

                    report = '%s,%s,%s,filtered,%.9f' % (subj, method_label, label_str, elapsed_time)

                    with open(TRACT_TIME_LOG, 'a') as fp:
                        fp.write(report+'\n')                                                

                    if run_unfiltered:
                        start_time = time.time()
                        fiber_name = method.run(filter=False) # RUN THE METHOD
                        stop_time = time.time()
                        elapsed_time = stop_time - start_time

                        report = '%s,%s,%s,unfiltered,%.9f' % (subj, method_label, label_str, elapsed_time)

                        with open(TRACT_TIME_LOG, 'a') as fp:
                            fp.write(report+'\n') 


                    if not stream_map.has_key(method_label):
                            stream_map[method_label] = [fiber_name]
                    else:
                        stream_map[method_label].append(fiber_name)



            chdir(origin)
        return stream_map

    def tracts_to_density(self, tracts_map):
        print ''''
        ===================== 
        ╔╦╗┬─┐┌─┐┌─┐┌┬┐┌─┐  ╔╦╗┌─┐┌┐┌┌─┐┬┌┬┐┬ ┬
         ║ ├┬┘├─┤│   │ └─┐   ║║├┤ │││└─┐│ │ └┬┘
         ╩ ┴└─┴ ┴└─┘ ┴ └─┘  ═╩╝└─┘┘└┘└─┘┴ ┴  ┴ 
        =======================
        '''
        c = self.config
        for subject in c.subjects:
            subj = subject.name
            tractography_path = subject.tractography_path
            chdir(path.join(tractography_path,subj))
            print '========== %s =========' % subj

            ref_file = 'ANTS_%s_invDeformed_T1.nii.gz' % subj
            ref_file = path.join(c.processed_path,ref_file)

            for method in tracts_map.keys():
                print '== Method %s ' % method
                tlabels = tracts_map[method]
                for tfile in tlabels:
                    print tfile
                    if path.isfile(tfile):
                        tdm.tracts_to_density(ref_file, tfile)
                    else:
                        print 'file %s not found' % tfile

            chdir(c.root)



    def viewTracks(self):
        print '===================== GENERATER TRACTS BROWSER ======================='

        c = self.config
        origin=getcwd()

        orig_path = c.orig_path
        imgext = c.imgext
        prefix = c.prefix
        affix = c.affix
        T1_processed_path = c.T1_processed_path
        processed_path = c.processed_path
        rois= c.template_rois
        ind_roi_path = c.ind_roi_path
        # tractography_path = c.tractography_path

        for label_name, seed_def in c.seeds_def.iteritems():
            for imethod in c.tract_method:

                tmethod = imethod['method']
                mlabel = imethod['label']
                roi = c.rois_def[seed_def['source']]

                fn = '%s_%s' % (mlabel, label_name)
                files = [fn,'%s_filtered' % fn]

                # HTML File output
                jsfile = 'tracts_%s_%s.js' % (mlabel, label_name)

                htmlfile="""
                <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
                   "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
                <html>
                    <head>
                        <script type="text/javascript" src="research/jslibs/jquery2.js"></script>
                        <script type="text/javascript" src="research/jslibs/jquery.visible.js"></script>
                        <script type="text/javascript" src="xtk/utils/xtk.js"></script>
                        <script type="text/javascript" src="http://get.goXTK.com/xtk_xdat.gui.js"></script>
                        <script type="text/javascript" src="research/jslibs/loadvtk.js"></script>
                        <script type="text/javascript" src="%s"></script>
                        <style type="text/css">
                        <!--
                            @import url(research/jslibs/style.css);
                        -->
                        </style>
                    </head>
                    <body>\n
                """ % jsfile

                # JS File output

                js="""
                var VTK_FILES = {};
                function partialLoadXtk() {
                    $(".xtkren").each(function(i, el) {
                        var contain = $(this);
                        var id = contain.attr('id');
                        if ($(this).visible(true)) {
                            //console.log('++: ' + contain);
                            loadVtk(contain, VTK_FILES[id]);
                        } else {
                            //console.log('--: ' + contain);
                            unloadVtk(contain);
                        }
                    });
                }

                $(window).load(function() {
                    partialLoadXtk();
                });

                $(window).scroll(function(event) {

                    partialLoadXtk();

                });
                """

                for subject in c.subjects:
                    subj = subject.name
                    tractography_path = subject.tractography_path

                    subj_tractography_path = path.join( tractography_path,subj)
                    print subj_tractography_path
                    chdir(subj_tractography_path)
                    print '-------------------------viewTracks---------------------------'
                    print getcwd()

                    # convert vtk to trackviz fibers for loading intlo dipy

                    htmlfile += '<div>'

                    for f in files:
                        print f
                        subjpath = '%s/%s' % (tractography_path, subj)
                        vtkfiber = '%s/%s.vtk' % (subjpath, f)
                        volume_base = '%s/%s' % (subjpath, subj)

                        labelmap_dwi = path.join(subjpath,roi.get_filename(subj, autogen=False))

                        labelmap_t1 = path.join(subjpath,roi.get_filename(subj, ref='t1', autogen=False))


                        ### javascript data

                        subj_vols = {'FA':['%s_FA.nii.gz' % volume_base, labelmap_dwi],
                            'T1': ['%s_T1_dwi.nii.gz' % volume_base, labelmap_t1],
                            'MDWI': ['%s_MDWI.nii.gz' % volume_base, labelmap_dwi] }

                        fname = f
                        if tmethod=='xst':
                            fname = f.split('/')
                            if len(fname) > 1:
                                fname = fname[1]
                            else:
                                fname = fname[0]

                        varname = 'r'+subj+fname
                        js += """
                        VTK_FILES['%s'] = ['%s',%s];

                        """ % (varname, vtkfiber, json.dumps(subj_vols))

                        ### html control div

                        vols_html = ""
                        for k in subj_vols.keys():
                            vols_html += """
                            <li>
                                <label>%s</label><input type="checkbox" name="%s" onclick="if ($(this).prop('checked')) {loadVolume('%s', '%s');} else {unloadVolume('%s');}"/>
                            </li>""" % (k, k, varname, k, varname)


                        ctl_html = """<label>%s %s</label>
                            <ul>
                            <li><input type="button" value="Expand" onclick="toggleFullScreen('%s', $(this));"/></li>
                                %s
                            </ul>
                            """ % (subj, f, varname, vols_html)

                        htmlfile+='<div id="%s" class="xtkren"><div class="controls">%s</div></div>\n' % (varname, ctl_html)

                    htmlfile += '<div style="clear:both" /></div>'
                    chdir(origin)
                htmlfile+='</body></html>\n'

                fname = 'tracts_%s_%s.html' % (mlabel, label_name)
                print 'write->',fname
                FILE = open(fname, 'w')
                FILE.write(htmlfile)
                FILE.close()

                FILE = open(jsfile, 'w')
                FILE.write(js)
                FILE.close()

    def tracts_to_images(self, stream_map):

        from dipy.viz import fvtk
        from dipy.viz.colormap import line_colors

        print '===================== PERFORM tracts_to_images ======================='

        c = self.config
        img_path = c.processed_path+'/images'
        if not path.isdir(img_path):
            mkdir(img_path)

        for subject in c.subjects:
            subj = subject.name
            tractography_path = subject.tractography_path
            chdir(path.join(tractography_path,subj))
            for method, file_list in stream_map.iteritems():
                method_path = path.join( img_path,method)

                if not path.isdir(method_path):
                    mkdir(method_path)

                for filename in file_list:

                    reader = vtk.vtkPolyDataReader()
                    reader.SetFileName(filename)
                    reader.Update()
                    polydata = reader.GetOutput()

                    streamlines = []
                    for i in range(polydata.GetNumberOfCells()):
                        pts = polydata.GetCell(i).GetPoints()
                        npts = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())])
                        streamlines.append(npts)
                    if len(streamlines) == 0:
                        print '-- No streamlines for %s' % filename
                        continue


                    ren = fvtk.ren()
                    fvtk.clear(ren)
                    fvtk.add(ren, fvtk.streamtube(streamlines, line_colors(streamlines)))
                    ren.SetBackground(0,0,0)

                    #axial
                    cam = fvtk.camera(ren, pos=(0,0,-1), focal=(0,0,0), viewup=(0,1,0))
                    imgfile = self._g(subj, path.splitext(filename)[0],'axial.png', prefix=False, ext=False)
                    imgout = path.join( method_path,imgfile)
                    print 'Save %s' % imgout
                    fvtk.record(ren, n_frames=1, out_path=imgout, size=(1024,768))

                    ren = fvtk.ren()
                    fvtk.clear(ren)
                    fvtk.add(ren, fvtk.streamtube(streamlines, line_colors(streamlines)))
                    ren.SetBackground(0,0,0)

                    cam = fvtk.camera(ren, pos=(0,1,0), focal=(0,0,0), viewup=(0,0,1))
                    imgfile = self._g(subj, path.splitext(filename)[0],'coronal.png', prefix=False, ext=False)
                    imgout = path.join( method_path,imgfile)
                    print 'Save %s' % imgout
                    fvtk.record(ren, n_frames=1, out_path=imgout, size=(1024,768))

                    ren = fvtk.ren()
                    fvtk.clear(ren)
                    fvtk.add(ren, fvtk.streamtube(streamlines, line_colors(streamlines)))
                    ren.SetBackground(0,0,0)


                    cam = fvtk.camera(ren, pos=(-1,0,0), focal=(0,0,0), viewup=(0,0,1))
                    imgfile = self._g(subj, path.splitext(filename)[0],'sag.png', prefix=False, ext=False)
                    imgout = path.join( method_path,imgfile)
                    print 'Save %s' % imgout
                    fvtk.record(ren, n_frames=1, out_path=imgout, size=(1024,768))

                    ren = fvtk.ren()
                    fvtk.clear(ren)
                    fvtk.add(ren, fvtk.streamtube(streamlines, line_colors(streamlines)))
                    ren.SetBackground(0,0,0)


                    cam = fvtk.camera(ren, pos=(-1,1,1), focal=(0,0,0), viewup=(0,0,1))
                    imgfile = self._g(subj, path.splitext(filename)[0],'persp.png', prefix=False, ext=False)
                    imgout = path.join( method_path,imgfile)
                    print 'Save %s' % imgout
                    fvtk.record(ren, n_frames=1, out_path=imgout, size=(1024,768))

        chdir(c.root)






    def tracts_conjunction(self, streamnames, img_type="density", dry_run=False):
        """ Perform conjunction analysis of tracts by reverse project tracts to average T1 template, using density  maps

            Args:
                streamnames:
        """

        c = self.config
        # tract_path = c.tractography_path_full

        image_table = []
        not_found_table = []

        if img_type=='density':
            affix = 'den'
        elif img_type=='fiber':
            affix = 'fib'
        elif img_type=='binary':
            affix = 'bin'

        print '''
        ===================================================
        ╔╦╗┬─┐┌─┐┌─┐┌┬┐┌─┐  ╔═╗┌─┐┌┐┌ ┬┬ ┬┌┐┌┌─┐┌┬┐┬┌─┐┌┐┌
         ║ ├┬┘├─┤│   │ └─┐  ║  │ ││││ ││ │││││   │ ││ ││││
         ╩ ┴└─┴ ┴└─┘ ┴ └─┘  ╚═╝└─┘┘└┘└┘└─┘┘└┘└─┘ ┴ ┴└─┘┘└┘
        ===================================================
        '''
        root = getcwd()

        for subject in c.subjects:
            subj = subject.name
            tract_path = subject.tractography_path


            chdir(path.join(tract_path,subj))
            subj_files = []

            print '--------------------------- %s ----------------------------------' % subj
            for method, file_list in streamnames.iteritems():
                print '== Method %s ' % method
                for tfile in file_list:
                    print '>',tfile
                    # get the volume file associated with the track
                    seed_base = tfile.split('.')[0]
                    seed_name = seed_base[len(method)+1:]
                    seed_name = seed_name.split('_')[0]

                    fext = path.splitext(tfile)
                    den_file = fext[0]+'_%s.nii.gz' % affix
                    print '>',den_file

                    # t1 -> average, file in processed
                    den_t1_file = self._g(fext[0], affix, 'T1', prefix=False)
                    #print '#',den_t1_file
                    output = self._g(subj, path.basename(fext[0]), affix, prefix=False)
                    output = path.join( c.processed_path,output)
                    print '>',output

                    if path.isfile(den_file):
                        trans = ''

                        trans += self.ants_apply_transform_average_to_t1(subj, den_t1_file, output, reference='average', inverse=True, just_transform_param=True)

                        # dwi -> t1 projection first, file stay in the tractography dirs
                        trans += ' '+self.ants_apply_transform_t1_to_dwi(subj, den_file, den_t1_file, reference='t1', inverse=True, just_transform_param=True)

                        ref = c.group_template_file # % subj
                        ref = path.join( c.orig_path,ref)
                        interp = 'NearestNeighbor'

                        cmd=('antsApplyTransforms -d 3 -i %s -o %s -r %s -n %s ' % (den_file, output, ref, interp))
                        cmd += trans
                        if not dry_run:
                            exec_cmd(cmd)

                    else:                        
                        not_found_table.append(path.join(subj,den_file))
                    subj_files.append((output, method, seed_name))

            image_table.append(subj_files)
            chdir(root)

        # for i in not_found_table:
        #     print '!',i
        # return

        #image_table = np.array(image_table, dtype='object')
        image_table = map(list, zip(*image_table))
        for i in image_table:
            print '#'
            for j in i:
                print j



        ref_img_path = path.join(c.orig_path,c.template_def['reference'])
        refimg = nib.load(ref_img_path)

        print '=== Perform conjuction ==='

        conj_files_list=[]

        for r in image_table:            
            if len(r) > 0:
                print '--------------'
                label = path.basename(r[0][0])
                #get the group name, get rid of subject and bin/density image tag
                label = '_'.join(label.split('_')[1:-1])
                data_pool = np.zeros(refimg.get_shape()[:3])

                method = ''
                seed_name = ''

                for i,j in enumerate(r):
                    img_file = j[0]
                    method = j[1]
                    seed_name = j[2]

                    print img_file
                    if not path.isfile(img_file):
                        continue
                    img = nib.load(img_file)
                    idata = img.get_data()
                    data_pool = np.add(data_pool,idata)
                    del img
                    del idata
                data_pool = np.divide(data_pool,len(r))

                filebasename = self._g(label, affix, 'average', prefix=False, ext=False)

                figure_file = filebasename+'_figure.png'
                figure_file = path.join(c.processed_path, figure_file)
                score = imgs.get_score(data_pool, figure=figure_file, title=label)
                

                nifti = nib.Nifti1Image(data_pool, refimg.get_affine())
                conj_file = path.join(c.processed_path, filebasename+'.nii.gz')
                nib.save(nifti, conj_file)
                print '# Save to %s' % conj_file
                conj_files_list.append((conj_file,method,seed_name, score))
                del data_pool
                gc.collect()

        if len(not_found_table) > 0:
            print '=== The following are not found:'
            for i in not_found_table:
                print i

        return conj_files_list



    def conjunction_to_images(self, file_list, slice_indices=(0,0,0), name='', bg_file='' , auto_slice=True,dry_run=False):

        from gts.scripts import vol2iso_viz

        def get_slicing(focus_settings, skey):
                if not isinstance(focus_settings, dict):
                    return focus_settings
                else:
                    if skey in slicing_focus:
                        return focus_settings[skey]
                    if 'all' in slicing_focus:
                        return focus_settings['all']
                return None
                        
        c = self.config

        imgpath = c.processed_path+'/images/conjunctions'
        if not path.isdir(imgpath):
            os.makedirs(imgpath)
        basename=''
        all_imgs = []
        for i in file_list:            
            imgs = []
            filename = i[0]
            print '>',filename
            seed_name = i[2]
            group = {'method':i[1], 'seed':i[2]}            
            seed_conf = c.seeds_def[seed_name]
            if 'slicing_focus' in seed_conf:
                slicing_focus = seed_conf['slicing_focus']
                auto_slice=False

            fig_base = path.basename(filename).split('.')[0]
            print fig_base

            fig_name = self._g(fig_base, name, 'axial.png', prefix=False, ext=False)
            imgs.append(fig_name)
            fig_name = path.join(imgpath,fig_name)
            print fig_name
            if not dry_run:
                slice_indices = get_slicing(slicing_focus, 'axial')
                vol2iso_viz.vol2iso_viz(filename, bg_file, plane_orientation='z_axes', auto_slice=auto_slice, slice_index=slice_indices[2], save_fig=fig_name)

            fig_name = self._g(fig_base, name, 'coronal.png', prefix=False, ext=False)
            imgs.append(fig_name)            
            fig_name = path.join(imgpath,fig_name)            
            print fig_name
            if not dry_run:
                slice_indices = get_slicing(slicing_focus, 'coronal')
                vol2iso_viz.vol2iso_viz(filename, bg_file, plane_orientation='y_axes', auto_slice=auto_slice, slice_index=slice_indices[1],save_fig=fig_name)

            fig_name = self._g(fig_base, name, 'saggital.png', prefix=False, ext=False)
            imgs.append(fig_name)            
            fig_name = path.join(imgpath,fig_name)            
            print fig_name
            if not dry_run:
                slice_indices = get_slicing(slicing_focus, 'saggital')
                vol2iso_viz.vol2iso_viz(filename, bg_file, plane_orientation='x_axes', auto_slice=auto_slice, slice_index=slice_indices[0],save_fig=fig_name)

            fig_name = self._g(fig_base, name, 'perspective.png', prefix=False, ext=False)
            imgs.append(fig_name)            
            fig_name = path.join(imgpath,fig_name)            
            print fig_name
            if not dry_run:
                slice_indices = get_slicing(slicing_focus, 'persp')
                vol2iso_viz.vol2iso_viz(filename, bg_file, plane_orientation='iso', auto_slice=auto_slice, slice_index=slice_indices[2],save_fig=fig_name)
            
            group['images'] = imgs
            basename = fig_base.replace(i[1]+'_'+i[2]+'_','')
            all_imgs.append(group)

        return all_imgs, basename

    def conjunction_images_combine(self, images_list, group_names=['bg', 'nobg'], basename='filtered_bin_average'):
        print '''
        ============================================================
        ╔═╗┌─┐┌┐┌ ┬┬ ┬┌┐┌┌─┐┌┬┐  ╦┌┬┐┌─┐┌─┐┌─┐  ╔═╗┌─┐┌┬┐┌┐ ┬┌┐┌┌─┐
        ║  │ ││││ ││ │││││   │   ║│││├─┤│ ┬├┤   ║  │ ││││├┴┐││││├┤ 
        ╚═╝└─┘┘└┘└┘└─┘┘└┘└─┘ ┴   ╩┴ ┴┴ ┴└─┘└─┘  ╚═╝└─┘┴ ┴└─┘┴┘└┘└─┘
        ============================================================
        '''

        c = self.config
        img_path = c.processed_path+'/images'
        img_conjpath = path.join(img_path,'conjunctions')

        methods = list(OrderedSet([i['method'] for i in images_list]))
        methods.sort()
        
        print methods
        rois = list(OrderedSet([i['seed'] for i in images_list]))

        num_methods = len(methods)
        num_rois = len(rois)

        # compile a matrix of images

        bgOpt = group_names
        orientOpt = ['axial', 'coronal', 'saggital', 'perspective']


        res = (1024,768)

        for bg in bgOpt:
            for roi in rois:
                img_matrix = []
                longest_label = ''
                for method in methods:
                    mlabel = method
                    if len(mlabel) > len(longest_label):
                        longest_label = mlabel

                    for ori in orientOpt:
                        filename=self._g(mlabel,roi,basename,bg,ori+'.png', prefix=False, ext=False)
                        filename = path.join(img_conjpath,filename)
                        #files = glob(mlabel+'*'+roi+'*'+bg+'*'+ori+'*')
                        #for f in files:
                        img_matrix.append(filename)
                img_matrix = np.array(img_matrix).reshape(num_methods, len(orientOpt)).T


                img_name = self._g(roi,basename,bg+'.png', prefix=False, ext=False)
                print '> Creating %s' % img_name

                fontPath = "/usr/share/fonts/truetype/droid/DroidSans.ttf"
                f  =  ImageFont.truetype ( fontPath, 72 )
                new_img = Image.new('RGB', res)
                draw = ImageDraw.Draw(new_img)
                max_label_size=draw.textsize(longest_label, font=f)

                label_padding = max_label_size[0]+50
                label_padding_y = max_label_size[1]+50
                img_width = res[0]*img_matrix.shape[0] + label_padding
                img_height = res[1]*img_matrix.shape[1] + label_padding_y
                new_img = Image.new('RGB', (img_width, img_height))



                for (x,y), val in np.ndenumerate(img_matrix):
                    print x,y,val
                    try:
                        xpos = res[0]*x + label_padding
                        ypos = res[1]*y + label_padding_y
                        im = Image.open(val)
                        new_img.paste(im, (xpos, ypos))
                    except IOError:
                        print '! No such file {}'.format(val)

                # draw method labels on left side
                for i in range(0,num_methods):
                    text = methods[i]
                    print text
                    xpos = 10
                    ypos = res[1]*i+res[1]/2
                    draw = ImageDraw.Draw(new_img)

                    text_size=draw.textsize(text, font=f)
                    draw.text((xpos,ypos-text_size[1]/2), text, fill="white", font=f)

                # draw orientation texts at top
                for i,orient in enumerate(orientOpt):
                    text = orient 
                    print text
                    xpos = res[0]*i+res[0]/2
                    ypos = 10
                    draw = ImageDraw.Draw(new_img)

                    text_size=draw.textsize(text, font=f)
                    draw.text((xpos+text_size[0]/2,ypos), text, fill="white", font=f)                    

                new_img.save(path.join(img_path,img_name))
