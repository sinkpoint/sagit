#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:43:13 2014

@author: dchen
"""

import sys
import json
import argparse
import numpy as np
import shutil
import nrrd
from glob import glob
from os import getcwd
from os import path
from os import chdir
from os import mkdir

import nibabel as nib
from nibabel import trackvis as tv
from gtsutils import exec_cmd
#from dtproc import *



# def matchSpacing(reference, moving, output):
#     ref = nib.load(reference)
#     ref_zooms = ref.get_header().get_zooms()[:3]

#     mov = nib.load(moving)
#     mov_data = mov.get_data()
#     mov_zooms = mov.get_header().get_zooms()[:3]
#     mov_affine = mov.get_affine()

#     # reslice into reference spacing
#     data, affine = resample(mov_data, mov_affine, mov_zooms, ref_zooms)
#     img = nib.Nifti1Image(data, affine)

#     print data.shape
#     print img.get_header().get_zooms()
#     print "###"

#     nib.save(img, output)

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
        from gts import GtsConfig
        self.config = GtsConfig(conf_file)

    def _g(self, *args, **kwargs):
        return getGtsFilename(self.config, *args, **kwargs)

    def _d(self, subj, basename, ref=""):
        return getGtsFilename(self.config, subj, basename, ref, ext=False)


    def projectRoiT1TemplateToSingle(self):
        """ Project averaged T1 ROI to indidivual T1 space
        """

        c = self.config
        orig_path = c.orig_path
        imgext = c.imgext
        prefix = c.prefix
        T1_processed_path = c.T1_processed_path
        template_rois= c.template_def
        ind_roi_path = c.ind_roi_path

        for subj in c.subjects:
            print '------------------------------projectTemplateToSingle------'
            print subj
            subj_T1 = orig_path+'/'+subj+'_T1'+imgext

            subj_pref = T1_processed_path+'/'+self._g(subj, ext=False)

            warpAffix = 'InverseWarp.nii.gz'
            affAffix = 'Affine.txt'

            if len(glob(subj_pref+'*'+affAffix)) == 0:
                # can't find affine.txt, assume it's mat
                warpAffix = '1InverseWarp.nii.gz'
                affAffix = '0GenericAffine.mat'

            subj_warp = subj_pref+warpAffix
            subj_affine = subj_pref+affAffix


            for rname, filename in template_rois.iteritems():
                if rname == 'reference':
                    continue

                basename = filename.split('.')[0]
                roi = c.template_roi_path+'/'+basename
                roiName = basename(roi).rsplit('.')[0]
                inName = roi+imgext
                outName = self._g(subj, roiName)
                output = ind_roi_path+'/'+outName

                cmd="antsApplyTransforms -d 3 -i %s -o %s -r %s -n NearestNeighbor -t [%s,1] -t %s" % (inName, output, subj_T1,  subj_affine, subj_warp)

                exec_cmd(cmd)

                #cmd="fslmaths %s -kernel sphere 1 -dilM %s" % (output, output)
                #exec_cmd(cmd)

                #dir_roi = subject_dti_path+'/'+subj+'/'+outName
                #print 'copy %s to %s' %( output, dir_roi )
                #shutil.copyfile(output, dir_roi)

    # project individual ROI from T1 space to DWI space
    def projectRoiT1ToDwi(self):
        c = self.config
        orig_path = c.orig_path
        imgext = c.imgext
        prefix = c.prefix
        affix = c.affix
        T1_processed_path = c.T1_processed_path
        processed_path = c.processed_path
        ind_roi_path = c.ind_roi_path

        rois = []
        for k,v in c.template_def:
            if k=='reference':
                continue
            rois.append(v)



        self.projectImageT1ToDwi(ind_roi_path, rois, reference='dwi')
        self.projectImageT1ToDwi(ind_roi_path, rois, reference='t1')

        for subj in c.subjects:
            fspath = c.subject_freesurfer_path+'/'+subj+'/mri'
            fs_t1_file = c.preprocessed_path +'/'+ self._g(subj,'fs','T1', prefix=False)
            fs_aseg_file = c.preprocessed_path +'/'+ self._g(subj,'fs','aseg', prefix=False)
            fs2fsl_mat = c.preprocessed_path +'/'+ self._g(subj,'fs','fs2fsl','mat.txt', prefix=False, ext=False)
            subj_aseg = c.preprocessed_path +'/'+ self._g(subj,'aseg', prefix=False)
            subj_t1 = c.orig_path+'/'+self._g(subj,'T1', prefix=False)
            dwi_aseg = c.preprocessed_path+'/'+self._g(subj, 'aseg', 'dwi', prefix=False)

            print '#',fs2fsl_mat
            if not path.isfile(fs2fsl_mat):
                cmd = 'mri_convert %s/T1.mgz %s' % (fspath, fs_t1_file)
                exec_cmd(cmd)

                cmd = 'mri_convert %s/aparc+aseg.mgz %s' % (fspath, fs_aseg_file)
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
        orig_path = c.orig_path
        imgext = c.imgext
        prefix = c.prefix
        affix = c.affix
        processed_path = c.processed_path

        if output_path=='':
            output_path = processed_path

        for subj in c.subjects:
            print '------------------------------projectImageT1ToDwi-----------------------------------'
            print subj

            for name in name_base:
                inName = input_path+'/'+self._g(subj, name)
                outName = self._d(subj, name)
                output = output_path+'/'+outName



                self.ants_apply_transform_t1_to_dwi(subj, inName, output, **kwargs)

                #dir_roi = subject_dti_path+'/'+subj+'/'+outName
                #print 'copy %s to %s' %( output, dir_roi )
                #shutil.copyfile(output, dir_roi)

    def ants_apply_transform_t1_to_dwi(self, subj, input, output, inverse=False, reference='dwi', interp='NearestNeighbor', just_transform_param=False, auto_naming=True):

        c = self.config
        orig_path = c.orig_path
        imgext = c.imgext
        prefix = c.prefix
        affix = c.affix
        processed_path = c.processed_path

        subj_pref = self._g(subj, ext=False)


        ref = '%s_FA.nii.gz' % subj
        if reference=='t1':
            ref = '%s_T1.nii.gz' % subj

        if auto_naming:
            output = output+'_'+reference+imgext
        #print '######t1_to_dwi#######',output

        ref = orig_path+'/'+ref

        invWarpAffix = 'InverseWarp.nii.gz'
        warpAffix = 'Warp.nii.gz'
        affAffix = 'Affine.txt'

        if len(glob(subj_pref+'*'+affAffix)) == 0:
            # can't find affine.txt, assume it's mat
            invWarpAffix = '1InverseWarp.nii.gz'
            warpAffix = '1Warp.nii.gz'
            affAffix = '0GenericAffine.mat'

        subj_invWarp = processed_path+'/'+subj_pref+invWarpAffix
        subj_warp = processed_path+'/'+subj_pref+warpAffix
        subj_affine = processed_path+'/'+subj_pref+affAffix

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

    def ants_apply_transform_average_to_t1(self, subj, input, output, reference='t1', interp='NearestNeighbor', inverse=False,just_transform_param=False):

        c = self.config
        orig_path = c.orig_path
        imgext = c.imgext
        prefix = c.prefix
        affix = c.affix
        processed_path = c.T1_processed_path

        subj_pref = self._g(subj,ext=False)

        ref = '%s_T1.nii.gz' % subj
        if reference=='average':
            ref = 'con_average.nii.gz'
        ref = orig_path+'/'+ref

        invWarpAffix = 'InverseWarp.nii.gz'
        warpAffix = 'Warp.nii.gz'
        affAffix = 'Affine.txt'

        if len(glob(processed_path+'/'+subj_pref+'*'+affAffix)) == 0:
            # can't find affine.txt, assume it's mat
            invWarpAffix = '1InverseWarp.nii.gz'
            warpAffix = '1Warp.nii.gz'
            affAffix = '0GenericAffine.mat'

        subj_invWarp = processed_path+'/'+subj_pref+invWarpAffix
        subj_warp = processed_path+'/'+subj_pref+warpAffix
        subj_affine = processed_path+'/'+subj_pref+affAffix

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
        dwi_path = c.dwi_base_path
        root_path = getcwd()
        for subj in c.subjects:
            print '--------------------------------preprocessDWI---------------------------------'
            subj_path = dwi_path+subj
            chdir(subj_path)
            dwi_folder = glob("*DTI*60*")[0]
            chdir(dwi_folder+'/nifti')
            print getcwd()

            #if not path.isfile("vol0000.nii.gz"):
                #print "vol not found"
                #cmd="fslsplit Motion_Corrected_DWI_nobet.nii.gz"
                #exec_cmd(cmd)

                #cmd="slicerFileConvert.sh -i vol0000.nii.gz -o %s/%s_b0.nii.gz" % (orig_path, subj)
                #
            cmd="fslmaths Motion_Corrected_DWI_nobet.nii.gz -Tmean %s" % (c.orig_path+'/'+subj+'_mdwi')
            exec_cmd(cmd)

            cmd="slicerTensorScalar.sh -i DWI_CORRECTED.nhdr -p %s_ -d %s" % (subj,c.orig_path)
            exec_cmd(cmd)

            chdir(root_path)

            shutil.copyfile('T1s/orig/'+subj+'.nii.gz', c.orig_path+'/'+subj+'_T1.nii.gz')

        chdir(root_path)

    def runAntsDwiToT1(self,bet='0.1'):
        """ Use ANTS to register individual DWI space to T1 space.
            For now registers average dwi to T1.

            Args:
                bet: The -f parameter value for FSL bet command. Default is 0.1.
        """
        c = self.config
        preprocessed_path = c.preprocessed_path
        root_path = getcwd()
        orig_path = c.orig_path

        betfiles = ""
        #_SIMULATE = True
        for subj in c.subjects:
            chdir(orig_path)
            print '-----------------------------runAntsDwiToT1------------------------------------'
            print getcwd()

            cmd='bet2 %s_T1 %s_T1_bet -f %s ' % (subj, subj, bet)
            exec_cmd(cmd)

            _SIMULATE = True

            cmd='bet2 %s_mdwi %s_MDWI_bet -f %s' % (subj, subj, bet)
            exec_cmd(cmd)

            # erode the mask to remove fringe skull intensities in FA
            # cmd='fslmaths %s_MDWI_bet_mask -kernel sphere 4 -ero %s_b0_bet_mask' % (subj, subj)
            # exec_cmd(cmd)

            # cmd='fslmaths %s_FA.nii.gz -mul %s_b0_bet_mask.nii.gz %s_FA_bet.nii.gz' % (subj, subj, subj)
            # exec_cmd(cmd)

            _SIMULATE = False

            #matchSpacing("%s_MDWI_bet.nii.gz"%subj, "%s_T1_bet.nii.gz"%subj, "%s_T1_bet.nii.gz"%subj)

            cmd='mv *bet* %s' % (preprocessed_path)

            exec_cmd(cmd)

            betfiles += "%s_T1_bet.nii.gz " % subj
            betfiles += "%s_MDWI_bet.nii.gz " % subj
            chdir(root_path)

        chdir(preprocessed_path)
        cmd='slicesdir '+betfiles
        exec_cmd(cmd)

        chdir(root_path)
        antsparam = ""
        if c.manual_subjects:
            antsparam = " ".join(c.subjects)

        cmd='./research/runAntsFAToT1.sh %s' % antsparam
        exec_cmd(cmd)


    def seedIndividualTracts(self, labels=[1],recompute=False,overwrite=False, reorganize_paths=False):

        print '===========================seedIndividualTracts============================'        
        c = self.config
        origin=getcwd()
        dwi_path = c.dwi_base_path

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


        for subj in c.subjects:
            print '-----------------------------------------------------------------'
            print subj
            subjdir = tractography_path+'/'+subj+'/'
            subjRes = { 'name' : subj }

            if not path.isdir(subjdir):
                ## setup subject tractography directory

                mkdir(subjdir)


                subj_source_path = path.join(dwi_path,subj)
                chdir(subj_source_path)

                dwi_folder = glob("*DTI*60*")[0]
                chdir(dwi_folder+'/nifti')

                subj_dwi_source_path_abs = getcwd()
                chdir(subjdir)

                cmd = 'ln -s %s dwi' % (subj_dwi_source_path_abs)
                exec_cmd(cmd)


                print 'copy DWI file to '+subjdir

                # copy dwi_corrected to tractography/subj dir
                # correct this to RAS, because xst doesn't understand LPS 

                dwi_file = 'DWI_CORRECTED.nhdr'
                slicer_comp_file = path.join(subjdir,dwi_file)

                cmd = 'slicerDwiFileConvert.sh %s %s' % (path.join(subj_dwi_source_path_abs, dwi_file), slicer_comp_file)
                exec_cmd(cmd)

                ras_corrected_file = subj+'_dwi_ras.nhdr'
                reader = nrrd.NrrdReader()
                header = reader.getFileAsHeader(slicer_comp_file)
                # once slicer reads the file, it will convert the file it's a space that it understands                
                header.correctSpaceRas() # convert spacing to RAS, this is needed for xst, else geometry will be inverted.
                writer = nrrd.NrrdWriter()
                writer.write(header, subjdir+'/'+ras_corrected_file)

                #dwiCorrectRas(dwi_file+'.nhdr', subj+'_test.nhdr')


                # generate Tensor, FA, RD, AD, MD maps
                cmd="slicerTensorScalar.sh -i %s -p %s_ -d %s" % (ras_corrected_file, subj,subjdir)

                exec_cmd(cmd)


                # copy t1s over
                t1i = self._g(subj, 'invDeformed', 'DWS', prefix=False, ext=False)
                t1o = self._g(subj, 'T1', 'dwi', prefix=False)
                shutil.copyfile(processed_path+'/'+t1i, subjdir+'/'+t1o)

                t1i = self._g(subj, 'invDeformed', 'T1')
                t1o = self._g(subj, 'T1', prefix=False)
                shutil.copyfile(processed_path+'/'+t1i, subjdir+'/'+t1o)

                # copy over mdwi files, this should be in preprocessing step
                t1i = self._g(subj, 'mdwi', prefix=False)
                t1o = t1i
                shutil.copyfile(orig_path+'/'+t1i, subjdir+'/'+t1o)

                # copy freesurfer segmentation to tractography folder
                dwi_aseg = self._g(subj, 'aseg', 'dwi', prefix=False)
                shutil.copyfile(c.preprocessed_path+'/'+dwi_aseg, subjdir+'/'+dwi_aseg)                

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

            rois = c.rois_def
            for k, roi in rois.iteritems():
                if roi.type == 'from_template':
                    roi_file = roi.get_filename(subj)

                    shutil.copyfile(processed_path+'/'+roi_file, subjdir+'/'+roi_file)
                    roi_filebase = roi_file.split('.')[0]

                    cmd="slicerFileConvert.sh -i %s -o %s " % (roi_file, roi_filebase+'.nhdr');
                    exec_cmd(cmd, display=False)

                    roi_file = roi.get_filename(subj, ref='t1')
                    shutil.copyfile(processed_path+'/'+roi_file, subjdir+'/'+roi_file)
            stream_map = {}
            fiber_name = ''
            method_name = ''


            ############ Pass on methods list and generate tracts

            from gtstractography import TractographyMethod
            seeds = c.seeds_def
            for k,seed_map in seeds.iteritems():
                label_str = k
                seed_map['name'] = k

                for imethod in c.tract_method:
                    method_label  = imethod['label']


                    print '\n-- PERFORM %s' % method_label
                    method = TractographyMethod.factory(subj, seed_map, imethod, c)
                    fiber_name = method.run()


                    #     if not mrtrix_has_recompute:
                    #         recompute = True
                    #         mrtrix_has_recompute = True
                    #     else:
                    #         recompute = False
                        


                    #     fiber_name = gtracts.tractsMrtrix(subj, roibase, excludebase, ilabel, config=c, label_str=label_str, label_include=c.roi_includes[i],
                    #                  label_exclude=c.roi_excludes[i], recompute=recompute, overwrite=overwrite, params=iparam, method_label=method_label)

                    # elif itract == 'xst':
                    #     print '\n-- PERFORM XST'
                    #     fiber_name = gtracts.tractsXst(subj, roibase, excludebase, ilabel, dwi_file=ras_corrected_file, config=c, label_str=label_str,
                    #               label_include=c.roi_includes[i],label_exclude=c.roi_excludes[i], overwrite=overwrite, params=iparam)
                    # elif itract =='slicer':
                    #     print '\n-- PERFORM SLICER'
                    #     fiber_name = gtracts.tractsSlicer(subj, roibase, excludebase, ilabel, config=c, label_str=label_str,
                    #               label_include=c.roi_includes[i],label_exclude=c.roi_excludes[i], overwrite=overwrite, params=iparam)

                    if not stream_map.has_key(method_label):
                            stream_map[method_label] = [fiber_name]
                    else:
                        stream_map[method_label].append(fiber_name)



            chdir(origin)
        return stream_map

    def tracts_to_density(self, tracts_map):
        import gts.tractDensityMap as tdm
        print '===================== TRACTS TO DENSITY ======================='
        c = self.config
        for subj in c.subjects:
            chdir(c.tractography_path+'/'+subj)
            print '========== %s =========' % subj

            ref_file = 'ANTS_%s_invDeformed_T1.nii.gz' % subj
            ref_file = c.processed_path+'/' + ref_file

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
        import vtk
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
        tractography_path = c.tractography_path

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

                for subj in c.subjects:
                    subj_tractography_path = tractography_path+'/'+subj+'/'
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
                            'MDWI': ['%s_mdwi.nii.gz' % volume_base, labelmap_dwi] }

                        fname = f
                        if tmethod=='xst':
                            fname = f.split('/')[1]
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
        import vtk
        from dipy.viz.colormap import line_colors
        from dipy.viz import fvtk

        print '===================== PERFORM tracts_to_images ======================='

        c = self.config
        img_path = c.processed_path+'/images'
        if not path.isdir(img_path):
            mkdir(img_path)

        for subj in c.subjects:
            chdir(c.tractography_path_full+'/'+subj)
            for method, file_list in stream_map.iteritems():
                method_path = img_path+'/'+method

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
                    imgout = method_path+'/'+imgfile
                    print 'Save %s' % imgout
                    fvtk.record(ren, n_frames=1, out_path=imgout, size=(1024,768))

                    ren = fvtk.ren()
                    fvtk.clear(ren)
                    fvtk.add(ren, fvtk.streamtube(streamlines, line_colors(streamlines)))
                    ren.SetBackground(0,0,0)

                    cam = fvtk.camera(ren, pos=(0,1,0), focal=(0,0,0), viewup=(0,0,1))
                    imgfile = self._g(subj, path.splitext(filename)[0],'coronal.png', prefix=False, ext=False)
                    imgout = method_path+'/'+imgfile
                    print 'Save %s' % imgout
                    fvtk.record(ren, n_frames=1, out_path=imgout, size=(1024,768))

                    ren = fvtk.ren()
                    fvtk.clear(ren)
                    fvtk.add(ren, fvtk.streamtube(streamlines, line_colors(streamlines)))
                    ren.SetBackground(0,0,0)


                    cam = fvtk.camera(ren, pos=(-1,0,0), focal=(0,0,0), viewup=(0,0,1))
                    imgfile = self._g(subj, path.splitext(filename)[0],'sag.png', prefix=False, ext=False)
                    imgout = method_path+'/'+imgfile
                    print 'Save %s' % imgout
                    fvtk.record(ren, n_frames=1, out_path=imgout, size=(1024,768))

                    ren = fvtk.ren()
                    fvtk.clear(ren)
                    fvtk.add(ren, fvtk.streamtube(streamlines, line_colors(streamlines)))
                    ren.SetBackground(0,0,0)


                    cam = fvtk.camera(ren, pos=(-1,1,1), focal=(0,0,0), viewup=(0,0,1))
                    imgfile = self._g(subj, path.splitext(filename)[0],'persp.png', prefix=False, ext=False)
                    imgout = method_path+'/'+imgfile
                    print 'Save %s' % imgout
                    fvtk.record(ren, n_frames=1, out_path=imgout, size=(1024,768))

        chdir(c.root)






    def tracts_conjunction(self, streamnames, img_type="density"):
        """ Perform conjunction analysis of tracts by reverse project tracts to average T1 template, using density  maps

            Args:
                streamnames:
        """

        c = self.config
        tract_path = c.tractography_path_full

        image_table = []
        not_found_table = []

        if img_type=='density':
            affix = 'den'
        elif img_type=='fiber':
            affix = 'fib'
        elif img_type=='binary':
            affix = 'bin'

        print '===================== PERFORM CONJUNCTION ======================='
        root = getcwd()

        for subj in c.subjects:
            chdir(tract_path+'/'+subj)
            subj_files = []

            print '--------------------------- %s ----------------------------------' % subj
            for method, file_list in streamnames.iteritems():
                print '== Method %s ' % method
                for tfile in file_list:
                    # get the volume file associated with the track
                    fext = path.splitext(tfile)
                    den_file = fext[0]+'_%s.nii.gz' % affix
                    print '>',den_file

                    # t1 -> average, file in processed
                    den_t1_file = self._g(fext[0], affix, 'T1', prefix=False)
                    #print '#',den_t1_file
                    output = self._g(subj, path.basename(fext[0]), affix, prefix=False)
                    output = c.processed_path+'/'+output
                    print '>',output

                    if path.isfile(den_file):
                        trans = ''

                        trans += self.ants_apply_transform_average_to_t1(subj, den_t1_file, output, reference='average', inverse=True, just_transform_param=True)

                        # dwi -> t1 projection first, file stay in the tractography dirs
                        trans += ' '+self.ants_apply_transform_t1_to_dwi(subj, den_file, den_t1_file, reference='t1', inverse=True, just_transform_param=True)

                        ref = 'con_average.nii.gz' # % subj
                        ref = c.orig_path+'/'+ref
                        interp = 'NearestNeighbor'

                        cmd=('antsApplyTransforms -d 3 -i %s -o %s -r %s -n %s ' % (den_file, output, ref, interp))
                        cmd += trans
                        exec_cmd(cmd)

                    else:                        
                        not_found_table.append(subj+'/'+den_file)
                    subj_files.append(output)

            image_table.append(subj_files)
            chdir(root)

        # for i in not_found_table:
        #     print '!',i
        # return

        #image_table = np.array(image_table, dtype='object')
        image_table = map(list, zip(*image_table))
        # for i in image_table:
        #     print '#'
        #     for j in i:
        #         print j


        import nibabel as nib
        import gc

        ref_img_path = path.join(c.orig_path,c.template_def['reference'])
        refimg = nib.load(ref_img_path)

        print '=== Perform conjuction ==='

        conj_files_list=[]

        for r in image_table:            
            if len(r) > 0:
                print '--------------'
                label = path.basename(r[0])
                #get the group name, get rid of subject and bin/density image tag
                label = '_'.join(label.split('_')[1:-1])
                data_pool = np.zeros(refimg.get_shape()[:3])

                for i,j in enumerate(r):
                    print j
                    if not path.isfile(j):
                        continue
                    img = nib.load(j)
                    idata = img.get_data()
                    data_pool = np.add(data_pool,idata)
                    del img
                    del idata
                data_pool = np.divide(data_pool,len(r))
                nifti = nib.Nifti1Image(data_pool, refimg.get_affine())
                conj_file = path.join(c.processed_path, label+'_%s_average.nii.gz' % affix)
                nib.save(nifti, conj_file)
                print '# Save to %s' % conj_file
                conj_files_list.append(conj_file)
                del data_pool
                gc.collect()

        if len(not_found_table) > 0:
            print '=== The following are not found:'
            for i in not_found_table:
                print i

        return conj_files_list



    def conjunction_to_images(self, file_list, slice_indices=(0,0,0), name='', bg_file='' , auto_slice=True):
        import vol2iso_viz
        c = self.config

        imgpath = c.processed_path+'/images'
        for filename in file_list:
            fig_base = path.basename(filename).split('.')[0]
            fig_name = imgpath+'/'+self._g(fig_base, name, 'axial.png', prefix=False, ext=False)
            print fig_name
            vol2iso_viz.vol2iso_viz(filename, bg_file, plane_orientation='z_axes', auto_slice=auto_slice, slice_index=slice_indices[2], save_fig=fig_name)

            fig_name = imgpath+'/'+self._g(fig_base, name, 'coronal.png', prefix=False, ext=False)
            print fig_name
            vol2iso_viz.vol2iso_viz(filename, bg_file, plane_orientation='y_axes', auto_slice=auto_slice, slice_index=slice_indices[1],save_fig=fig_name)

            fig_name = imgpath+'/'+self._g(fig_base, name, 'sag.png', prefix=False, ext=False)
            print fig_name
            vol2iso_viz.vol2iso_viz(filename, bg_file, plane_orientation='x_axes', auto_slice=auto_slice, slice_index=slice_indices[0],save_fig=fig_name)

            fig_name = imgpath+'/'+self._g(fig_base, name, 'isometric.png', prefix=False, ext=False)
            print fig_name
            vol2iso_viz.vol2iso_viz(filename, bg_file, plane_orientation='iso', auto_slice=auto_slice, slice_index=slice_indices[2],save_fig=fig_name)

    def conjunction_images_combine(self):
        import Image
        import ImageDraw
        import ImageFont

        print '===================== PERFORM conjunction_images_combine ======================='
        c = self.config
        imgpath = c.processed_path+'/images'
        num_methods = len(c.tract_method)
        num_rois = len(c.seeds_def.keys())

        # compile a matrix of images
        chdir(imgpath)

        bgOpt = ['bg', 'nobg']
        orientOpt = ['axial', 'coronal', 'sag', 'isometric']

        res = (1024,768)

        for bg in bgOpt:
            for roi, val in c.seeds_def.iteritems():
                img_matrix = []
                longest_label = ''
                for method in c.tract_method:
                    mlabel = method['label']
                    if len(mlabel) > len(longest_label):
                        longest_label = mlabel

                    for ori in orientOpt:
                        filename=self._g(mlabel,roi,'filtered_bin_average',bg,ori+'.png', prefix=False, ext=False)

                        #files = glob(mlabel+'*'+roi+'*'+bg+'*'+ori+'*')
                        #for f in files:
                        img_matrix.append(filename)
                img_matrix = np.array(img_matrix).reshape(num_methods, len(orientOpt)).T

                img_name = self._g(roi,bg+'.png', prefix=False, ext=False)
                print 'Creating %s' % img_name

                fontPath = "/usr/share/fonts/truetype/droid/DroidSans.ttf"
                f  =  ImageFont.truetype ( fontPath, 72 )
                new_img = Image.new('RGB', res)
                draw = ImageDraw.Draw(new_img)
                max_label_size=draw.textsize(longest_label, font=f)

                label_padding = max_label_size[0]+50
                img_width = res[0]*img_matrix.shape[0] + label_padding
                img_height = res[1]*img_matrix.shape[1]
                new_img = Image.new('RGB', (img_width, img_height))

                for (x,y), val in np.ndenumerate(img_matrix):
                    print x,y,val
                    xpos = res[0]*x + label_padding
                    ypos = res[1]*y
                    im = Image.open(val)
                    new_img.paste(im, (xpos, ypos))

                for i in range(0,num_methods):
                    text = c.tract_method[i]['label']
                    print text
                    xpos = 10
                    ypos = res[1]*i+res[1]/2
                    draw = ImageDraw.Draw(new_img)

                    text_size=draw.textsize(text, font=f)
                    draw.text((xpos,ypos-text_size[1]/2), text, fill="white", font=f)

                for i,orient in enumerate(orientOpt):
                    text = orient 
                    print text
                    xpos = res[0]*i+res[0]/2
                    ypos = 10
                    draw = ImageDraw.Draw(new_img)

                    text_size=draw.textsize(text, font=f)
                    draw.text((xpos+text_size[0]/2,ypos), text, fill="white", font=f)                    

                new_img.save(img_name)





        chdir(c.root)
















