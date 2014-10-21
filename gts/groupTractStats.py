#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:43:13 2014

@author: dchen
"""

import sys
import os
import json
import argparse
from os.path import basename
import numpy as np
import shutil
import nrrd
from glob import glob

import nibabel as nib
from nibabel import trackvis as tv

from gtsconfig import GtsConfig
from gtsutils import exec_cmd
import gtstractography as gtracts

from vtk import *
import tractconverter
from dtproc import *



def matchSpacing(reference, moving, output):
    ref = nib.load(reference)
    ref_zooms = ref.get_header().get_zooms()[:3]

    mov = nib.load(moving)
    mov_data = mov.get_data()
    mov_zooms = mov.get_header().get_zooms()[:3]
    mov_affine = mov.get_affine()

    # reslice into reference spacing
    data, affine = resample(mov_data, mov_affine, mov_zooms, ref_zooms)
    img = nib.Nifti1Image(data, affine)

    print data.shape
    print img.get_header().get_zooms()
    print "###"

    nib.save(img, output)        



class GroupTractStats:
    def __init__(self, conf_file=''):
        self.config = GtsConfig(conf_file)

    def getDwiRoiName(self,subj,basename, ref=""):
        c = self.config
        name=c.prefix+'_'+subj+'_'+basename
        if ref!="":
            name+='_'+ref
        return name

    def getGtsFilename(self, prefix=True, affix=True, *args):
        c = self.config
        if prefix:
            name=c.prefix
        else:
            name=args[0]
            args = args[1:]
        for i in args:
            name+='_'+i
        if affix:
            name += c.imgext

        return name


    def projectRoiT1TemplateToSingle(self):
        """ Project averaged T1 ROI to indidivual T1 space
        """

        c = self.config
        orig_path = c.orig_path
        imgext = c.imgext
        prefix = c.prefix
        T1_processed_path = c.T1_processed_path
        template_rois= c.template_rois
        ind_roi_path = c.ind_roi_path

        for subj in c.subjects:
            print '------------------------------projectTemplateToSingle------'
            print subj
            subj_T1 = orig_path+'/'+subj+'_T1'+imgext

            subj_pref = T1_processed_path+'/'+self.getGtsFilename(True, False, subj)

            warpAffix = 'InverseWarp.nii.gz'
            affAffix = 'Affine.txt'

            if len(glob(subj_pref+'*'+affAffix)) == 0:
                # can't find affine.txt, assume it's mat
                warpAffix = '1InverseWarp.nii.gz'
                affAffix = '0GenericAffine.mat'

            subj_warp = subj_pref+warpAffix
            subj_affine = subj_pref+affAffix


            for roi in template_rois:
                roi = c.template_roi_path+'/'+roi
                roiName = basename(roi).rsplit('.')[0]
                inName = roi+imgext
                outName = self.getGtsFilename(True, True, subj, roiName)
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
        rois= c.rois
        ind_roi_path = c.ind_roi_path

        self.projectImageT1ToDwi(ind_roi_path, rois, reference='dwi')
        self.projectImageT1ToDwi(ind_roi_path, rois, reference='t1')


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
                inName = input_path+'/'+self.getGtsFilename(True, True, subj, name)
                outName = self.getDwiRoiName(subj, name)
                output = output_path+'/'+outName



                self.ants_apply_transform_t1_to_dwi(subj, inName, output, **kwargs)

                #dir_roi = subject_dti_path+'/'+subj+'/'+outName
                #print 'copy %s to %s' %( output, dir_roi )
                #shutil.copyfile(output, dir_roi)

    def ants_apply_transform_t1_to_dwi(self, subj, input, output, inverse=False, reference='dwi', interp='NearestNeighbor', just_transform_param=False):

        c = self.config
        orig_path = c.orig_path
        imgext = c.imgext
        prefix = c.prefix
        affix = c.affix
        processed_path = c.processed_path

        subj_pref = self.getGtsFilename(True, False, subj)


        ref = '%s_FA.nii.gz' % subj
        if reference=='t1':
            ref = '%s_T1.nii.gz' % subj
        
        output = output+'_'+reference+imgext
        print '######t1_to_dwi#######',output

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

        subj_pref = self.getGtsFilename(True, False, subj)

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
        root_path = os.getcwd()
        for subj in c.subjects:
            print '--------------------------------preprocessDWI---------------------------------'
            subj_path = dwi_path+subj
            os.chdir(subj_path)
            dwi_folder = glob("*DTI*60*")[0]
            os.chdir(dwi_folder+'/nifti')
            print os.getcwd()

            #if not os.path.isfile("vol0000.nii.gz"):
                #print "vol not found"
                #cmd="fslsplit Motion_Corrected_DWI_nobet.nii.gz"                
                #exec_cmd(cmd)

                #cmd="slicerFileConvert.sh -i vol0000.nii.gz -o %s/%s_b0.nii.gz" % (orig_path, subj)
                #
            cmd="fslmaths Motion_Corrected_DWI_nobet.nii.gz -Tmean %s" % (c.orig_path+'/'+subj+'_mdwi')
            exec_cmd(cmd)

            cmd="slicerTensorScalar.sh -i DWI_CORRECTED.nhdr -p %s_ -d %s" % (subj,c.orig_path)
            exec_cmd(cmd)

            os.chdir(root_path)

            shutil.copyfile('T1s/orig/'+subj+'.nii.gz', c.orig_path+'/'+subj+'_T1.nii.gz')

        os.chdir(root_path)

    def runAntsDwiToT1(self,bet='0.1'):
        """ Use ANTS to register individual DWI space to T1 space.
            For now registers average dwi to T1. 

            Args:
                bet: The -f parameter value for FSL bet command. Default is 0.1.
        """
        c = self.config
        preprocessed_path = c.preprocessed_path
        root_path = os.getcwd()
        orig_path = c.orig_path

        betfiles = ""
        #_SIMULATE = True
        for subj in c.subjects:
            os.chdir(orig_path)
            print '-----------------------------runAntsDwiToT1------------------------------------'
            print os.getcwd()

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
            os.chdir(root_path)

        os.chdir(preprocessed_path)
        cmd='slicesdir '+betfiles
        exec_cmd(cmd)

        os.chdir(root_path)
        antsparam = ""
        if c.manual_subjects:
            antsparam = " ".join(c.subjects)

        cmd='./research/runAntsFAToT1.sh %s' % antsparam
        exec_cmd(cmd)


    def seedIndividualTracts(self, labels=[1],recompute=False,overwrite=False):
        c = self.config
        origin=os.getcwd()
        dwi_path = c.dwi_base_path

        orig_path = c.orig_path
        imgext = c.imgext
        prefix = c.prefix
        affix = c.affix
        T1_processed_path = c.T1_processed_path
        processed_path = c.processed_path
        rois= c.rois
        ind_roi_path = c.ind_roi_path        
        tractography_path = c.tractography_path_full

        if c.roi_labels:
            labels = c.roi_labels

        if not os.path.isdir(tractography_path):
            os.mkdir(tractography_path)

        res = []        

        for subj in c.subjects:
            print '----------------------------seedIndividualTracts-------------------------------------'
            print subj
            subjdir = tractography_path+'/'+subj+'/'
            ras_corrected_file = subj+'_dwi_ras.nhdr'

            subjRes = { 'name' : subj }

            if not os.path.isdir(subjdir):
                os.mkdir(subjdir)


                subj_source_path = dwi_path+subj
                os.chdir(subj_source_path)

                dwi_folder = glob("*DTI*60*")[0]
                os.chdir(dwi_folder+'/nifti')

                subj_dwi_path = os.getcwd()
                os.chdir(subjdir)

                cmd = 'ln -s %s dwi' % (subj_dwi_path)
                exec_cmd(cmd)


                print 'copy DWI file to '+subjdir

                dwi_file = '/DWI_CORRECTED'
                slicer_comp_file = subjdir+dwi_file+'.nhdr'

                cmd = 'slicerDwiFileConvert.sh %s %s' % (subj_dwi_path+'/'+dwi_file+'.nhdr', slicer_comp_file)

                exec_cmd(cmd)

                reader = nrrd.NrrdReader()
                header = reader.getFileAsHeader(slicer_comp_file)
                header.correctSpaceRas()
                writer = nrrd.NrrdWriter()
                writer.write(header, subjdir+'/'+ras_corrected_file)

                #dwiCorrectRas(dwi_file+'.nhdr', subj+'_test.nhdr')


                cmd="slicerTensorScalar.sh -i %s -p %s_ -d %s" % (ras_corrected_file, subj,subjdir)

                exec_cmd(cmd)

            

            os.chdir(subjdir)
            print 'copy ROI file to '+subjdir   

            # copy t1s over
            t1i = self.getGtsFilename(True, True, subj, 'invDeformed', 'DWS')
            t1o = self.getGtsFilename(False, True, subj, 'T1', 'dwi')
            shutil.copyfile(processed_path+'/'+t1i, subjdir+'/'+t1o)

            t1i = self.getGtsFilename(True, True, subj, 'invDeformed', 'T1')
            t1o = self.getGtsFilename(False, True, subj, 'T1')
            shutil.copyfile(processed_path+'/'+t1i, subjdir+'/'+t1o)

            t1i = self.getGtsFilename(False, True, subj, 'mdwi')
            t1o = t1i
            shutil.copyfile(orig_path+'/'+t1i, subjdir+'/'+t1o)            

            for roi in rois:
                roi_file = self.getDwiRoiName(subj,roi, ref='dwi')
                roi_nifti = roi_file+imgext
                shutil.copyfile(processed_path+'/'+roi_nifti, subjdir+'/'+roi_nifti)

                cmd="slicerFileConvert.sh -i %s -o %s" % (roi_nifti, roi_file+'.nhdr');
                exec_cmd(cmd)                

                roi_file = self.getDwiRoiName(subj,roi, ref='t1')
                roi_nifti = roi_file+imgext
                shutil.copyfile(processed_path+'/'+roi_nifti, subjdir+'/'+roi_nifti)


            # if recompute is true, then only run it once given multiple labels per subject
            roibase=self.getDwiRoiName(subj,rois[0], ref='dwi')          
            if len(rois) > 1:
                excludebase = self.getDwiRoiName(subj,rois[1],ref='dwi')
            else:
                excludebase = roibase

            mrtrix_has_recompute = not recompute

            stream_map = {}
            fiber_name = ''
            method_name = ''                   

            for i,ilabel in enumerate(labels):
                label_str = c.roi_labels_str[i]
        

                for itract, iparam in c.tract_method.iteritems():
                    if itract == 'mrtrix':
                        print '\n-- PERFORM MRTRIX'
                        method_name = 'mrtrix'

                        if not mrtrix_has_recompute:
                            recompute = True
                            mrtrix_has_recompute = True
                        else:
                            recompute = False
                        fiber_name = gtracts.tractsMrtrix(subj, roibase, excludebase, ilabel, config=c, label_str=label_str, label_include=c.roi_includes[i], 
                                     label_exclude=c.roi_excludes[i], recompute=recompute, overwrite=overwrite, params=iparam)

                    elif itract == 'xst':
                        print '\n-- PERFORM XST'
                        method_name = 'xst'
                        fiber_name = gtracts.tractsXst(subj, ras_corrected_file, roibase, excludebase, ilabel, config=c, label_str=label_str, 
                                  label_include=c.roi_includes[i],label_exclude=c.roi_excludes[i], overwrite=overwrite, params=iparam)

                    if not stream_map.has_key(method_name):
                            stream_map[method_name] = [fiber_name]
                    else:
                        stream_map[method_name].append(fiber_name)

            os.chdir(origin)
        return stream_map

    def tracts_to_density(self, tracts_map):
        c = self.config
        for subj in c.subjects:
            os.chdir(c.tractography_path+'/'+subj)
            print '========== %s =========' % subj

            ref_file = 'ANTS_%s_invDeformed_T1.nii.gz' % subj
            ref_file = c.processed_path+'/' + ref_file

            for method in tracts_map.keys():
                print '== Method %s ' % method
                if method=='xst':
                    os.chdir('xst')               
                tlabels = tracts_map[method]
                for tfile in tlabels:
                    print tfile
                    if os.path.isfile(tfile):
                        gtracts.tracts_to_density(ref_file, tfile)
                    else:
                        print 'file %s not found' % tfile
                if method=='xst':
                    os.chdir('..')   
            os.chdir(c.root)



    def viewTracks(self):
        import vtk
        c = self.config
        origin=os.getcwd()

        orig_path = c.orig_path
        imgext = c.imgext
        prefix = c.prefix
        affix = c.affix
        T1_processed_path = c.T1_processed_path
        processed_path = c.processed_path
        rois= c.rois
        ind_roi_path = c.ind_roi_path
        tractography_path = c.tractography_path

        for i,label_name in enumerate(c.roi_labels_str):
            for tmethod in c.tract_method:
                
                if tmethod == 'mrtrix':
                    files = ['cst_%s' % label_name,'cst_%s_filtered' % label_name]
                elif tmethod == 'xst':
                    files = ['xst/%d' % c.roi_labels[i],'xst/%s_filtered' % label_name]

                # HTML File output
                jsfile = 'tracts_%s_%s.js' % (tmethod, label_name)

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
                    os.chdir(subj_tractography_path)
                    print '-------------------------viewTracks---------------------------'
                    print os.getcwd()

                    # convert vtk to trackviz fibers for loading intlo dipy

                    htmlfile += '<div>'

                    for f in files:
                        print f
                        subjpath = '%s/%s' % (tractography_path, subj)
                        vtkfiber = '%s/%s.vtk' % (subjpath, f)
                        volume_base = '%s/%s' % (subjpath, subj)
                        roi_name = self.getDwiRoiName(subj, rois[0], ref='dwi')
                        labelmap_dwi = '%s/%s.nii.gz' % (subjpath, roi_name)

                        roi_name = self.getDwiRoiName(subj, rois[0], ref='t1')
                        labelmap_t1 = '%s/%s.nii.gz' % (subjpath, roi_name)                        


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
                    os.chdir(origin)
                htmlfile+='</body></html>\n'

                FILE = open('tracts_%s_%s.html' % (tmethod, label_name), 'w')
                FILE.write(htmlfile)
                FILE.close()

                FILE = open(jsfile, 'w')
                FILE.write(js)
                FILE.close()

    def tracts_conjunction(self, streamnames, img_type="density"):
        """ Perform conjunction analysis of tracts by reverse project tracts to average T1 template, using density  maps 

            Args: 
                streamnames: 
        """

        c = self.config
        tract_path = c.tractography_path

        image_table = []
        not_found_table = []

        if img_type=='density':
            affix = 'den'
        elif img_type=='fiber':
            affix = 'fib'
        elif img_type=='binary':
            affix = 'bin'
        
        print '===================== PERFORM CONJUNCTION ======================='
        root = os.getcwd()        

        for subj in c.subjects:
            os.chdir(tract_path+'/'+subj)
            subj_files = []

            print '--------------------------- %s ----------------------------------' % subj
            for method, file_list in streamnames.iteritems():
                print '-- Method %s ' % method
                if method=='xst':
                    os.chdir('xst')
                for tfile in file_list:
                    fext = os.path.splitext(tfile)

                    den_file = fext[0]+'_%s.nii.gz' % affix
                    print '>',den_file 

                    if os.path.isfile(den_file):
                        trans = ''

                        # t1 -> average, file in processed
                        output1 = self.getGtsFilename(False,True, fext[0], affix, 'T1')
                        output2 = self.getGtsFilename(False, True, subj, method, fext[0], affix)
                        #print '#',output2

                        output2 = c.processed_path+'/'+output2
                        trans += self.ants_apply_transform_average_to_t1(subj, output1, output2, reference='average', inverse=True, just_transform_param=True)

                        # dwi -> t1 projection first, file stay in the tractography dirs
                        trans += ' '+self.ants_apply_transform_t1_to_dwi(subj, den_file, output1, reference='t1', inverse=True, just_transform_param=True) 
                        


                        ref = 'con_average.nii.gz' # % subj
                        ref = c.orig_path+'/'+ref          
                        interp = 'NearestNeighbor'

                        cmd=('antsApplyTransforms -d 3 -i %s -o %s -r %s -n %s ' % (den_file, output2, ref, interp))
                        cmd += trans
                        exec_cmd(cmd)                                                           

                        subj_files.append(output2)
                    else:
                        not_found_table.append(subj+'/'+den_file)
                if method=='xst':
                    os.chdir('..')

            image_table.append(subj_files)            
            os.chdir(root)

        # for i in image_table:
        #     print '#',image_table

        # for i in not_found_table:
        #     print '!',i
        # return

        #image_table = np.array(image_table, dtype='object')
        image_table = map(list, zip(*image_table))

        import nibabel as nib
        import gc

        refimg = nib.load(c.orig_path+'/con_average.nii.gz')

        print '=== Perform conjuction ==='

        for r in image_table:
            if len(r) > 0:
                label = os.path.basename(r[0])
                label = '_'.join(label.split('_')[1:-1])            
                print '###',label
                data_pool = np.zeros(refimg.get_shape()[:3])

                for i,j in enumerate(r):
                    print j
                    img = nib.load(j)
                    idata = img.get_data()
                    data_pool = np.add(data_pool,idata)     
                    del img
                    del idata           
                data_pool = np.divide(data_pool,len(r))
                nifti = nib.Nifti1Image(data_pool, refimg.get_affine())
                nib.save(nifti, c.processed_path+'/'+label+'_%s_average.nii.gz' % affix)
                del data_pool
                gc.collect()

        print '=== The following are not found:'
        for i in not_found_table:
            print i

















