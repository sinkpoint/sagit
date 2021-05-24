# -*- coding: utf-8 -*-

import os
from gts import exec_cmd
from os import path
from tract_method import TractographyMethod
import time
class Mrtrix(TractographyMethod):
    def __init__(self, subj, seed_config, method_config, global_config):
        super(Mrtrix, self).__init__(subj, seed_config, method_config, global_config)

    def compute_tensors(self):
        root = os.getcwd()
        os.chdir('dwi')
        # no longer need to convert dwi to mif format
        # calculate CSD estimation
        # cmd = 'rm dwi.mif'
        # exec_cmd(cmd)
        # cmd = 'mrconvert DWI_CORRECTED.nii.gz dwi.mif'
        # exec_cmd(cmd)
        bval_file = 'DWI.bval'
        if os.path.isfile(self.subject+'.bval'):
            bval_file = self.subject+'.bval'

        print '''
            =========================================================== 
            WARNING: flip Y,Z directions of the tensor 
            CHECK THAT IT IS APPROPRIATE FOR YOUR SCANS 
            ===========================================================
        '''

        cmd = 'mrtrix_grad.py -v DWI_CORRECTED.bvec -a %s -o dwi.grad' % (bval_file)
        exec_cmd(cmd)

        exec_cmd('bet2 DWI_CORRECTED.nii.gz dwibet -m -n -f 0.1')
        exec_cmd('mrconvert dwibet_mask.nii.gz mask.mif -datatype Bit -force')
        
        if not os.path.isfile('response.txt'):
            cmd = 'dwi2response -mask mask.mif -grad dwi.grad DWI_CORRECTED.nii.gz response.txt'
            exec_cmd(cmd)
        
        cmd = 'dwi2fod -mask mask.mif -grad dwi.grad DWI_CORRECTED.nii.gz response.txt CSD8.mif -force'
        exec_cmd(cmd)

        # cmd = 'mrtrix_compcsd.sh -g dwi.grad DWI_CORRECTED.nii.gz'
        # exec_cmd(cmd)
        time.sleep(1)
        os.chdir(root)



    def run(self, filter=True, recompute=False):
        def convert_to_mif(input):
            basename = input.split('.')[0]
            cmd = 'mrconvert -force %s.nii.gz -datatype Bit %s.mif' % (basename,basename)
            exec_cmd(cmd)
            return basename+'.mif'


        self.goto_working_path()

        print '''
        ╔╦╗╦═╗┌┬┐┬─┐┬─┐ ┬
        ║║║╠╦╝ │ ├┬┘│┌┴┬┘
        ╩ ╩╩╚═ ┴ ┴└─┴┴ └─
        '''

        if not path.isfile('dwi/CSD8.mif'):
            self.compute_tensors()

#pchu:added: go back to the previous path
        os.chdir(current_path)
#pchu:end
            
            
        exclude_param = ''
        include_param = ''
        mask_param = ''

        inc = self.get_includes_info()
        if inc:
            include_file = self.combine_masks(inc, name='includes')
            # include_file = convert_to_mif(include_file)
            include_param = ' -include %s ' % (include_file)

        ex = self.get_excludes_info()
        if ex:
            exclude_file = self.combine_masks(ex, name='excludes')
            # exclude_file = convert_to_mif(exclude_file)
            exclude_param = ' -exclude %s ' % (exclude_file)

        maskinfo = self.get_named_info('mask')
        if maskinfo:
            mask_file = self.combine_masks(maskinfo, name='mask')
            mask_param = ' -mask {}'.format(mask_file)

        seed_info = self.get_seed_info()
        seed_file = seed_info['filename']
        seed_file = self.extract_label_from_image(seed_file, seed_info['label'], name='seed', save=True)
        seed_file = convert_to_mif(seed_file)



        #default parameters        
        streamparam = "-algorithm iFOD2 -step 0.3 -angle 60 -minlength 10 -cutoff 0.15 -initcutoff 0.15 -force"
        if 'params' in self.method_config:
            streamparam = self.method_config['params']


        fiber_basename = self.get_unique_name()
        output = '%s.vtk' % fiber_basename    
        output2 = '%s_filtered.vtk' % fiber_basename            

        streamparam += " -seed_image %s " % (seed_file)

#pchu: added change to track command based on mrtrix version
        from subprocess import Popen, PIPE
        cmd = 'mrconvert -version | head -n 1'
        line_mrtrix_version = Popen(cmd,stdout=PIPE, shell=True)
        (out,err) = line_mrtrix_version.communicate()
        str1 = "mrconvert ";
        str2 = "_";
        mrtrix_version = out[out.index(str1)+len(str1):out.index(str2)]

        print("mrtrix version: "+mrtrix_version+", 1st number: "+mrtrix_version[0])
        if mrtrix_version[0] == "0":
            #use tracks2vtk for mrtrix0.2
            trackConvertCommand="tracks2vtk"
        else:
            #use tckconvert for mrtrix3
            trackConvertCommand="tckconvert"
#pchu: added change to track command based on mrtrix version

        
        if not filter:
            output_tck = '%s.tck' % fiber_basename
            cmd = 'tckgen  %s  dwi/CSD8.mif  %s' % (streamparam, output_tck)
            exec_cmd(cmd)
            #cmd = 'tracks2vtk %s.tck %s.vtk' % (fiber_basename, fiber_basename)
            cmd = '%s %s.tck %s.vtk' % (trackConvertCommand, fiber_basename, fiber_basename)
            exec_cmd(cmd)
        else:
            output_tck = '%s_filtered.tck' % fiber_basename
            cmd = 'tckgen  {streamparam} {include_param} {exclude_param} {mask_param} dwi/CSD8.mif {output_tck}'.format(**locals())
            exec_cmd(cmd)
            #cmd = 'tracks2vtk %s_filtered.tck %s_filtered.vtk' % (fiber_basename, fiber_basename)
            cmd = '%s %s_filtered.tck %s_filtered.vtk' % (trackConvertCommand, fiber_basename, fiber_basename)
            exec_cmd(cmd)

        # cmd = 'copyTensors.py -t dti.nhdr -f %s.vtk -o %s ' % (fiber_basename, output)
        # #exec_cmd(cmd)

        # cmd = 'copyTensors.py -t dti.nhdr -f %s_filtered.vtk -o %s ' % (fiber_basename, output2)
        # #exec_cmd(cmd)

        self.reset_path()
        return output2
