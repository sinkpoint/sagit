
import gts
from gts import exec_cmd
import os
from os import path

class Mrtrix(gts.TractographyMethod):
    def __init__(self, subj, seed_config, method_config, global_config):
        super(Mrtrix, self).__init__(subj, seed_config, method_config, global_config)

    def compute_tensors(self):
        os.chdir('dwi')
        # calculate CSD estimation
        cmd = 'rm dwi.mif'
        exec_cmd(cmd)
        cmd = 'mrconvert Motion_Corrected_DWI_nobet.nii.gz dwi.mif'
        exec_cmd(cmd)
        cmd = 'mrtrix_grad.py -v newdirs.dat -a %s.bval -o dwi.grad' % (self.subject)
        exec_cmd(cmd)
        cmd = 'mrtrix_compcsd.sh -g dwi.grad dwi.mif'
        exec_cmd(cmd)
        os.chdir('..')     



    def run(self, filter=True):
        def convert_to_mif(input):
            basename = input.split('.')[0]
            cmd = 'mrconvert -force %s.nii.gz -datatype Bit %s.mif' % (basename,basename)
            exec_cmd(cmd)
            return basename+'.mif'


        self.goto_working_path()

        if not path.isfile('dwi/CSD8.mif'):
            self.compute_tensors()

        exclude_param = ''
        include_param = ''

        inc = self.get_includes_info()
        if inc:
            include_file = self.combine_masks(inc, name='includes')
            include_file = convert_to_mif(include_file)
            include_param = ' -include %s ' % (include_file)

        ex = self.get_excludes_info()
        if ex:
            exclude_file = self.combine_masks(ex, name='excludes')
            exclude_file = convert_to_mif(exclude_file)
            exclude_param = ' -exclude %s ' % (exclude_file)

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

        if not filter:
            output_tck = '%s.tck' % fiber_basename
            cmd = 'tckgen  %s  dwi/CSD8.mif  %s' % (streamparam, output_tck)
            exec_cmd(cmd)
            cmd = 'tracks2vtk %s.tck %s.vtk' % (fiber_basename, fiber_basename)
            exec_cmd(cmd)
        else:
            output_tck = '%s_filtered.tck' % fiber_basename
            cmd = 'tckgen  %s %s %s dwi/CSD8.mif %s ' % (streamparam, include_param, exclude_param, output_tck)
            exec_cmd(cmd)
            cmd = 'tracks2vtk %s_filtered.tck %s_filtered.vtk' % (fiber_basename, fiber_basename)
            exec_cmd(cmd)

        # cmd = 'copyTensors.py -t dti.nhdr -f %s.vtk -o %s ' % (fiber_basename, output)
        # #exec_cmd(cmd)

        # cmd = 'copyTensors.py -t dti.nhdr -f %s_filtered.vtk -o %s ' % (fiber_basename, output2)
        # #exec_cmd(cmd)

        self.reset_path()
        return output2