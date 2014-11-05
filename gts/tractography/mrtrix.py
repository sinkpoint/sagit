
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
        cmd = 'rm dwi.mif; mrconvert Motion_Corrected_DWI_nobet.nii.gz dwi.mif'
        exec_cmd(cmd)
        cmd = 'mrtrix_grad.py -v newdirs.dat -a %s.bval -o dwi.grad' % (self.subject)
        exec_cmd(cmd)
        cmd = 'mrtrix_compcsd.sh -g dwi.grad dwi.mif'
        exec_cmd(cmd)
        os.chdir('..')     



    def run(self):
        def convertToMif(input):
            basename = input.split('.')[0]
            cmd = 'mrconvert -force %s.nii.gz -datatype Bit %s.mif' % (basename,basename)
            exec_cmd(cmd)
            return basename+'.mif'


        self.gotoWorkingPath()

        if not path.isfile('dwi/CSD8.mif'):
            self.compute_tensors()

        exclude_param = ''
        include_param = ''

        inc = self.get_includes_info()
        if inc:
            include_file = self.combine_masks(inc, name='includes')
            include_file = convertToMif(include_file)
            include_param = ' -include %s ' % (include_file)

        ex = self.get_excludes_info()
        if ex:
            exclude_file = self.combine_masks(ex, name='excludes')
            exclude_file = convertToMif(exclude_file)
            exclude_param = ' -exclude %s ' % (exclude_file)

        seed_info = self.get_seed_info()
        seed_file = seed_info['filename']
        seed_file = self.extract_label_from_image(seed_file, seed_info['label'], name='seed', save=True)
        seed_file = convertToMif(seed_file)



        #default parameters        
        streamparam = "-algorithm iFOD2 -step 0.3 -angle 60 -minlength 10 -cutoff 0.15 -initcutoff 0.15 -force"
        if 'params' in self.method_config:
            streamparam = self.method_config['params']


        fiber_basename = self.get_unique_name()
        output = '%s.vtk' % fiber_basename    
        output2 = '%s_filtered.vtk' % fiber_basename            

        streamparam += " -seed_image %s " % (seed_file)

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

        self.resetPath()
        return output2


    def tract2(self, subj, roi_base, exclude_base, label, 
        label_exclude=10,label_str="", label_include=11, params='', method_label='cst'):

        c = self.global_config
        roiBase = roi_base
        filterBase = exclude_base

        #exec_cmd('rm dwi/CSD8.mif')
        if not path.isfile('dwi/CSD8.mif') or self.recompute:
            self.compute_tensors()

        base = label_str 
        if label_str == "":
            base="%s_%d"%(roi_base, label)

        fiber_basename = '%s_%s' % (method_label, base)
        fiber_output = '%s.tck' % (fiber_basename)

        output = '%s.vtk' % fiber_basename    
        output2 = '%s_filtered.vtk' % fiber_basename    

        if not path.isfile(fiber_output) or self.overwrite:

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