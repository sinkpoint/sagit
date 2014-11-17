import gts
from gts import exec_cmd
import os
from os import path
from vtk import vtkXMLPolyDataReader
from vtk import vtkPolyDataWriter
        
class Slicer3(gts.TractographyMethod):
    def __init__(self, subj, seed_config, method_config, global_config):
        super(Slicer3, self).__init__(subj, seed_config, method_config, global_config)

    def convert_nifti_to_nrrd(input):
        basename = os.path.basename(input).split('.')[0]
        ipath = os.path.split(input)[0]
        output = os.path.join(ipath,basename+'.nrrd')
        cmd = 'slicerFileConvert.sh -i %s -o %s' % (input, output)    
        exec_cmd(cmd, display=False)        

        return output  

    convert_nifti_to_nrrd = staticmethod(convert_nifti_to_nrrd)

    def prep_seed_files(self):
        include_file = ''
        exclude_file = ''

        inc = self.get_includes_info()
        if inc:
            include_file = self.combine_masks(inc, name='includes')
            include_file = self.convert_nifti_to_nrrd(include_file)

        ex = self.get_excludes_info()
        if ex:
            exclude_file = self.combine_masks(ex, name='excludes')
            exclude_file = self.convert_nifti_to_nrrd(exclude_file)

        seed_info = self.get_seed_info()
        seed_file = seed_info['filename']
        seed_file = self.convert_nifti_to_nrrd(seed_file)

        return include_file, exclude_file, seed_file        

    def run(self):
        self.goto_working_path()

        include_file, exclude_file, seed_file = self.prep_seed_files()

        seed_info = self.get_seed_info()

        fiber_basename = self.get_unique_name()
        unfiltered_file = '%s.vtp' % fiber_basename    
  

        params = '--stoppingvalue 0.2 --minimumlength 5 --clthreshold 0.2 --randomgrid --seedspacing 0.5 ' 
        if 'params' in self.method_config:
            streamparam = self.method_config['params']

        cmd = 'slicerTractography.sh --label %s %s dti.nhdr  %s  %s' % (seed_info['label'], params, seed_file, unfiltered_file)
        exec_cmd(cmd, truncate=True)

        vreader = vtkXMLPolyDataReader()
        vreader.SetFileName(unfiltered_file)
        vreader.Update()
        polydata = vreader.GetOutput()

        vwriter = vtkPolyDataWriter()
        output_file = '%s.vtk' % fiber_basename
        vwriter.SetFileName(output_file)
        vwriter.SetInput(polydata)
        vwriter.Write()

        output = self.filter_step(unfiltered_file, include_file, exclude_file)

        self.reset_path()
        return output

    def filter_step(self, unfiltered_file, include_file, exclude_file):
        import shutil

        fiber_basename = self.get_unique_name()
        filtered_temp_file = '%s_filtered.vtp' % fiber_basename
        shutil.copy2(unfiltered_file, filtered_temp_file)        

        if include_file: 
            cmd='slicerFilterFibers.sh --pass 1 %s %s %s' % (include_file, filtered_temp_file, filtered_temp_file)
            exec_cmd(cmd)

        if exclude_file:
            cmd='slicerFilterFibers.sh --nopass 1 %s %s %s' % (exclude_file, filtered_temp_file, filtered_temp_file)
            exec_cmd(cmd)

        vreader = vtkXMLPolyDataReader()
        vreader.SetFileName(filtered_temp_file)
        vreader.Update()
        polydata = vreader.GetOutput()

        vwriter = vtkPolyDataWriter()
        output_file = '%s_filtered.vtk' % fiber_basename
        vwriter.SetFileName(output_file)
        vwriter.SetInput(polydata)
        vwriter.Write()

        return output_file 
 