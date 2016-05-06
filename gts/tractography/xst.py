import slicer
from gts import exec_cmd
import os
from os import path
from pynrrd import NrrdReader, NrrdWriter

class Xst(slicer.Slicer3):
    def __init__(self, subj, seed_config, method_config, global_config):
        super(Xst, self).__init__(subj, seed_config, method_config, global_config)

    def run(self, filter=True):
        self.goto_working_path()
        includes_file, excludes_file, seed_file = self.prep_seed_files()
        seed_info = self.get_seed_info()
        print seed_info
        seed_basename = seed_info['name']
        fiber_basename = self.get_unique_name()
        unfiltered_file = '%s.vtk' % fiber_basename    

        params = "-stop aniso:cl2,0.15 frac:0.1 radius:0.8 minlen:10 -step 0.1"
        if 'params' in self.method_config:
            params = self.method_config['params']
        
        seed_params = '-n 10 -r '
        if 'seed' in self.method_config:
            seed_params = self.method_config['seed']
            
        cmd='seedTend2Points.py -i %s %s -m %s' % (seed_file, seed_params, seed_basename)
        exec_cmd(cmd, display=False)

        dwi_file = self.subject+'_dwi_ras.nhdr'

        if not path.isfile(dwi_file):
            print '-- PREPROCESS DWI FOR XST'
            # This is required as XST cannot correct read space directions other than RAS                             
            ras_corrected_file = dwi_file
            reader = NrrdReader()
            header, b = reader.load('DWI_CORRECTED.nhdr')
            # once slicer reads the file, it will convert the file it's a space that it understands                
            header.correctSpaceRas() # convert spacing to RAS, this is needed for xst, else geometry will be inverted.
            writer = NrrdWriter()
            writer.write(header, ras_corrected_file)

            print header.b0num
            if header.b0num > 1:
                cmd = 'b0avg.py -i %s -o %s' % (ras_corrected_file,ras_corrected_file)
                exec_cmd(cmd)

        label = seed_info['label']
        seed_file = seed_basename+label
        cmd="tend2 fiber -i %s -dwi -wspo -ns seeds/%s.txt -o %s -ap -v 2 -t 2evec0 -k cubic:0.0,0.5 -n rk4 %s" % (dwi_file, seed_file, unfiltered_file, params)
        exec_cmd(cmd, truncate=False, watch='stderr')

        #convert vtk to vtp files to filter
        from vtk import vtkPolyDataReader
        from vtk import vtkXMLPolyDataWriter

        vreader = vtkPolyDataReader()
        vreader.SetFileName(unfiltered_file)
        vreader.Update()
        polydata = vreader.GetOutput()

        unfiltered_file = fiber_basename+'.vtp'
        vwriter = vtkXMLPolyDataWriter()
        vwriter.SetFileName(unfiltered_file)
        vwriter.SetInput(polydata)
        vwriter.Update()
        vwriter.Write()
        output = '%s_filtered.vtp' % fiber_basename
        
        if filter:
            output = self.filter_step(unfiltered_file, includes_file, excludes_file)
        self.reset_path()
        return output