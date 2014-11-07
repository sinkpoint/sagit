import gts
from gts import exec_cmd
import os
from os import path

class Xst(gts.TractographyMethod):
    def __init__(self, subj, seed_config, method_config, global_config):
        super(Xst, self).__init__(subj, seed_config, method_config, global_config)

    def run(self):
        include_file = ''
        exclude_file = ''

        inc = self.get_includes_info()
        if inc:
            include_file = self.combine_masks(inc, name='includes')

        ex = self.get_excludes_info()
        if ex:
            exclude_file = self.combine_masks(ex, name='excludes')

        seed_info = self.get_seed_info()
        seed_file = seed_info['filename']

        fiber_basename = self.get_unique_name()
        unfiltered_file = '%s.vtk' % fiber_basename    
        filtered_temp_file = '%s_filtered.vtp' % fiber_basename      

        params = "-stop aniso:cl2,0.15 frac:0.1 radius:0.8 minlen:10 -step 0.1"
        if 'params' in self.method_config:
            streamparam = self.method_config['params']

        cmd='seedTend2Points.py -i %s -n 10 -r -m %s' % (seed_file, fiber_basename)
        exec_cmd(cmd)

        cmd="tend2 fiber -i %s -dwi -wspo -ns seeds/%s%d.txt -o %s -ap -v 2 -t 2evec0 -k cubic:0.0,0.5 -n rk4 %s" % (dwi_file, roiBase,label,unfiltered_file, params)

        exec_cmd(cmd)

        from vtk import vtkPolyDataReader
        from vtk import vtkPolyDataWriter
        from vtk import vtkXMLPolyDataWriter
        from vtk import vtkXMLPolyDataReader

        #convert vtk to vtp files to filter
        vreader = vtkPolyDataReader()
        vreader.SetFileName(unfiltered_file)
        vreader.Update()
        polydata = vreader.GetOutput()

        vwriter = vtkXMLPolyDataWriter()
        vwriter.SetFileName(fiber_basename+'.vtp')
        vwriter.SetInput(polydata)
        vwriter.Update()
        vwriter.Write()

        import shutil
        filtered_temp = 'xst/%s_filtered.vtp' % fiber_basename
        shutil.copy2(fiber_basename+'.vtp', filtered_temp_file)    

        if include_file:
            cmd='slicerFilterFibers.sh --pass 1 %s %s %s' % (include_file, filtered_temp_file, filtered_temp_file)
            exec_cmd(cmd)

        if exclude_file:
            cmd='slicerFilterFibers.sh --nopass 1 %s %s %s' % (exclude_file, filtered_temp_file, filtered_temp_file)
            exec_cmd(cmd)

        # convert vtp back to vtk to load in dipy
        # TODO: load vtp directly in VTK renderer instead;
        # or consider using webgl

        vreader = vtkXMLPolyDataReader()
        vreader.SetFileName(filtered_temp_file)
        vreader.Update()
        polydata = vreader.GetOutput()

        vwriter = vtkPolyDataWriter()
        output_file = 'xst/%s_filtered.vtk' % fiber_basename
        vwriter.SetFileName(output_file)
        vwriter.SetInput(polydata)
        vwriter.Write()

        return output_file



    def tract(self, subj, roi_base, exclude_base, label=1, label_exclude=10, overwrite=False, label_str="", label_include=11, dwi_file='', config=None, gen_only=False, params='', method_label='xst'):
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
        vreader = vtkPolyDataReader()
        vreader.SetFileName(out_file)
        vreader.Update()
        polydata = vreader.GetOutput()

        vwriter = vtkXMLPolyDataWriter()
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

        vreader = vtkXMLPolyDataReader()
        vreader.SetFileName(filtered_temp)
        vreader.Update()
        polydata = vreader.GetOutput()

        vwriter = vtkPolyDataWriter()
        output_file = 'xst/%s_filtered.vtk' % label_name_base
        vwriter.SetFileName(output_file)
        vwriter.SetInput(polydata)
        vwriter.Write()

        return output_file