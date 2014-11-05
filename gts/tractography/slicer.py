import os
from ... import exec_cmd
from ... import TractographyMethod
from vtk import vtkPolyDataReader
from vtk import vtkXMLPolyDataWriter
from vtk import vtkXMLPolyDataReader
from vtk import vtkPolyDataWriter

class Slicer3(TractographyMethod):
    def run(self):
        tract()

    def tract(self, subj, roi_base, exclude_base, label, recompute=False, overwrite=False, label_exclude=10,label_str="", label_include=11, config=None, params='', method_label='slicer'):
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

        vreader = vtkXMLPolyDataReader()
        vreader.SetFileName(fitered_temp)
        vreader.Update()
        polydata = vreader.GetOutput()

        vwriter = vtkPolyDataWriter()
        output_file = '%s_filtered.vtk' % output_base
        vwriter.SetFileName(output_file)
        vwriter.SetInput(polydata)
        vwriter.Write()

        return output_file     