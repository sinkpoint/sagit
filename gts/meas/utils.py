import nibabel as nib
import vtk

from meas import sample_scalars
from vtkFileIO import vtkToStreamlines

def image_to_vtk(nifti_img, vtk_file, output_file, scalar_name='Scalar'):
    img = nib.load(nifti_img)

    streams, vtkdata = vtkToStreamlines(vtk_file)
    scalars = sample_scalars(streams, img)


    vtkScalars = vtk.vtkFloatArray()
    vtkScalars.SetName(scalar_name)
    vtkScalars.SetNumberOfComponents(1)
    #vtkScalars.SetNumberOfTuples(len(scalars))
    for i,v in enumerate(scalars):
        vtkScalars.InsertNextTuple1(v)
    vtkdata.GetPointData().SetScalars(vtkScalars)

    writer = vtk.vtkXMLPolyDataWriter()
    #writer = vtk.vtkPolyDataWriter()
    writer.SetInput(vtkdata)
    writer.SetFileName(output_file)
    writer.Write()

def vtk_to_image():
    pass