#!/usr/bin/env python
import vtk
import sys
from vtk.util.numpy_support import vtk_to_numpy

file = sys.argv[1]
print 'reading', file
outfile = sys.argv[2]

vreader = vtk.vtkXMLPolyDataReader()
vreader.SetFileName(file)
vreader.Update()
polydata = vreader.GetOutput()

scalars = {}
pointdata = polydata.GetPointData()
for si in range(pointdata.GetNumberOfArrays()):
    sname =  pointdata.GetArrayName(si)
    scalars[sname] = vtk_to_numpy(pointdata.GetArray(si))

print scalars

scalars['MD'] = []
# calc md
for i,rd in enumerate(scalars['RD']):
    md = (rd*2+scalars['AD'][i])/3
    scalars['MD'].append(md)

arr = vtk.vtkFloatArray()
arr.SetName('MD')
arr.SetNumberOfComponents(1)
for v in scalars['MD']:
    arr.InsertNextTuple1(v)
pointdata.AddArray(arr)
pointdata.SetActiveScalars('MD')

vwriter = vtk.vtkXMLPolyDataWriter()
vwriter.SetInput(polydata)
vwriter.SetFileName(outfile)
vwriter.Write()
print 'saved',outfile