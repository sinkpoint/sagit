
import vtk
import numpy as np
from dipy.viz import fvtk
from time import sleep
from dipy.io.pickles import load_pickle
from dipy.data import two_cingulum_bundles


def getStreamlines(filename):
  reader = vtk.vtkPolyDataReader()
  reader.SetFileName(filename)
  reader.Update()

  polydata = reader.GetOutput()

  streamlines = []
  for i in range(polydata.GetNumberOfCells()):
      pts = polydata.GetCell(i).GetPoints()
      npts = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())])
      streamlines.append(npts)
  return streamlines

cb_subj1 = getStreamlines("c10_test.vtk")
cb_subj2 = getStreamlines("c50_test.vtk")

from dipy.align.streamlinear import (StreamlineLinearRegistration,
                                     vectorize_streamlines)

"""
An important step before running the registration is to resample the streamlines
so that they both have the same number of points per streamline. Here we will
use 20 points.
"""

cb_subj1 = vectorize_streamlines(cb_subj1, 20)
cb_subj2 = vectorize_streamlines(cb_subj2, 20)

"""
Let's say now that we want to move the ``cb_subj2`` (moving) so that it can be
aligned with ``cb_subj1`` (static). Here is how this is done.
"""

srr = StreamlineLinearRegistration()

srm = srr.optimize(static=cb_subj1, moving=cb_subj2)

"""
After the optimization is finished we can apply the learned transformation to
``cb_subj2``.
"""

cb_subj2_aligned = srm.transform(cb_subj2)


def show_both_bundles(bundles, colors=None, show=False, fname=None):

    ren = fvtk.ren()
    ren.SetBackground(1., 1, 1)
    for (i, bundle) in enumerate(bundles):
        color = colors[i]
        lines = fvtk.streamtube(bundle, color, linewidth=0.3)
        lines.RotateX(-90)
        lines.RotateZ(90)
        fvtk.add(ren, lines)
    if show:
        fvtk.show(ren)
    if fname is not None:
        sleep(1)
        fvtk.record(ren, n_frames=1, out_path=fname, size=(900, 900))


#([cb_subj1, cb_subj2],
#                  colors=[fvtk.colors.orange, fvtk.colors.red], show=True)

"""
.. figure:: before_registration.png
   :align: center

   **Before bundle registration**.
"""

show_both_bundles([cb_subj1, cb_subj2_aligned],
                  colors=[fvtk.colors.orange, fvtk.colors.red], show=True,
                  fname='after_registration.png')