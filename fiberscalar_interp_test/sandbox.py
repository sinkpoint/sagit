#!/usr/bin/python

#convert vtk to vtp files to filter
import sys
import numpy as np
from vispy import app, gloo, scene
from vispy.util.transforms import perspective, translate, rotate

from vtkFileIO import vtkToStreamlines

streams = vtkToStreamlines(sys.argv[1])

# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


VERT_SHADER = """
    uniform mat4 view;
    uniform mat4 model;
    uniform mat4 projection;
    attribute vec3 position;
attribute float a_id;
varying float v_id;    
    void main()
    {
    v_id = a_id;
        gl_Position = projection * view * model * vec4(position, 1.0);
        gl_PointSize = 4;        
    }
"""

FRAG_SHADER = """
    varying float v_id;
    void main()
    {
        gl_FragColor = vec4(1,1,1,1);

    }
"""


all_verts = None

for i in streams:
    if all_verts is None:
        all_verts = i
    else:
        all_verts = np.vstack((all_verts, i))

all_verts = all_verts.astype(np.float32)

n = len(all_verts)
a_id = np.random.randint(0, 30, (n, 1))
a_id = np.sort(a_id, axis=0).astype(np.float32)

c = app.Canvas(keys='interactive')
program = gloo.Program(VERT_SHADER, FRAG_SHADER, count=n)

@c.connect
def on_resize(event):
    gloo.set_viewport(0,0, *event.size)

@c.connect
def on_draw(event):
    gloo.clear((0,0,0,1))
    program.draw('line_strip')    

# view = c.central_widget.add_view()
# view.set_camera('turntable', mode='perspective', up='z', distance=100,
#                 azimuth=30., elevation=30.)
# axis = scene.visuals.XYZAxis(parent=view.scene)

c.show()
app.run()

# # plt.show()

# import nibabel as nib

# img = nib.load('xst_fornix_bin_average.nii.gz')
# avg_mask = img.get_data()

# # t1 -> dwi
# # t1 -> group

# img = nib.load('ANTS_C101InverseWarp.nii.gz')
# dwi2t1_warp = img.get_data()

# import scipy.io as sio
# dwi2t1_affine = sio.loadmat('ANTS_C100GenericAffine.mat')


# img = nib.load('ANTS_C10_template_warp.nii.gz')
# t12group_warp = img.get_data()

# t12group_affine = sio.loadmat('ANTS_C10_template_affine.mat')

