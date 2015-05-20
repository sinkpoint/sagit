#!/usr/bin/python

#convert vtk to vtp files to filter
import sys
import numpy as np
import numpy.linalg as la
from vtkFileIO import vtkToStreamlines
from dipy.tracking.metrics import *

streams = vtkToStreamlines(sys.argv[1])
#data = np.concatenate(streams)
curves = []

# get total average direction (where majority point towards)
avg_d = np.zeros(3)
# for line in streams:
#     d = np.array(line[-1]) - np.array(line[0])
#     d = d / la.norm(d)
#     avg_d += d
#     avg_d /= la.norm(avg_d)   


avg_com = np.zeros(3)
avg_mid = np.zeros(3)

# unify streamline list orders to point in the same general direction
for i, line in enumerate(streams):    
    d = np.array(line[-1]) - np.array(line[0])
    #print line[-1],line[0],d
    d = d / la.norm(d)
    if d.dot(avg_d) < 0:
        streams[i] = line[::-1]
        line = streams[i]
        d*=-1
        #print 'reverse'
    avg_d += d
    #avg_d /= la.norm(avg_d)   

    #print avg_d
    est = downsample(line, n_pols=10)
    curve = spline(est, s=1, k=3)
    curves.append(downsample(curve,50))
    curve_com = center_of_mass(curve)
    avg_com = (avg_com + curve_com)/2

    curve_mid = midpoint(curve)
    avg_mid = (avg_mid + curve_mid)/2    


avg_d /= (la.norm(avg_d)*0.1)
print 'average direction',avg_d 
print 'average center-of-mass',avg_com

from mayavi import mlab

def Arrow_From_A_to_B(x1, y1, z1, x2, y2, z2):
    ar1=visual.arrow(x=x1, y=y1, z=z1)
    ar1.length_cone=0.4

    arrow_length=np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    ar1.actor.scale=[arrow_length, arrow_length, arrow_length]
    ar1.pos = ar1.pos/arrow_length
    ar1.axis = [x2-x1, y2-y1, z2-z1]
    return ar1

fig = mlab.figure()
scene = mlab.gcf().scene
scene.renderer.render_window.set(alpha_bit_planes=1,multi_samples=0)
scene.renderer.set(use_depth_peeling=True,maximum_number_of_peels=4,occlusion_ratio=0.1)

from tvtk.tools import visual
visual.set_viewer(fig)
Arrow_From_A_to_B(0,0,0,avg_d[0],avg_d[1],avg_d[2])

for i, line in enumerate(curves):
    c = np.random.rand(3,1)
    mlab.plot3d(line[:,0], line[:,1], line[:,2], range(0,len(line)))
    orig = streams[i]
    mlab.plot3d(orig[:,0], orig[:,1], orig[:,2], color=(1,1,1), opacity=0.5)

mlab.points3d(avg_com[0],avg_com[1],avg_com[2],color=(1,0,0), mode='axes',scale_factor=5)
mlab.points3d(avg_mid[0],avg_mid[1],avg_mid[2],color=(0,0,1), mode='axes',scale_factor=5)
mlab.show() 

sys.exit()

# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.patches import FancyArrowPatch
# from mpl_toolkits.mplot3d import proj3d

# class Arrow3D(FancyArrowPatch):
#     def __init__(self, xs, ys, zs, *args, **kwargs):
#         FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
#         self._verts3d = xs, ys, zs

#     def draw(self, renderer):
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
#         self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
#         FancyArrowPatch.draw(self, renderer)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
# # colors = np.hstack([colors] * 20)

# # plt.scatter(norm_p[:, 0], norm_p[:, 1], zs=norm_p[:,2], color=colors[y_pred].tolist(), s=1)

# # if hasattr(algo, 'cluster_centers_'):
# #     centers = algo.cluster_centers_
# #     center_colors = colors[:len(centers)]
# #     plt.scatter(centers[:, 0], centers[:, 1], zs=centers[:,2], s=100, c=center_colors)
# for line in curves:
#     c = np.random.rand(3,1)
#     ax.plot(line[:,0], line[:,1], line[:,2], color=c)
#     ax.scatter(line[:,0], line[:,1], line[:,2], c=range(0,len(line)),s=10, lw = 0, depthshade=False)
#     #ax.scatter(line[0,0], line[0,1], line[0,2], s=20, c=u'r')
#     d = np.array(line[len(line)-1]) - np.array(line[0])
#     #ax.plot([0,d[0]], [0,d[1]], [0,d[2]],lw=2,color='b')
    

# plt.show()


