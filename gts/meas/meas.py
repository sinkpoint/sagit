#!/usr/bin/env python

import numpy as np

from interps import trilinear_interp

def get_ijk_indices(point, origin, vmat, dirs):
    point = np.array(point)
    origin = np.array(origin)

    #print '#xyz= ',point,

    # adjust for origin offeset
    point = point - origin

    # apply inverse transform
    vpoint =  point * vmat

    # adjust to ijk index (0 start), and convert to int
    vpoint = vpoint - 1
    vpoint = np.ravel(np.asarray(vpoint))
    #print '  #ijk= ',vpoint

    return vpoint

def sample_scalars(streams, img):
    affine = np.matrix(img.get_affine())
    img_data = img.get_data()

    ph_data = np.zeros(img_data.shape)
    dims = ph_data.shape
    for i,v in enumerate(np.linspace(0,10,num=dims[0])):
        ph_data[i] += v

    aff_inv = affine.I

    points = np.concatenate(streams)
    print 'world'
    max = np.amax(points, axis=0)
    print max
    min = np.amin(points, axis=0)
    print min

    points = np.hstack((points,np.ones((points.shape[0],1))))
    #coords = np.hstack((coords,np.ones((coords.shape[0],1))))

    #coords_world = coords * affine
    points_ijk = aff_inv * points.T
    points_ijk = points_ijk.T[:,:3]

    print 'ijk'
    max = np.amax(points_ijk, axis=0)
    print max
    min = np.amin(points_ijk, axis=0)
    print min

    scalars = trilinear_interp(img, points_ijk)
    return scalars    

def streams_to_vol(streams, img):
    pass



