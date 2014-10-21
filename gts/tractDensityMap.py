import numpy as np
import nibabel as nib
import argparse
import os.path
import sys
from multiprocessing import Pool
from vtkFileIO import vtkToStreamlines
import nipype.interfaces.mrtrix.convert as mrt

_DEBUG = 0

def run(args):
    outbasename = args.out.split('.')[0]

    tracts_to_density(args.ref, args.in_fiber, outbasename)


def tracts_to_density(ref, tracts, basename=''):
    ref_image = nib.load(ref)
    streamlines = file_to_streamline(tracts)

    outImage, outFibImage, outBinImage = streamlines_to_density(ref_image, streamlines)

    if basename=='':
        basename = os.path.splitext(tracts)[0]

    nib.save(outImage, basename+'_den.nii.gz')
    nib.save(outFibImage, basename+'_fib.nii.gz')
    nib.save(outBinImage, basename+'_bin.nii.gz')    

def file_to_streamline(filename):
    ext = os.path.splitext(filename)

    if ext[1]=='.vtk':
        return vtkToStreamlines(filename)
    elif ext[1]=='.tck':
        print filename
        hdr, streamlines = mrt.read_mrtrix_tracks(filename, as_generator=False)
        return np.array(streamlines)


def streamlines_to_density(ref_image, streamlines):
    """ Generates density image from a set of streamlines.

    Args:
        ref_image: Reference nibabel object, output image uses its spacing and affine.
        streamlines: set of streamlines, array of n3 array.
    Returns:
        [densityImage, fiberCountImage, binaryMaskImage]
    """

    imgdim = ref_image.get_shape()[:3]
    outData = np.zeros(imgdim)
    outBinData = np.zeros(imgdim)
    outFibData = np.zeros(imgdim)

    print '# of streamlines: %d' % len(streamlines)
    if len(streamlines) > 0:

        points = np.concatenate(streamlines)    

        #points = np.array([[100,100,20,1],[101,101,21,1]])
        pones = np.ones([points.shape[0],1])

        #stream_bounds = np.matrix(bounding_box(points))
        #stream_bounds =  np.vstack((stream_bounds,[1,1])).T

        rheader = ref_image.get_header()
        ijk2rasMx = np.matrix(rheader.get_best_affine())

        ras2ijkMx = ijk2rasMx.I

        #points = (points * ijk2rasMx.T)[:,:3]
        #st_ijk_bounds = np.int_(stream_bounds * ras2ijkMx.T)
        pt_ijk = np.int_(np.append(points, pones, axis=1) * ras2ijkMx.T)

        # get unique voxel index for tract mask

        b = np.ascontiguousarray(pt_ijk).view(np.dtype((np.void, pt_ijk.dtype.itemsize * pt_ijk.shape[1])))
        pt_ijk3_unique = np.unique(b).view(pt_ijk.dtype).reshape(-1, pt_ijk.shape[1])
        pt_ijk3_unique = pt_ijk3_unique[:,:3]


        outBinData[pt_ijk3_unique[:,0],pt_ijk3_unique[:,1],pt_ijk3_unique[:,2]] = 1

        # output binary tract mask

        for i in pt_ijk:
            p = np.ravel(i)        
            #vals = getIjkDensity(streamlines, ref_image, i) * 100
            #outData[i[0],i[1],i[2]] = vals[0]
            outFibData[p[0],p[1],p[2]] += 1        
            
        outData = np.divide(outFibData, len(streamlines))

    outImage = nib.Nifti1Image( outData, ref_image.get_affine() )
    outFibImage = nib.Nifti1Image(outFibData, ref_image.get_affine())
    outBinImage = nib.Nifti1Image( outBinData, ref_image.get_affine() )

    return outImage, outFibImage, outBinImage


def myrange(i,j):
    if i>j:
        return range(j,i)
    else:
        return range(i,j)

def bounding_box(nparray):
    """
    Find max and min values given a Nx3 numpy array
    nparray -- Nx3 numpy array
    Returns [(min_x, max_x), (min_y, max_y), (min_z, max_z)]
    """
    min_x, min_y, min_z = nparray.min(axis=0)
    max_x, max_y, max_z = nparray.max(axis=0)

    return (min_x, max_x), (min_y, max_y), (min_z, max_z)

def getIjkDensity(streamlines, ref_image, index):
    """
    docs
    """

    rheader = ref_image.get_header()
    ijk2rasMx = rheader.get_best_affine()

    # get voxel center in RAS
    ijkvec = np.matrix(np.append(index,1))

    voxel_ras = ijkvec * ijk2rasMx.T
    voxel_ras = np.array(voxel_ras).flatten()[:3]
    # get the bounds of voxel

    shape = rheader.get_zooms()[:3]
    minCorner = voxel_ras - shape
    maxCorner = voxel_ras + shape


    vox_density, num_fiber = getDensityOfBounds(streamlines, minCorner, maxCorner)

    #print index, vox_density, num_fiber
    sys.stdout.write('.')
    sys.stdout.flush()

    return vox_density, num_fiber


def getDensityOfBounds( streamlines, minCornerRAS, maxCornerRAS):
    num_pass = 0
    for line in streamlines:
        if np.any(np.all(line <= maxCornerRAS, axis=1)) and np.any(np.all(line>= minCornerRAS, axis=1)):
            num_pass+=1
#            print num_pass,'pass!'

#        continue
#        for point in line:
#            print point
#            print point < maxCornerRAS,point > minCornerRAS
#            if np.all(point < maxCornerRAS) and np.all(point > minCornerRAS):
#                print point
#                num_pass += 1
#                continue

    numFibers = len(streamlines)
    density = float(num_pass)/float(numFibers)

    return density, num_pass


def debug( msg ):
    global _DEBUG
    if _DEBUG:
        print msg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts vtk tracts to nifti density image")
    parser.add_argument("-f", "--in_fiber", required=True, dest="in_fiber",help="Input fiber bundle in vtk")
    parser.add_argument("-m", "--mask", dest="mask",help="Integer binary mask")
    parser.add_argument("-r", "--reference", dest="ref",help="Reference image to use as image space")
    parser.add_argument("-s", "--resolution", dest="res",help="If no reference image, then the resolution of voxel, default=1")
    parser.add_argument("-o", "--output", dest="out",help="Output nifti filename")

    args =  parser.parse_args()

    run(args)