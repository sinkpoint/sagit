#!/usr/bin/python

from gts import vtkFileIO as vfio
from gts.groupTractStats import *
import json
import sys
import os

conf_file = sys.argv[1]
gts = GroupTractStats(conf_file)

with open(sys.argv[2], 'r') as fp:
    track_conf = json.load(fp)

track_name =  track_conf[sys.argv[3]][0]

tracks = []
def getTrackPerSubj(subj, c):
    global tracks
    from scipy.io import loadmat

    tfile = os.path.join(os.getcwd(),'tractography',subj, track_name)
    #os.path.join('tractography',subj,track_conf['cst_sd'])
    # dwi_t1_afname = os.path.join(c.processed_path, 'ANTS_%s0GenericAffine.mat' % subj)
    # dwi_t1_affine = loadmat(dwi_t1_afname)

    # t1_avg_afname = os.path.join(c.processed_path, 'ANTS_%s0GenericAffine.mat' % subj)
    # dwi_t1_affine = loadmat(dwi_t1_afname)


    # tracks.append((tfile,affine))
    tracks.append(tfile)
gts.runPerSubject(getTrackPerSubj)

data = {'file':tracks, 'streamlines':[], 'aligned_strms':[],'num_max':0, 'ref_idx':-1}


for i,fval in enumerate(data['file']):
    print 'loading %s' % fval
    strm, pdata = vfio.vtkToStreamlines(fval)
    data['streamlines'].append(strm)

    num_strms = len(strm)
    rank = 0
    if num_strms > 0:
        points = np.concatenate(strm)
        corner_max = np.amax(points, axis=0)
        corner_min = np.amin(points, axis=0)

        dist = np.sum(np.power(corner_max-corner_min,2))
        rank = dist*num_strms



    if rank > data['num_max']:
        data['num_max'] = rank
        data['ref_idx'] = i

print data['num_max'], data['ref_idx']

from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.tracking.streamline import set_number_of_points


ref_idx = data['ref_idx']
p_per_strm =20

ref_vec = set_number_of_points(data['streamlines'][ref_idx], p_per_strm)

srr = StreamlineLinearRegistration()

for i,strm in enumerate(data['streamlines']):
    print 'registering %d/%d' % (i,len(data['file'])-1)
    print '# streamlines = %d' %len(strm)
    if len(strm) == 0 or i==ref_idx:
        print 'skipping'
        continue
    mov_vec = set_number_of_points(strm, 20)
    srm = srr.optimize(static=ref_vec, moving=mov_vec)
    data['aligned_strms'].append(srm.transform(mov_vec))

from dipy.viz import fvtk
ren = fvtk.ren()
ren.SetBackground(1., 1, 1)

reflines = fvtk.streamtube(ref_vec, fvtk.colors.red, linewidth=0.2)
fvtk.add(ren, reflines)

for (i, bundle) in enumerate(data['aligned_strms']):
    lines = fvtk.streamtube(bundle, np.random.rand(3), linewidth=0.1)
    # lines.RotateX(-90)
    # lines.RotateZ(90)
    fvtk.add(ren, lines)


fvtk.show(ren)
