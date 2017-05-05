# -*- coding: utf-8 -*-
from gts.gtsutils import exec_cmd
from os import path
import os
import multiprocessing
from glob import glob

def per_subj_ants_dwi_to_t1(self, subject, **kwargs):
    overwrite = False
    try:
        overwrite = kwargs['overwrite']
    except KeyError:
        pass
    
    print '''
    ====================
    ╔╦╗╦ ╦╦  ┌┬┐┌─┐  ╔╦╗
     ║║║║║║   │ │ │   ║ 
    ═╩╝╚╩╝╩   ┴ └─┘   ╩ 
    ====================
    {}
    '''.format(subject.name)
    print kwargs
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(multiprocessing.cpu_count())


    inputDir = self.config.preprocessed_path            #folder containing anatomical images
    outDir = self.config.processed_path          #outputdir of normalized T1 files

    #specify parameters for antsIntroduction.sh
    #compulsory arguments
    ImageDimension = 3
    OutPrefix = 'ANTS_'

    #optional arguments
    IgnoreHDRWarning = 1
    MaxIteration = '40x90x60'
    N3Correct = 0
    QualityCheck = 1
    MetricType = 'PR'
    TransformationType = 'GR'

    ITS = " -i 100x100x30 " # 3 optimization levels
    # different transformation models you can choose
    TAFFINE = " -t Affine[0.1]"
    TSYNWITHTIME = " -t SyN[0.25,5,0.01] " #" -r Gauss[3,0] " # spatiotemporal (full) diffeomorphism
    TGREEDYSYN = " -t SyN[0.15,3.0,0.0] " #" -r Gauss[3,0] " # fast symmetric normalization 
    TBSPLINESYN = " -t BSplineSyN[0.1,3.0,0.0] " #" -r Gauss[3,0] " # fast symmetric normalization 
    TELAST = " -t Elast[1] -r Gauss[0.5,3] " # elastic
    TEXP = " -t Exp[0.5,10] -r Gauss[0.5,3] " # exponential

    subj  =  subject.name
    T1 = "{0}_T1_bet.nii.gz".format(subj)

    through = 'AP' # use AP by default
    if 'through' in kwargs:
        through  =  kwargs['through']
        
    MEDIUM = "{0}_{1}_bet.nii.gz".format(subj, through) 

    outRoot = OutPrefix+subj
    warp_file  =  path.join(outDir, outRoot+'1Warp.nii.gz')
    
    OUT = path.join(outDir,outRoot)
    FIXED = path.join(inputDir, T1)
    MOVING = path.join(inputDir, MEDIUM)    

    if not path.isfile(warp_file) or overwrite:
        # different metric choices for the user
        INTMSQ = " -m MSQ[{0},{1},1,0] ".format(FIXED, MOVING)
        INTMI = " -m MI[{0},{1},1,32] ".format(FIXED, MOVING)
        INTCC = " -m CC[{0},{1},1,3] ".format(FIXED, MOVING)
        INTMATTS = " -m Mattes[{0},{1},1,32] ".format(FIXED, MOVING)
        
        INT = INTMI
        TRANS = TGREEDYSYN

        cmd = ('time antsRegistration -d {ImageDimension} -o {OUT}'
             ' {TAFFINE} {INTMI} --convergence [10000x10000x10000x10000x10000] --shrink-factors 5x4x3x2x1 --smoothing-sigmas 4x3x2x1x0mm'
             ' {TGREEDYSYN} {INTCC} --convergence [50x35x15,1e-7] --shrink-factors 3x2x1 --smoothing-sigmas 2x1x0mm  --use-histogram-matching 1 '
             ' -z 1 ').format(**locals())

        exec_cmd(cmd)

    else:
        print 'Deformation info already exists'

    cmd = "antsApplyTransforms -d 3 -i {MOVING} -r {MOVING} -n BSpline -o {OUT}_deformed_DWS.nii.gz -t {OUT}1Warp.nii.gz -t {OUT}0GenericAffine.mat ".format(**locals())
    exec_cmd(cmd)
    
    cmd = "antsApplyTransforms -d 3 -i {MOVING} -r {FIXED} -n BSpline -o {OUT}_deformed_T1.nii.gz -t {OUT}1Warp.nii.gz -t {OUT}0GenericAffine.mat ".format(**locals())  
    exec_cmd(cmd)

    cmd = "antsApplyTransforms -d 3 -i {FIXED} -r {FIXED} -n BSpline -o {OUT}_invDeformed_T1.nii.gz -t [{OUT}0GenericAffine.mat,1] -t {OUT}1InverseWarp.nii.gz".format(**locals())
    exec_cmd(cmd)

    cmd = "antsApplyTransforms -d 3 -i {FIXED} -r {MOVING} -n BSpline -o {OUT}_invDeformed_DWS.nii.gz -t [{OUT}0GenericAffine.mat,1] -t {OUT}1InverseWarp.nii.gz".format(**locals())
    exec_cmd(cmd) 


def per_subj_ants_t1_to_template(self, subject, **kwargs):
    c = self.config
    subj = subject.name

    overwrite = False
    try:
        overwrite = kwargs['overwrite']
    except KeyError:
        pass

    print '''
    ======================================
    ╔╦╗  ╔╦╗┌─┐  ╔╦╗┌─┐┌┬┐┌─┐┬  ┌─┐┌┬┐┌─┐
     ║    ║ │ │   ║ ├┤ │││├─┘│  ├─┤ │ ├┤ 
     ╩    ╩ └─┘   ╩ └─┘┴ ┴┴  ┴─┘┴ ┴ ┴ └─┘
    ====================================== 
    {}
    '''.format(subj)
    print '------------------- T1 To TEMPLATE %s -------------------' % subj
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(multiprocessing.cpu_count())

    NUM_THREADS=multiprocessing.cpu_count()
    overwrite=0
    
    #specify folder names
    template_file = path.join(c.orig_path, c.group_template_file)
    t1_dir = "group_template"
    inputDir = path.join(c.T1_path)
    outDir = path.join(c.T1_processed_path)

    #specify parameters for antsIntroduction.sh
    #compulsory arguments
    ImageDimension=3
    OutPrefix='ANTS_'

    #If not created yet, let's create a new output folder
    if not path.exists(outDir):
        os.mkdir(outDir)

    mov="{inputDir}/{subj}.nii.gz".format(**locals())
    ref=template_file
    out="{outDir}/{OutPrefix}{subj}".format(**locals())

    existing = glob(out+'1InverseWarp*')
    if len(existing) > 0 and not overwrite:
        print 'found files matching ',out
        print 'skipping'
        return

    cmd="time antsRegistrationSyN.sh -d {ImageDimension} -m {mov} -f {ref} -o {out} -t s -n {NUM_THREADS}".format(**locals())
    exec_cmd(cmd)
