# -*- coding: utf-8 -*-

import shutil
import os.path as path
import os

def per_subject_prepareRoiFiles(self, subject, **kwargs):
    ############# Deal with ROIs 

    '''
    pseudo code

    for roi object in roi definitions
        roi_obj.generate(subject)
    for seeds
        seeds seed the relevant 
    '''
    c = self.config
    subj = subject.name        
    processed_path = c.processed_path
    tractography_path = subject.tractography_path
    subjdir = path.join( tractography_path,subj)

    print '''
    =============================================
    ╔═╗┬─┐┌─┐┌─┐┌─┐┬─┐┌─┐  ╦═╗╔═╗╦  ╔═╗┬┬  ┌─┐┌─┐
    ╠═╝├┬┘├┤ ├─┘├─┤├┬┘├┤   ╠╦╝║ ║║  ╠╣ ││  ├┤ └─┐
    ╩  ┴└─└─┘┴  ┴ ┴┴└─└─┘  ╩╚═╚═╝╩  ╚  ┴┴─┘└─┘└─┘
    =============================================
    '''

    root = os.getcwd()
    os.chdir(subjdir)
    print 'copy ROI file to '+subjdir

    # copy t1 projected roi files to individual tractography folder    
    rois = c.rois_def
    for k, roi in rois.iteritems():
        if roi.type == 'from_template':
            roi_file = roi.get_filename(subj)

            shutil.copyfile(path.join(processed_path,roi_file), path.join(subjdir,roi_file))
            roi_filebase = roi_file.split('.')[0]

            # cmd="slicerFileConvert.sh %s %s " % (roi_file, roi_filebase+'.nhdr');
            # exec_cmd(cmd, display=False)

            roi_file = roi.get_filename(subj, ref='t1')
            shutil.copyfile(path.join(processed_path,roi_file), path.join(subjdir,roi_file))
    os.chdir(root)
