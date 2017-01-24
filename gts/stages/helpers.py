from gts import exec_cmd
import os

def per_subject_dwi_roi_overlay(self, subject, **kwargs):
    c = self.config
    subj = subject.name
    processed_path = c.processed_path
    tractography_path = c.tractography_path_full
    subjdir = os.path.join( tractography_path,subj)
    output_path = os.path.join(self.config.processed_path, 'tractography')

    if 'roi_name' in kwargs:
        roi_name = kwargs['roi_name']
    print '============================',subj


    exec_cmd('slicer {subjdir}/{subj}_FA.nii.gz {subjdir}/ANTS_{subj}_{roi_name}_dwi.nii.gz -L -e -0 -A 1920 {output_path}/{subj}_{roi_name}_overlay.png'.format(**locals()))

