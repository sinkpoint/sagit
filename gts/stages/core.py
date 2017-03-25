from gts import exec_cmd
import os
from glob import glob
import shutil

def per_subj_collect_t1_from_dicom(self, subject, *args, **kwargs):
    c = self.config
    collect_path = c.T1_path    
    t1_dicom = c.T1_dicom_path
    t1_search = c.T1_autodetect

    print '------------ collect T1 from dicom --------------'

    subj = subject.name
    for match in t1_search:
        search_path = os.path.join(t1_dicom,subj,match)
        print search_path
        gb = glob(search_path)
        if len(gb) == 0:
            print subj,match,'T1 not found'
            continue

        dicom_folder = gb[0]

        print 'found',dicom_folder
        tmp_output = '/tmp/{subj}'.format(**locals())
        if not os.path.exists(tmp_output):
            os.mkdir(tmp_output)

        exec_cmd('dcm2nii -d N -e N -f Y -i N -p N -o {tmp_output} {dicom_folder}'.format(**locals()))

        myfile = glob('{tmp_output}/*.nii.gz'.format(**locals()))[0]
        dest = os.path.join(collect_path,'{subj}.nii.gz'.format(**locals()))
        shutil.move(myfile, dest)
        print 'saved to',dest
        break





    