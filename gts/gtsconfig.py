import gtsroi
from builtins import str
import hjson
# from builtins import str
from os import getcwd
from os import path
from os.path import abspath
import pandas as pd
from glob import glob
from collections import namedtuple


class GtsConfig(object):
    _CONFIG={
    'tract_time_log_file' : 'tract_time_stats.csv'
    }

    Subject = namedtuple('Subject', 'name group dwi_path freesurfer_path dwi_autodetect_folder tractography_path processed_path')

    def __init__(self,conf_file=None, configure=True):
        self._SIMULATE = False

        # if not conf_file:
            # defaults 
            #self._CONFIG = {}
        self._CONFIG["name"] = "study"
        self._CONFIG["root"] = getcwd()
        self._CONFIG["orig_path"] = "./orig"
        self._CONFIG["preprocessed_path"] = "./preprocessed"
        self._CONFIG["T1_processed_path"] = "./T1s/processed"
        self._CONFIG["T1_dicom_path"] = "./dicom"
        self._CONFIG["T1_autodetect"] = "*T1*"
        self._CONFIG["processed_path"] = "./processed"
        self._CONFIG["subject_dti_path"] = '/media/AMMONIS/projects/controls'
        self._CONFIG["template_roi_path"] = "/media/AMMONIS/proejcts/mtt_anat/C5_to_avgs/processed"
        self._CONFIG['default_dwi_path']  = '/media/AMMONIS/scans/normals/'
        self._CONFIG['dwi_autodetect_folder'] = "*DTI*"
        self._CONFIG['group_paths'] = {}
        self._CONFIG["tractography_path"] = './tractography'
        self._CONFIG["ind_roi_path"] = "./rois"
        self._CONFIG["prefix"]="ANTS_"
        self._CONFIG["affix"]=''
        self._CONFIG["template_rois"] = ['roi_con', 'all-targets_con']
        self._CONFIG["rois"] = ['roi', 'all-targets']
        self._CONFIG["imgext"] = '.nii.gz'
        self.subjects = []

        if conf_file:
            self.conf_file = conf_file
            self.loadFromJson(conf_file)
            if configure:
                self.configure()


    def __getattr__(self,var):
        try:
            return self._CONFIG[var]
        except KeyError:
            return None
            
    def __setattr__(self,var,val):
        if var == '_CONFIG':
            super(GtsConfig, self).__setattr__(var, val)
        else:
            self._CONFIG[var]=val

    def configure(self):
        self.root = getcwd()
        conf_file = self.conf_file
        print conf_file
        basename = ''

        if conf_file:
            self.conf_name = path.basename(conf_file).split('.')[0]
            basename = conf_file.split('.')[0]
        
        for key in self._CONFIG.keys():
            if '_path' in key:
                val = self._CONFIG[key]
                if isinstance(val, str): 
                    self._CONFIG[key] = abspath(val)

        # self.orig_path = abspath(self.orig_path)
        # self.preprocessed_path = abspath(self.preprocessed_path)
        # self.T1_processed_path = abspath(self.T1_processed_path)
        # self.processed_path = abspath(self.processed_path)
        # self.ind_roi_path = abspath(self.ind_roi_path)
        self.tractography_path_full = abspath(self.tractography_path)

        if not self.group_paths:
            self.group_paths = {}
        if not self.collect_maps:
            self.collect_maps = ['default_set']

        self.tract_time_log_file = path.join(self.root, '_'.join([basename, self.tract_time_log_file]))

        if not self.manual_subjects:
            self.load_subjects(self.subjects_file)
        else:
            self.subjects = [self.subj_to_tuple(s) for s in self.subjects]
        print self.subjects            

        if 'rois_def' in self._CONFIG:
            self.rois_def = {k:gtsroi.GtsRoi(k, v, global_config=self) for (k,v) in self.rois_def.iteritems()}        

    def load_subjects(self, filename):
        self.subjects = []

        if filename.find('.txt') > -1:
            with open(filename, "r") as f:
                self.subjects= [ self.subj_to_tuple(l) for l in f ]
        elif filename.find('.csv') > -1:
            self.subject_df = pd.read_csv(filename, skipinitialspace=True)
            for i, name, grp in self.subject_df.itertuples():
                group = str(grp)
                print i,name,group

                myinfo = {
                    "name": name,
                    "group": group
                }

                PASSABLE = {
                    "dwi_path": self.default_dwi_path,
                    "dwi_autodetect_folder": self.dwi_autodetect_folder,
                    "freesurfer_path": self.default_freesurfer_path,
                    "tractography_path": self.tractography_path_full,
                    "processed_path": self.processed_path                    
                }

                for key,val in PASSABLE.iteritems():
                    myinfo[key] = val

                if self.group_paths and group in self.group_paths:
                    for key,val in PASSABLE.iteritems():
                        try:
                            myinfo[key] = self.group_paths[group][key]
                        except KeyError:
                            pass

                self.subjects.append(self.Subject(**myinfo))
        self.subjects_pool = self.subjects

    def subj_to_tuple(self, name):
        return self.Subject(name=name.rstrip(),group='0', dwi_path=self.default_dwi_path, 
            freesurfer_path=self.default_freesurfer_path, dwi_autodetect_folder=self.dwi_autodetect_folder, 
            tractography_path=self.tractography_path_full, processed_path=self.processed_path )

    def loadFromJson(self,conf=""):
        if not conf == "":
            print '#> ',conf
            fp = open(conf, 'r')

            config_map = hjson.load(fp)
            for k,v in config_map.iteritems():
                if k=='import':
                    self.loadFromJson(v)
                self._CONFIG[k] = v
            fp.close()

