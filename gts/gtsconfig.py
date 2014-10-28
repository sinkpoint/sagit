import os
import json
from jsoncomment import JsonComment

class GtsConfig(object):
    _CONFIG={}
    def __init__(self,conf_file=""):
        self._SIMULATE = False

        if not conf_file == "":
            self.loadFromJson(conf_file)
            self.root = os.getcwd()
            self.orig_path = self.root+'/'+self.orig_path
            self.preprocessed_path = self.root+'/' + self.preprocessed_path
            self.T1_processed_path = self.root + '/' + self.T1_processed_path
            self.processed_path = self.root+'/'+self.processed_path
            self.tractography_path_full = self.root+'/'+self.tractography_path

            if not self.manual_subjects:
                with open(self.subjects_file, "r") as f:
                    self.subjects= [ l.rstrip() for l in f ]

            print self.subjects            
            return

        self._CONFIG = {}
        self._CONFIG["root"] = os.getcwd()
        self._CONFIG["orig_path"] = "./orig"
        self._CONFIG["preprocessed_path"] = "./preprocessed"
        self._CONFIG["T1_processed_path"] = "./T1s/processed"
        self._CONFIG["processed_path"] = "./processed"
        self._CONFIG["subject_dti_path"] = '/media/AMMONIS/projects/controls'
        self._CONFIG["template_roi_path"] = "/media/AMMONIS/proejcts/mtt_anat/C5_to_avgs/processed"
        self._CONFIG['dwi_base_path']  = '/media/AMMONIS/scans/normals/'
        self._CONFIG["tractography_path"] = './tractography'
        self._CONFIG["ind_roi_path"] = "./rois"
        self._CONFIG["prefix"]="ANTS_"
        self._CONFIG["affix"]=''
        self._CONFIG["template_rois"] = ['roi_con', 'all-targets_con']
        self._CONFIG["rois"] = ['roi', 'all-targets']
        self._CONFIG["imgext"] = '.nii.gz'
        self.subjects = []



    def __getattr__(self,var):
        try:
            return self._CONFIG[var]
        except KeyError:
            return False
    def __setattr__(self,var,val):
        if var == '_CONFIG':
            super(GtsConfig, self).__setattr__(var, val)
        else:
            self._CONFIG[var]=val

    def loadFromJson(self,conf=""):
        if not conf == "":
            print '>',conf
            fp = open(conf, 'r')
            parser = JsonComment(json)
            config_map = parser.load(fp)
            for k,v in config_map.iteritems():
                if k=='import':
                    self.loadFromJson(v)
                self._CONFIG[k] = v
            fp.close()

