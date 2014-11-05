import os
import os.path as path

class TractographyMethod(object):
    def factory(type, subj, seed_config, method_config, global_config):
        from tractography.mrtrix import Mrtrix

        if type=='mrtrix':
            return Mrtrix(subj, seed_config, method_config, global_config)        
        # if type=='xst':
        #     return Xst(subj, seed_config, method_config, global_config)

        # if type=='slicer3':
        #     return Slicer3(subj, seed_config, method_config, global_config)
    factory = staticmethod(factory)

    global_config = {}

    def __init__(self, subj, seed_config, method_config, global_config):
        
        self.global_config = global_config

        self.path = self.global_config.tractography_path+'/'+subj

        self.subject = subj
        self.seed_config = seed_config
        self.method_config = method_config

        self.recompute=False
        self.overwrite=False

    def gotoWorkingPath(self):
        self.path = self.global_config.tractography_path_full+'/'+self.subject
        os.chdir(self.path)

    def resetPath(self):
        os.chdir(self.global_config.root)

    def get_seed_info(self):
        l =  [self.seed_config['source']]
        return self.get_info(l)[0]

    def get_includes_info(self):
        if not 'includes' in self.seed_config:
            return None

        l =  self.seed_config['includes']
        return self.get_info(l)

    def get_excludes_info(self):
        if not 'excludes' in self.seed_config:
            return None

        l =  self.seed_config['excludes']
        return self.get_info(l)

    def get_unique_name(self):
        method_name = self.method_config['label']
        seed_name = self.seed_config['name']
        return method_name+'_'+seed_name

    def get_info(self, blist):
        roi_defs = self.global_config.rois_def
        res = []
        for i in blist:
            roi = roi_defs[i]
            item = {'name':roi.get_basename(), 'filename':roi.get_filename(self.subject), 'label':roi.label}
            res.append(item)
        return res    

    def compute_tensors(self):
        return None


    def extract_label_from_image(self, filename, label, name=None, binarize=True, save=False):
        import nibabel as nib
        import numpy as np
        img = nib.load(path.join(self.path, filename))
        data = img.get_data()        
        if binarize:
            data = np.array(data == int(label), dtype=np.uint8)
        else:
            data = np.array(data[data > 0], dtype=np.uint8)

        if not save:
            return data, img.get_affine()
        else:
            if not name:
                name = '%d' % label
            filename = self.seed_config['name']+'_'+name+'.nii.gz'
            nib.save(nib.Nifti1Image(data, img.get_affine()), filename)
            return filename

    def combine_masks(self, maps_list, name='combine'):
        """
        Given a list of maps [{"name":"foo", "label":"1"},...]
        combine them into a single binarized mask and return roi object
        """
        import nibabel as nib
        import numpy as np
        data = None
        aff = None
        for i in maps_list:
            filename = path.join(self.path, i['filename'])
            ndata, aff = self.extract_label_from_image(filename, i['label'])
            if not data:
                data = ndata
            else:
                data = np.add(data, ndata)

        data = np.array(data > 0, dtype=np.uint8)
        filename = self.seed_config['name']+'_'+name+'.nii.gz'
        nib.save(nib.Nifti1Image(data, aff), filename)
        return filename

    def run(self):
        raise NotImplementedError()