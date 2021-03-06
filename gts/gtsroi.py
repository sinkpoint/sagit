import groupTractStats as gts
import nibabel as nib
import numpy as np
import os
import scipy.ndimage as simg
import types

from pyparsing import Forward
from pyparsing import Group
from pyparsing import Literal
from pyparsing import ParseException
from pyparsing import Suppress
from pyparsing import Word
from pyparsing import alphas
from pyparsing import alphanums
from pyparsing import delimitedList
from pyparsing import nums
from pyparsing import oneOf
import pyparsing as pp



class GtsRoi(object):
    """
    Class holding definition of a region of interest (ROI)
    @param def_conf 
        Defined by json format:
            "name": {
                "type": "from_template",
                "image" : "vestib_roi",
                "label": "raw label definitions; see label expression syntax",
            },      
    pyparsing documentation:
    http://infohost.nmt.edu/~shipman/soft/pyparsing/web/index.html
    """
    aseg_file = 'aseg_dwi.nii.gz'

    def __init__(self, name, config_map, global_config=None):
        if config_map:
            self._CONFIG = config_map
            self.name = name
            self.global_config = global_config

    def __getattr__(self,var):
        try:
            return self._CONFIG[var]
        except KeyError:
            return False

    def __setattr__(self,var,val):
        if var == '_CONFIG':
            super(GtsRoi, self).__setattr__(var, val)
        else:
            self._CONFIG[var]=val

    def get_basename(self):
        if self.type=='from_template':
            gc = self.global_config
            template_name = gc.template_def[self.image].split('.')[0]
            return template_name

        elif self.type=='freesurfer':
            return self.aseg_file.split('.')[0]

        elif self.type=='composite':
            return self.name

    def get_filename(self, subj, ref='dwi', autogen=True):
        if self.type == 'direct':            
            return self.filename

        elif self.type=='from_template':
            gc = self.global_config
            template_name = gc.template_def[self.image].split('.')[0]
            projected_filename = gts.getDwiRoiName(gc,subj,template_name,ref=ref)+'.nii.gz'
            return projected_filename        

        elif self.type=='freesurfer' or self.type=='composite':
            gen_filename = self.name+'_gen.nii.gz'
            if autogen:
                gen_filename = self.generate(subj, filename=gen_filename)

            return gen_filename


    def goto_working_path(self, subj):
        self.origin_path = os.getcwd()
        path = self.global_config.tractography_path_full+'/'+subj
        os.chdir(path)

    def reset_path(self):
        if not self.origin_path:
            os.chdir(self.global_config.root)
        else:
            os.chdir(self.origin_path)

    def generate(self, subj_name, ref='dwi', filename=''):
        print subj_name, ref, filename

        # image functions, always return (filename, associated_label(default to 1))
        def get_data(param_tuple):
            if type(param_tuple[0]) == np.ndarray:
                data = param_tuple[0]
                return data
            else: 
                img = nib.load(str(param_tuple[0]))
                data = img.get_data()                

            return np.array(data==int(param_tuple[1]), dtype=np.uint8)
        def img_select(img):
            return (get_data(img),None)

        def img_intersect(img1, img2):
            print img1,img2
            print os.getcwd()
            data1 = get_data(img1)
            data2 = get_data(img2)
            data = np.multiply(data1, data2)
            return (np.array(data>0,dtype=np.uint8), None)

        def img_union(img1, img2):
            data1 = get_data(img1)
            data2 = get_data(img2)

            data = data1+data2
            data = np.array(data>0, dtype=np.uint8)

            return (data, None)            

        def img_sub(img1, img2):
            data1 = get_data(img1)
            data2 = get_data(img2)

            data = np.subtract(data1,data2)
            data = np.array(data > 0, dtype=np.uint8)

            return (data, None)

        def img_dilate(img, amount=1):
            data = get_data(img)
            data = np.array(simg.morphology.binary_dilation(data,iterations=amount), dtype=np.uint8)
            return (data,None)


        def func_factory(name):
            if name=='select':
                return img_select
            if name=='intersect':
                return img_intersect
            if name=='sub':
                return img_sub
            if name=='dilate':
                return img_dilate
            if name=='union':
                return img_union
            else:
                raise NameError('expression function %s is not found/implemented',name)


        def exec_parsed(parsed, subj_name):
            """
            Parse the image processing expression, recursively traverse a nested list that 
            describes it, and execute it.
            """
            print parsed
            if isinstance(parsed, list):
                # recursively descend
                func_name = parsed[0]
                args = []
                for i in range(1,len(parsed)):
                    param = exec_parsed(parsed[i], subj_name)
                    args.append(param)
                func = func_factory(func_name)
                print func_name, args
                return func(*args)                                

            # leaf processing
            try:                
                parsed = int(parsed)
                # return parameter as integer                 
                return parsed                
            except (TypeError, ValueError), e:
                try:
                    parsed = str(parsed)

                    if '@' in parsed:
                        # freesurfer reference, return freesurfer seg file
                        parsed = parsed.split('@')[1]
                        # return parameter as (file, label) tuple
                        return (subj_name+'_'+self.aseg_file, parsed)
                    elif '$' in parsed:
                        parsed = parsed.split('$')[1]
                        # return the generated file name
                        refroi = self.global_config.rois_def[parsed]
                        mlabel = 1
                        if refroi.type == 'from_template':
                            mlabel = refroi.label
                        return (refroi.get_filename(subj_name),mlabel)

                except Exception, e:
                    return None



        self.goto_working_path(subj_name)

        from pyparsing import WordStart

        iw = alphas+'$_'

        exp = Forward()
         
        funcname = Word(alphas) 
        labels = pp.Combine('@'+Word(nums))
        imglabel = pp.Combine('$'+Word(pp.alphanums+'_'))
        number = Word(nums).setParseAction(lambda s, l, t: int(t[0]))

        lparen = Literal('(').suppress()
        rparen = Literal(')').suppress()

        exp << Group(funcname + lparen + delimitedList( labels |  imglabel | number | exp ) + rparen)

        print '>',self.image
        etree = exp.parseString(self.image).asList()[0]
        data, _ = exec_parsed(etree,subj_name)

        if not filename:
            filename = self.name+'_gen.nii.gz'

        reffile = subj_name +'_'+ self.aseg_file
        affine = nib.load(reffile).get_affine()
        img = nib.Nifti1Image(data, affine)

        nib.save(img, filename)
        self.label = '1'

        self.reset_path()

        return filename





