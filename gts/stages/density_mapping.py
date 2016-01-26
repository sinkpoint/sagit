import gts
from os import path
import json

def density_mapping(tracts, dry_run=False, name=''):
    global conf_name
    if not dry_run:
        gts.tracts_to_density(tracts)

    conj_files = gts.tracts_conjunction(tracts, img_type='binary',dry_run=dry_run)
    print conj_files

    conj_files_list = conf_name+name+'_conj_files.txt'

    with open(conj_files_list, 'w') as outfile:
        json.dump(conj_files, outfile, sort_keys=True, indent=4, separators=(',', ': '))

    with open(conj_files_list, 'r') as fp:
        conj_files = json.load(fp)

    import pandas as pd
    conj_df = pd.DataFrame(conj_files)
    conj_df.drop(0,axis=1,inplace=True)
    conj_df.to_csv(path.basename(conj_files_list)+'.csv')


    fig_list, basename = gts.conjunction_to_images(conj_files, name='nobg', bg_file='', dry_run=dry_run)
    if not dry_run:
        gts.conjunction_images_combine(fig_list, basename=basename, group_names=['nobg'])

    fig_list, basename = gts.conjunction_to_images(conj_files, name='bg', bg_file=path.join(gts.config.orig_path,gts.config.group_template_file), slice_indices=(128,128,80), dry_run=dry_run)
    if not dry_run:    
        gts.conjunction_images_combine(fig_list, basename=basename, group_names=['bg'])
