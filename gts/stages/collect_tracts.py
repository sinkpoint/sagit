# -*- coding: utf-8 -*-

from gts import exec_cmd
import os
from glob import glob

def find_scalar_file(mapname):
    """
    Map scalar map based on its basename in the individual tractography folder.

    Args:
        mapname: The image file basename, exclude the extensions
    """
    mapfile = glob('{mapname}.nii.gz'.format(**locals()))
    if len(mapfile) == 0:
        mapfile = glob('{mapname}_dwi.nii.gz'.format(**locals()))
        if len(mapfile) == 0:
            print 'WARNING',mapname,' not found during collecting maps.'
            return None

    return mapfile[0]


def per_subj_tract_to_template_space(self, subject, **kwargs):
    """
    Per subject function, project vtk/vtp tracts from DWI to template space
    
    Args:
        subject: The current subject object
        dry_run=False: Mock run only
    """

    dry_run = False
    if 'dry_run' in kwargs:
        dry_run = kwargs['dry_run']
        
    subj = subject.name
    print '''
╔╦╗┬─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐  ╔╦╗┌─┐┌┬┐┌─┐┬  ┌─┐┌┬┐┌─┐
 ║ ├┬┘├─┤│   │    ║ │ │   ║ ├┤ │││├─┘│  ├─┤ │ ├┤ 
 ╩ ┴└─┴ ┴└─┘ ┴    ╩ └─┘   ╩ └─┘┴ ┴┴  ┴─┘┴ ┴ ┴ └─┘    
 ================================================
    '''
    print subj
    tract_path = subject.tractography_path
    output_path = os.path.join(self.config.processed_path, 'tractography')

    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    streamnames = kwargs['tracts']

    os.chdir(os.path.join(tract_path, subj))

    # transform applied to points is inverse of the desired direction 
    dwi_t1_arg, r = self.get_ants_t1_to_dwi_transforms(subj, inverse=False, reference='t1')
    t1_avg_arg, ref_avg = self.get_ants_t1_to_avg_transforms(subj, inverse=False, reference='average')

    tparams = ' '.join((t1_avg_arg, dwi_t1_arg))

    tdb_file = '_'.join([subj,'tracts.tdb'])

    print 'tract file: ',tdb_file

    if os.path.isfile(tdb_file):
        print '> remove old %s' % (tdb_file)
        os.remove(tdb_file)    
    # cmd = 'rm '+tdb_file
    # exec_cmd(cmd, dryrun=dry_run)
    file_queue = []

    maps = []
    collect_settings = self.config.collect_maps
    if not collect_settings or 'default_set' in collect_settings:
        maps = ['FA','RD','AD','MD']

    print collect_settings
    for m in collect_settings:
        if m == 'default_set':
            continue
        maps.append(m)

    maps = list(set(maps))

    print '>>>',maps

    for method, file_list in streamnames.iteritems():
        print '== Method %s' % method
        for tfile in file_list:
            print '>',tfile
            trkwscalar = tfile.split('.')[0]+'.vtp'
            input_tract = tfile

            for mtype in maps:
                mapname = '_'.join([subj, mtype])
                mapfile = find_scalar_file(mapname)
                if not mapfile:
                    # try to find in preprocessed files
                    mapname = os.path.join(self.config.preprocessed_path, mapname+'*')
                    mapfile = find_scalar_file(mapname)
                    if not mapfile:
                        continue                    

                cmd = 'copyScalarsToTract.py -i {input_tract} -o {trkwscalar} -m {mapfile} -n {mtype}'.format(**locals())
                exec_cmd(cmd, dryrun=dry_run)

                if os.path.isfile(trkwscalar):
                    input_tract = trkwscalar
            # cmd = 'copyScalarsToTract.py -i %s -o %s -m %s_RD.nii.gz -n RD' % (trkwscalar, trkwscalar, subj)

            # exec_cmd(cmd, dryrun=dry_run)
            # cmd = 'copyScalarsToTract.py -i %s -o %s -m %s_FA.nii.gz -n FA' % (trkwscalar, trkwscalar, subj)

            # exec_cmd(cmd, dryrun=dry_run)
            # cmd = 'copyScalarsToTract.py -i %s -o %s -m %s_MD.nii.gz -n MD' % (trkwscalar, trkwscalar, subj)            

            exec_cmd(cmd, dryrun=dry_run)
            cmd = 'fascicle add -i %s -d %s' % (trkwscalar, tdb_file)

            exec_cmd(cmd, dryrun=dry_run)

            file_queue.append(trkwscalar)

    # dump points to csv
    points_before = 'trk_points_dws.csv'
    points_after = 'trk_points_tps.csv'

    cmd = 'fascicle expcsv -d %s -o %s' % (tdb_file, points_before)
    exec_cmd(cmd, dryrun=dry_run)

    # transform points to template space
    cmd = 'antsApplyTransformsToPoints -d 3 -i %s -o %s %s' % (points_before, points_after, tparams)
    exec_cmd(cmd, dryrun=dry_run)

    mapping_name = 'template_space'
    # reimport transformed points to tdb
    cmd = 'fascicle tradd -d %s -i %s -n %s -p "%s"' % (tdb_file, points_after, mapping_name, tparams)
    exec_cmd(cmd, dryrun=dry_run)

    out_path = os.path.join(self.config.processed_path, 'tractography')
    if not os.path.isdir:
        os.mkdir(out_path)

    # export the transformed points into tract vtk
    for tfile in file_queue:
        out_name = os.path.join(out_path,'_'.join([subj,tfile]))
        
        if os.path.isfile(out_name):
            print '> remove old %s' % (out_name)
            os.remove(out_name)

        cmd = 'fascicle expvtk -d %s -t %s -m %s -o %s' % (tdb_file,tfile,mapping_name,out_name)

        exec_cmd(cmd, dryrun=dry_run)

def tracts_merge(self, **kwargs):
    """
    Merges tracts in template space into Fascicle .tdb file, then export as one .vtp

    Args:
        dry_run=False: Just do a mock run
        overwrite=True: Overwrite the existing .tdb file
    """
    
    dry_run = False
    overwrite = True

    if 'dry_run' in kwargs:
        dry_run = kwargs['dry_run']
    if 'overwrite' in kwargs:
        overwrite = kwargs['overwrite']
        
    print '''
    ===================================
    ╔╦╗┬─┐┌─┐┌─┐┌┬┐┌─┐  ╔╦╗┌─┐┬─┐┌─┐┌─┐
     ║ ├┬┘├─┤│   │ └─┐  ║║║├┤ ├┬┘│ ┬├┤ 
     ╩ ┴└─┴ ┴└─┘ ┴ └─┘  ╩ ╩└─┘┴└─└─┘└─┘
    ===================================
    '''

    # tract_path = self.config.tractography_path_full
    output_path = os.path.join(self.config.processed_path, 'tractography')
    print output_path

    streamnames = kwargs['tracts']

    merged_files_queue = []

    os.chdir(output_path)

    for method, file_list in streamnames.iteritems():
        print '== Method %s' % method
        for tfile in file_list:
            tbasename = tfile.split('.')[0]
            merged_file_basename = '_'.join([tbasename,'merged'])
            merged_tdb = merged_file_basename+'.tdb'
            if os.path.isfile(merged_tdb) and overwrite:
                print 'remove old %s' % (merged_tdb)
                os.remove(merged_tdb)

            # for idx,[subj,grp] in self.config.subject_df.iterrows():
            for subject in self.config.subjects:
                subj = subject.name
                grp = subject.group
                subj = subj.strip()
                trk_file = '_'.join([subj,tbasename+'.vtp'])
                print self.config.processed_path, subject.processed_path
                if subject.processed_path != self.config.processed_path:
                    trk_file = os.path.join(subject.processed_path,'tractography', trk_file)
                cmd = 'fascicle add -d %s -i %s --group %s' % (merged_tdb, trk_file, grp)
                exec_cmd(cmd)

            merged_model_file = merged_file_basename+'.vtp'
            # if os.path.isfile(merged_model_file):
            #     print '> remove old %s' % (merged_model_file) 
            #     os.remove(merged_model_file)
            cmd = 'fascicle expvtk -d %s --merged -o %s' % (merged_tdb,merged_model_file)
            exec_cmd(cmd)













