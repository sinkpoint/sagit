from gts import exec_cmd
import os

def per_subj_tract_to_template_space(self, subj, **kwargs):
    dry_run = False
    if 'dry_run' in kwargs:
        dry_run = kwargs['dry_run']
    print '============================',subj
    tract_path = self.config.tractography_path_full
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
    cmd = 'rm '+tdb_file
    exec_cmd(cmd, dryrun=dry_run)
    file_queue = []

    for method, file_list in streamnames.iteritems():
        print '== Method %s' % method
        for tfile in file_list:
            print '>',tfile
            trkwscalar = tfile.split('.')[0]+'.vtp'
        
            cmd = 'copyScalarsToTract.py -i %s -o %s -m %s_AD.nii.gz -n AD' % (tfile, trkwscalar, subj)
    
            exec_cmd(cmd, dryrun=dry_run)
            cmd = 'copyScalarsToTract.py -i %s -o %s -m %s_RD.nii.gz -n RD' % (trkwscalar, trkwscalar, subj)
    
            exec_cmd(cmd, dryrun=dry_run)
            cmd = 'copyScalarsToTract.py -i %s -o %s -m %s_FA.nii.gz -n FA' % (trkwscalar, trkwscalar, subj)
    
            exec_cmd(cmd, dryrun=dry_run)
            cmd = 'trkmanage.py init -i %s -d %s' % (trkwscalar, tdb_file)
    
            exec_cmd(cmd, dryrun=dry_run)

            file_queue.append(trkwscalar)

    # dump points to csv
    points_before = 'trk_points_dws.csv'
    points_after = 'trk_points_tps.csv'

    cmd = 'trkmanage.py expcsv -d %s -o %s' % (tdb_file, points_before)
    exec_cmd(cmd, dryrun=dry_run)

    # transform points to template space
    cmd = 'antsApplyTransformsToPoints -d 3 -i %s -o %s %s' % (points_before, points_after, tparams)
    exec_cmd(cmd, dryrun=dry_run)

    mapping_name = 'template_space'
    # reimport transformed points to tdb
    cmd = 'trkmanage.py tradd -d %s -i %s -n %s -p "%s"' % (tdb_file, points_after, mapping_name, tparams)
    exec_cmd(cmd, dryrun=dry_run)

    out_path = os.path.join(self.config.processed_path, 'tractography')
    if not os.path.isdir:
        os.mkdir(out_path)

    # export the transformed points into tract vtk
    for tfile in file_queue:
        out_name = os.path.join(out_path,'_'.join([subj,tfile]))
        cmd = 'trkmanage.py expvtk -d %s -t %s -m %s -o %s' % (tdb_file,tfile,mapping_name,out_name)

        exec_cmd(cmd, dryrun=dry_run)


def tracts_merge(self, **kwargs):
    dry_run = False
    if 'dry_run' in kwargs:
        dry_run = kwargs['dry_run']
    print '============= Tract Merge ==============='
    tract_path = self.config.tractography_path_full
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
            if os.path.isfile(merged_tdb):
                os.remove(merged_tdb)
            for subj in self.config.subjects:
                trk_file = '_'.join([subj,tbasename+'.vtp'])
                cmd = 'trkmanage.py init -d %s.tdb -i %s' % (merged_file_basename, trk_file)
                exec_cmd(cmd)
            cmd = 'trkmanage.py expvtk -d %s.tdb --merged -o %s.vtp' % (merged_file_basename,merged_file_basename)
            exec_cmd(cmd)













