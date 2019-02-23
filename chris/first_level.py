#! /usr/bin/env python3
# Time-stamp: <2019-02-23 18:40:46 christophe@pallier.org>

import glob
import os
import os.path as op
import pickle
import re

#import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
import nistats
import nilearn
from nilearn.image import high_variance_confounds, mean_img
from nilearn.plotting import plot_epi, plot_glass_brain, plot_stat_map
#from nistats.design_matrix import make_first_level_design_matrix
from nistats.first_level_model import FirstLevelModel
from nistats.reporting import plot_design_matrix


DATADIR = os.getenv('DATADIR')  # location of fMRI data (one folder per subject)
if DATADIR is None:
    DATADIR = '/home/carlo.reverberi/bcn/data/'


#CACHE=None
CACHE='/nas/carlo.reverberi/scratch'
#CACHE='/media/cp983411/FAT-32G/scratch'

def get_sub_run_from_fname(runname):

    """ Returns the subject and run numbers from a BIDS bold filename. """
    info_re = re.compile('f(..)r(..)')
    return info_re.findall(runname)[0]


def get_rp_params(runname):
    """ Returns the nx6 matrix of motion parameters associated to a BIDS bold filename. """
    rp_name = re.sub('.nii', '.txt', re.sub('/uf', '/rp_f', runname))
    return pd.DataFrame(np.loadtxt(rp_name))


def read_events(fname):
    """ Parses the results file to extract onsets of events """
    allsubj = pd.read_csv(fname, header=None,
                          names=['subject', 'block', 'trial', 'time',
                                 'phase', 'condition', 'itlen', 'jitterlen',
                                 'isquestion', 'movieid'], sep=',')

    allsubj['duration'] = np.hstack([allsubj.time[1:], [0]]) - allsubj.time
    allsubj2 = allsubj.loc[allsubj.phase != 'Steady1']
    allsubj3 = allsubj2.loc[allsubj2.phase != 'Steady2']
    allsubj4 = allsubj3.loc[allsubj3.phase != 'Occlusion']
    #allsubj5 = allsubj4.loc[allsubj4.phase!= 'GreyScreen']
    allsubj5 = allsubj4.loc[allsubj4.phase != 'VideoOnset']
    allsubj6 = allsubj5.loc[allsubj5.phase != 'ForcePause']
    allsubj7 = allsubj6.loc[allsubj6.phase != 'EndForcePause']
    allsubj8 = allsubj7.loc[allsubj6.phase != 'PostMovie']
    allsubj9 = allsubj8.copy()
    allsubj9['condphase'] = allsubj9.condition + '_' + allsubj9.phase
    allsubj9.to_csv('allsubj9.csv')

    return allsubj9


def get_contrasts():
    Infobject = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    LoIgrass  = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
    NoIgrass  = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]);
    NoIobject = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]);

    emptyscreen = np.tile(np.array([1, 0, 0, 0]), (1, 4));
    objout      = np.tile(np.array([0, 1, 0, 0]), (1, 4));
    scooping    = np.tile(np.array([0, 0, 1, 0]), (1, 4));
    response    = np.tile(np.array([0, 0, 0, 1]), (1, 4));

    return {
        'emptyscreen': emptyscreen,
        'response': response,
        'scooping': scooping,
        'objout': objout,
        'scooping:(Infobject+LoIgrass).vs.(NoIgrass+NoIobject)': (Infobject + LoIgrass - (NoIgrass + NoIobject)) * scooping,
        'scooping:(Infobject+LoIgrass).vs.(NoIgrass + NoIobject)': (Infobject + LoIgrass - (NoIgrass + NoIobject)) * scooping,
        'objout:Infobject.vs.LoIgrass':  (Infobject - LoIgrass) * objout,
        'objout:LoIgrass.vs.NoIobject': (LoIgrass - NoIobject) * objout,
        'objout:Infobject.vs.NoIobject': (Infobject - NoIobject) * objout,
        'objout:Infobject.vs.NoIgrass': (Infobject - NoIgrass) * objout,
        'objout:LoIgrass.vs.NoIgrass': (LoIgrass - NoIgrass) * objout,
        'objout:NoIobject.vs.NoIgrass': (NoIobject - NoIgrass) * objout
    }


def do_model(slabel, runs, funcfiles, rpfiles, timings, outputdir):
    fmri_model_path = op.join(outputdir, 'glm2.pkl')

    if op.isfile(fmri_model_path):
        print(f'Loading already existing model {fmri_model_path}')
        fmri_model = pickle.load(open(fmri_model_path, 'rb'))
    else:
        fmri_model = do_fmri_model(slabel, runs, funcfiles, rpfiles, timings, outputdir)
        with open(fmri_model_path, 'wb') as f:
            pickle.dump(fmri_model, f)
        print(f'Model saved to {fmri_model_path}')

    return fmri_model


def do_fmri_model(slabel, runs, funcfiles, rpfiles, timings, outputdir):
    tr = 1.810  # repetition time
    #n_scans = nibabel.load(r).shape[3]
    #frame_times = np.arange(n_scans) * tr

    fmri_glm = FirstLevelModel(t_r=1.810,
                               hrf_model='spm',
                               #noise_model='ar1',
                               noise_model='ols',
                               mask='mask_ICV.nii',
                               drift_model='polynomial',
                               drift_order=3,
                               smoothing_fwhm=None,
                               period_cut=128,
                               verbose=0,
                               #minimize_memory=True,
                               memory=CACHE)

    events_list = []
    confounds_list = []

    for run, img, rp  in zip(runs, funcfiles, rpfiles):
        print(f'Adding run {run} : {img}')

        onsets = timings.time[timings.block == run] / 1000.0
        conds = timings.condphase[timings.block == run]
        durations = timings.duration[timings.block == run]
        durations[timings.phase != 'GreyScreen'] = 0.0
        events = pd.DataFrame({'trial_type': conds, 'onset': onsets,
                               'duration': durations})
        events.to_csv('example.csv')
        events_list.append(events)
        if len(events.trial_type.unique()) != 16:
            print("Did not find 16 types of events:")
            print(events.trial_type.unique())

        if not op.isfile(rp):
                print(f"Could not find {rp}")

        motions = pd.DataFrame(np.loadtxt(rp))
        confounds_list.append(motions)
        #hv_confounds = pd.DataFrame(high_variance_confounds(img, percentile=1))
        #confounds_list.append(pd.concat([motions, hv_confounds]))

    print("Fitting...")
    fmri_glm.fit(funcfiles, events=events_list, confounds=confounds_list)

    # save design matrices
    for ix, dmtx in  enumerate(fmri_glm.design_matrices_):
        dmtx.to_csv(outputdir + f'design_matrix{runs[ix]}.csv')

    print("  done")

    return fmri_glm


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    return value


def do_contrasts(fmri_model, contrasts, outputdir):
    for name, value in contrasts.items():
        print(f'Computing contrast "{name}"')
    
        value = np.hstack([value[0], np.zeros(10)])
        z_map = fmri_model.compute_contrast(value,
                                            output_type='z_score')
        nib.save(z_map, op.join(outputdir, slugify(f'{name}_zmap.nii.gz')))
        con_map = fmri_model.compute_contrast(value,
                                              output_type='effect_size')
        nib.save(con_map, op.join(outputdir, slugify(f'{name}_con.nii.gz')))
        #display = nilearn.plotting.plot_stat_map(z_map, display_mode='z',
        #                                         threshold=3.0, title=name)
    #nilearn.plotting.show()

def do_subject(subjdir):
    subject_label = op.basename(subjdir)
    print(f'Subject {subject_label}')

    subjtimings = timings.loc[timings.subject == subject_label]
    assert len(subjtimings) > 0

    # get list of functional images and realignement parameters 
    pattern = op.join(subjdir, 'nifti', 'swauf??r0[1-5].nii')
    func_files = sorted(glob.glob(pattern))
    rp_files = [re.sub('.nii', '.txt',
                       re.sub('/swauf', '/rp_f', r)) for r in func_files]

    if len(func_files) == 0 or len(rp_files) == 0:
        print(f"Skipping {subject_label}: could not find {pattern} or the associated rp*.txt files")
    else:
        outputdir = op.join(subjdir, MODELNAME)
        outputdir = re.sub('data', 'results', outputdir)
        outputdir = re.sub('results/sub', 'results/chris/sub', outputdir)

        if not op.isdir(outputdir):
            os.makedirs(outputdir)

        runs = [int(get_sub_run_from_fname(f)[1]) for f in func_files]
        fmri_model = do_model(subject_label,
                              runs,
                              func_files,
                              rp_files,
                              subjtimings,
                              outputdir)

        do_contrasts(fmri_model, get_contrasts(), outputdir)


if __name__ == '__main__':
    timings = read_events('events_timings.csv')

    MODELNAME='glm3'

    #for subjdir in sorted(glob.glob(op.join(DATADIR, 'sub*')))[1:]:  # skip sub01
    #    do_subject(subjdir)

    Parallel(n_jobs=5)(delayed(do_subject)(subjdir) for subjdir in sorted(glob.glob(op.join(DATADIR, 'sub*')))[1:])
