#! /usr/bin/env python

import os
import os.path as op
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm
import pandas as pd
import nibabel as nib
from nilearn import plotting
from nistats.second_level_model import SecondLevelModel
from joblib import Parallel, delayed


ROOTDIR = os.getenv('ROOTDIR')
if ROOTDIR is None:
    ROOTDIR = '/home/carlo.reverberi/bcn/results/chris'


def do_contrast(con, ROOTDIR, MODEL):
    """ Input: con is the name of the contrast """
    confile = f'{con}_con.nii.gz'
    spmfile = f'{con}_zmap.nii.gz'
    cmaps = glob.glob(op.join(ROOTDIR, 'sub*', MODEL, confile))
    smaps = glob.glob(op.join(ROOTDIR, 'sub*', MODEL, spmfile))
    cmaps.sort()
    smaps.sort()

    fig, axes = plt.subplots(nrows=5, ncols=4)
    for cidx, tmap in enumerate(smaps):
        plotting.plot_glass_brain(tmap, colorbar=True, threshold=3.1,
                                  title=f'{cidx:02d}',
                                  axes=axes[int(cidx / 4), int(cidx % 4)],
                                  plot_abs=False, display_mode='z')
    fig.suptitle(f'contrast {con}')
    #pdf.savefig(fig)

    second_level_input = cmaps
    design_matrix = pd.DataFrame([1] * len(second_level_input),
                                 columns=['intercept'])

    second_level_model = SecondLevelModel(smoothing_fwhm=8.0)
    second_level_model = second_level_model.fit(second_level_input,
                                                design_matrix=design_matrix)

    z_map = second_level_model.compute_contrast(output_type='z_score')
    nib.save(z_map, f'group_{con}.nii.gz')
    p_val = 0.001
    p001_unc = norm.isf(p_val)
    display = plotting.plot_glass_brain(
        z_map, threshold=p001_unc, colorbar=True, display_mode='lzry',
        plot_abs=False,
        title=f'group contrasts {con} (unc p<0.001)')
    display.savefig(f'group_{con}.png')
    #pdf.savefig()
    display.close()



if __name__ == '__main__':
    MODEL = 'glm3'
    outputdir = op.join(ROOTDIR, 'group', MODEL)
    pdf = PdfPages(op.join(ROOTDIR, f'{MODEL}.pdf'))

    contrasts = ['emptyscreen',
                 'response',
                 'scooping',
                 'objout',
                 'scooping:(Infobject+LoIgrass).vs.(NoIgrass+NoIobject)',
                 'scooping:(Infobject+LoIgrass).vs.(NoIgrass + NoIobject)',
                 'objout:Infobject.vs.LoIgrass',
                 'objout:LoIgrass.vs.NoIobject',
                 'objout:Infobject.vs.NoIobject',
                 'objout:Infobject.vs.NoIgrass',
                 'objout:LoIgrass.vs.NoIgrass',
                 'objout:NoIobject.vs.NoIgrass'
    ]

    # for con in contrasts:
    #     do_contrast(con, ROOTDIR, MODEL, pdf)

    Parallel(n_jobs=4)(delayed(do_contrast)(con, ROOTDIR, MODEL) for con in contrasts)

    pdf.close()
