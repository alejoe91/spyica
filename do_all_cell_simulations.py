
"""
This is a bad hack to be able to run models in series. For some reason I run into problems if I try to run different
cell models in a for loop if the kernel in not completely restarted
"""


import os, sys
import glob
from os.path import join
import ipdb

only_intracellular = True
model_folder = 'cell_models'
allen_folder = 'all_active_L5_models'
cells = 'all'
# cell_models = glob.glob(join(model_folder, 'L5*'))


if __name__ == '__main__':

    if len(sys.argv) == 1:
        print 'Use: \n-model (bbp - hay - allen), \n-rot (Norot - 3drot - Physrot) ' \
              '\n-intraonly (only intracellular simulation) ' \
              '\n-probe (SqMEA_-_um - NeuroSeeker-128 - Neuronexus-32 (-cut-30) ' \
              '- Neuropixel-384 - Neuropixel-24 - Tetrode)' \
              '\n-n (numbero of observations)'
    else:
        if '-model' in sys.argv:
            pos = sys.argv.index('-model')
            model = sys.argv[pos+1]  # file full path
        else:
            model = 'bbp'
        if '-rot' in sys.argv:
            pos = sys.argv.index('-rot')
            rotation = sys.argv[pos+1]  # Na - NaRep - PCA - PCA3d - 3d
        else:
            rotation = 'physrot'
        if '-intraonly' in sys.argv:
            only_intracellular = True
        else:
            only_intracellular = False
        if '-probe' in sys.argv:
            pos = sys.argv.index('-probe')
            probe = sys.argv[pos+1]  # Na - NaRep - PCA - PCA3d - 3d
        else:
            probe = 'SqMEA-10-15um'
        if '-n' in sys.argv:
            pos = sys.argv.index('-n')
            nobs = int(sys.argv[pos+1])  # Na - NaRep - PCA - PCA3d - 3d
        else:
            nobs = 1000

        if model == 'bbp':
            if cells == 'all':
                cell_models = [f for f in os.listdir(join(model_folder, model)) if 'mods' not in f]
            else:
                cell_models = ['L5_BP_bAC217_1'] #,
                           # 'L5_BTC_bAC217_1',
                           # 'L5_ChC_cACint209_1',
                           # 'L5_DBC_bAC217_1',
                           # 'L5_LBC_bAC217_1',
                           # 'L5_MC_bAC217_1',
                           # 'L5_NBC_bAC217_1',
                           # 'L5_NGC_bNAC219_1',
                           # 'L5_SBC_bNAC219_1',
                           # 'L5_STPC_cADpyr232_1',
                           # 'L5_TTPC1_cADpyr232_1',
                           # 'L5_TTPC2_cADpyr232_1',
                           # 'L5_UTPC_cADpyr232_1']
        elif model == 'hay':
            cell_models = ['L5bPCmodelsEH']
        elif model == 'almog':
            if cells == 'all':
                cell_models = [f for f in os.listdir(join(model_folder, model))]
            else:
                cell_models = ['A140612']#,
        elif model == 'allen':
            if cells == 'all':
                cell_models = [f for f in os.listdir(join(model_folder, model, allen_folder))
                               if f.startswith("neur")]
            else:
                cell_models = ['neuronal_model_501349486']#,
        elif model == 'hybrid':
            if cells == 'all':
                cell_models_allen = [f for f in os.listdir(join(model_folder, 'allen', allen_folder))
                                     if f.startswith("neur")]
                cell.models_bbp = [f for f in os.listdir(join('bbp', model))]
            else:
                cell_models = ['neuronal_model_501349486']#,

        if model == 'hybrid':
            for numb, cell_model in enumerate(cell_models_bbp):
                print cell_model, numb + 1, "/", len(cell_models)
                hyb_model = model + '-' + 'bbp'
                os.system("python hbp_cells.py %s %s %d %r %s %s %d" % (join(model_folder, hyb_model, cell_model), model, numb,
                                                                     only_intracellular, rotation, probe, nobs))
            for numb, cell_model in enumerate(cell_models_allen):
                print cell_model, numb + 1, "/", len(cell_models)
                hyb_model = model + '-' + 'allen'
                os.system(
                    "python hbp_cells.py %s %s %d %r %s %s %d" % (join(model_folder, hyb_model, allen_folder, cell_model),
                                                                  model, numb, only_intracellular, rotation, probe, nobs))
        else:
            for numb, cell_model in enumerate(cell_models):
                print cell_model, numb + 1, "/", len(cell_models)
                if model == 'allen':
                    os.system(
                        "python hbp_cells.py %s %s %d %r %s %s %d" % (join(model_folder, model, allen_folder, cell_model),
                                                                   model, numb, only_intracellular, rotation, probe, nobs))
                else:
                    os.system("python hbp_cells.py %s %s %d %r %s %s %d" % (join(model_folder, model, cell_model), model, numb,
                                                                         only_intracellular, rotation, probe, nobs))


