# IMPORT PACKAGES AND FUNCTIONS
import numpy           as np
import multiprocessing as mp
import time, os, sys, h5py
from   argparse  import ArgumentParser
from   functools import partial
from   utils     import presample, merge_presamples, mix_datafiles, mix_presamples


# OPTIONS
parser = ArgumentParser()
parser.add_argument( '--n_e'        , default = None , type=float )
parser.add_argument( '--n_tasks'    , default = 10   , type=int   )
parser.add_argument( '--n_files'    , default = None , type=int   )
parser.add_argument( '--output_file', default = 'e-ID.h5'         )
parser.add_argument( '--sampling'   , default = 'ON'              )
parser.add_argument( '--merging'    , default = 'OFF'             )
parser.add_argument( '--mixing'     , default = 'OFF'             )
parser.add_argument( '--eta_region' , default = ''                )
parser.add_argument( '--indir'      , default = '/home/zp/denis/root2hdf5/outputs/discrepancy_test/MC/electron/myTag', type=str)
args = parser.parse_args()


# PRESAMPLES MIXING
if args.mixing == 'ON':
    #mix_datafiles() #run before mix_presamples
    mix_presamples(n_files=20, n_tasks=5, output_dir='/nvme1/atlas/godin/e-ID_data/0.0-2.5')
    sys.exit()


# DATASET
data_path  = args.indir
#if args.eta_region != '': data_path += '/'+args.eta_region
#else                    : sys.exit()
if not os.path.isdir(data_path+'/'+'output'): os.mkdir(data_path+'/'+'output')
output_dir = data_path+'/'+'output'
print(os.listdir(data_path))
data_files = [data_path+'/'+h5_file for h5_file in os.listdir(data_path) if '.h5' in h5_file]
data_files = sorted(data_files)[0:max(1,args.n_files) if args.n_files != None else len(data_files)]

print("Input path : ", data_path)
print("Input files: ", data_files)

# MERGING FILES / NO PRESAMPLING
if args.sampling == 'OFF' and args.merging == 'ON':
    print(); merge_presamples(output_dir, args.output_file)
    sys.exit()


# TEMPORARY
#data = h5py.File(data_files[0], 'r')
#for key in data: print( key, len(data[key]['eventNumber']) )
#for key, val in data['train'].items(): print(key, val.shape)


# ELECTRONS VARIABLES
images  =  ['em_barrel_Lr0'   , 'em_barrel_Lr1'   , 'em_barrel_Lr2'   , 'em_barrel_Lr3'                   ,
            'em_endcap_Lr0'   , 'em_endcap_Lr1'   , 'em_endcap_Lr2'   , 'em_endcap_Lr3'                   ,
            'lar_endcap_Lr0'  , 'lar_endcap_Lr1'  , 'lar_endcap_Lr2'  , 'lar_endcap_Lr3'                  ,
            'tile_barrel_Lr1' , 'tile_barrel_Lr2' , 'tile_barrel_Lr3' , 'tile_gap_Lr1'                    ]
tracks  =  ['tracks_pt'       , 'tracks_phi'      , 'tracks_eta'      , 'tracks_d0'        , 'tracks_z0'  ,
            'p_tracks_pt'     , 'p_tracks_phi'    , 'p_tracks_eta'    , 'p_tracks_d0'      , 'p_tracks_z0',
            'p_tracks_charge' , 'p_tracks_vertex' , 'p_tracks_chi2'   , 'p_tracks_ndof'    ,
            'p_tracks_pixhits', 'p_tracks_scthits', 'p_tracks_trthits', 'p_tracks_sigmad0'                ]
scalars =  ['p_truth_pt'      , 'p_truth_phi'     , 'p_truth_eta'     , 'p_truth_E'        , 'p_truth_e'  ,
            'p_et_calo'       , 'p_pt_track'      , 'p_EoverP'        , 'p_Eratio'         , 'p_phi'      ,
            'p_eta'           , 'p_e'             , 'p_Rhad'          , 'p_Rhad1'          , 'p_Rphi'     ,
            'p_Reta'          , 'p_d0'            , 'p_d0Sig'         , 'p_sigmad0'        , 'p_dPOverP'  ,
            'p_f1'            , 'p_f3'            , 'p_weta2'         , 'p_TRTPID'         , 'p_deltaEta1',
            'p_LHValue'       , 'p_chi2'          , 'p_ndof'          , 'p_ECIDSResult'    , 'p_wtots1'   ,
            'p_EptRatio'      , 'pileupWeight'    , 'correctedAverageMu', 'p_deltaPhiRescaled2'           ]
integers = ['p_truthType'     , 'p_TruthType'     , 'p_iffTruth'      , 'p_nTracks'        , 'p_charge'   ,
            'mcChannelNumber' , 'eventNumber'     , 'p_LHTight'       , 'p_LHMedium'       , 'p_LHLoose'  ,
            'p_LHLooseBL'     , 'p_LHVeryLoose'   , 'p_truthOrigin'   , 'p_TruthOrigin'    ,
            'p_vertexIndex'   , 'p_numberOfSCTHits', 'p_numberOfPixelHits', 'p_numberOfInnermostPixelHits',
            'p_firstEgMotherTruthType', 'p_firstEgMotherTruthOrigin', 'p_firstEgMotherPdgId'              ]
scalars += [ 'p_passWVeto', 'p_passZVeto', 'p_passPreselection', 'p_met', 'p_mTransW', "p_trigMatches_pTbin"]


# REMOVING TEMPORARY FILES (if any)
h5_files = [h5_file for h5_file in os.listdir(output_dir) if 'e-ID_' in h5_file]
for h5_file in h5_files: os.remove(output_dir+'/'+h5_file)

# MODIFYING FILE STRUCTURE
for h5_file in data_files:
    with h5py.File(h5_file,"a") as data:
        if 'train' not in data.keys():
            data.create_group('train')
            for key in data: data.move(key, 'train'+'/'+key)

# MODYFING FILE STRUCTURE
for h5_file in data_files:
    with h5py.File(h5_file,"a") as data:
        if 'train' not in data.keys():
            data.create_group('train')
            for key in data: data.move(key, 'train'+'/'+key)


# STARTING SAMPLING AND COLLECTING DATA
n_tasks = min(mp.cpu_count(), args.n_tasks)
max_e   = [len(h5py.File(h5_file,'r')[key]['eventNumber'])
           for h5_file in data_files for key in h5py.File(h5_file,'r')]
n_e = min(int(args.n_e), sum(max_e)) if args.n_e != None else sum(max_e)
n_e = np.int_(np.round(np.array(max_e)*min(1,n_e/sum(max_e)))) // n_tasks * n_tasks
print('\nSTARTING ELECTRONS COLLECTION (', '\b'+str(sum(n_e)), end=' ', flush=True)
print('electrons from', len(data_files),'files, using', n_tasks,'threads):')
pool = mp.Pool(n_tasks); sum_e = 0; index = 0
for h5_file in data_files:
    for file_key in h5py.File(h5_file,'r'):
        batch_size = n_e[index]//n_tasks; start_time = time.time()
        print('Collecting', format(str(n_e[index]),'>7s'), 'e from:', h5_file.split('/')[-1], end=' ')
        print(format('['+file_key+']','7s'), end=' ... ', flush=True)
        func_args = (h5_file, output_dir, batch_size, sum_e, images, tracks, scalars, integers, file_key)
        pool.map(partial(presample, *func_args), np.arange(n_tasks))
        sum_e += batch_size; index += 1
        print('(', '\b'+format(time.time() - start_time,'.1f'), '\b'+' s)')
pool.close(); pool.join(); print()


# MERGING FILES
if args.merging=='ON': merge_presamples(output_dir, args.output_file)
