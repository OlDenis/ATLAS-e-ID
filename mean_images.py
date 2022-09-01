# IMPORT PACKAGES AND FUNCTIONS
import tensorflow      as tf
import numpy           as np
import os, sys, h5py, pickle
from   argparse  import ArgumentParser
from   tabulate  import tabulate
from   itertools import accumulate
from   utils     import get_dataset, validation, make_sample, merge_samples, sample_composition
from   utils     import compo_matrix, get_sample_weights, get_class_weight, gen_weights, sample_weights
from   utils     import cross_valid, valid_results, sample_histograms, Batch_Generator
from   utils     import feature_removal, feature_ranking, fit_scaler, apply_scaler, fit_t_scaler, apply_t_scaler
from   models    import callback, create_model
#os.system('nvidia-modprobe -u -c=0') # for atlas15


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--n_train'        , default =  1e6,  type = float )
parser.add_argument( '--n_eval'         , default =    0,  type = float )
parser.add_argument( '--n_valid'        , default =  1e6,  type = float )
parser.add_argument( '--batch_size'     , default =  5e3,  type = float )
parser.add_argument( '--n_epochs'       , default =  100,  type = int   )
parser.add_argument( '--n_classes'      , default =    2,  type = int   )
parser.add_argument( '--n_tracks'       , default =    5,  type = int   )
parser.add_argument( '--bkg_ratio'      , default =    2,  type = float )
parser.add_argument( '--n_folds'        , default =    1,  type = int   )
parser.add_argument( '--n_gpus'         , default =    4,  type = int   )
parser.add_argument( '--verbose'        , default =    1,  type = int   )
parser.add_argument( '--patience'       , default =   10,  type = int   )
parser.add_argument( '--sbatch_var'     , default =    0,  type = int   )
parser.add_argument( '--node_dir'       , default = ''                  )
parser.add_argument( '--host_name'      , default = 'lps'               )
parser.add_argument( '--l2'             , default = 1e-7,  type = float )
parser.add_argument( '--dropout'        , default =  0.1,  type = float )
parser.add_argument( '--FCN_neurons'    , default = [200,200], type = int, nargs='+')
parser.add_argument( '--weight_type'    , default = 'none'              )
parser.add_argument( '--train_cuts'     , default = ''                  )
parser.add_argument( '--valid_cuts'     , default = ''                  )
parser.add_argument( '--NN_type'        , default = 'CNN'               )
parser.add_argument( '--images'         , default = 'ON'                )
parser.add_argument( '--scalars'        , default = 'ON'                )
parser.add_argument( '--scaling'        , default = 'ON'                )
parser.add_argument( '--t_scaling'      , default = 'OFF'               )
parser.add_argument( '--plotting'       , default = 'OFF'               )
parser.add_argument( '--generator'      , default = 'OFF'               )
parser.add_argument( '--sep_bkg'        , default = 'OFF'               )
parser.add_argument( '--metrics'        , default = 'val_accuracy'      )
parser.add_argument( '--eta_region'     , default = '0.0-2.5'           )
parser.add_argument( '--pt_region'      , default = '5-200'           )
parser.add_argument( '--output_dir'     , default = 'outputs'           )
parser.add_argument( '--model_in'       , default = ''                  )
parser.add_argument( '--model_out'      , default = 'model.h5'          )
parser.add_argument( '--scaler_in'      , default = ''                  )
parser.add_argument( '--scaler_out'     , default = 'scaler.pkl'        )
parser.add_argument( '--t_scaler_in'    , default = ''                  )
parser.add_argument( '--t_scaler_out'   , default = 't_scaler.pkl'      )
parser.add_argument( '--results_in'     , default = ''                  )
parser.add_argument( '--results_out'    , default = ''                  )
parser.add_argument( '--runDiffPlots'   , default = 0, type = int       )
parser.add_argument( '--feature_removal', default = 'OFF'               )
parser.add_argument( '--correlations'   , default = 'OFF'               )
args = parser.parse_args()

def mean_images(sample, labels, scalars, scaler_file, output_dir,dataset_name, pt_region, eta_region):
    if eta_region == '0.0-1.37':
        layers  = [ 'em_barrel_Lr0',   'em_barrel_Lr1',   'em_barrel_Lr2',   'em_barrel_Lr3',
                    'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3']
    if eta_region == '1.6-2.5':
        layers = ['tile_gap_Lr1','em_endcap_Lr0',   'em_endcap_Lr1',   'em_endcap_Lr2',   'em_endcap_Lr3',
                  'lar_endcap_Lr0',  'lar_endcap_Lr1',  'lar_endcap_Lr2',  'lar_endcap_Lr3']
    suffix = "_{}_{}GeV_{}".format(dataset_name, pt_region, eta_region,)
    cal_images(sample, labels, layers, output_dir, mode='mean', soft=True, suffix=suffix)

def cal_images(sample, labels, layers, output_dir, mode='random', scale='free', soft=True, suffix=''):
    import multiprocessing as mp
    def get_image(sample, labels, e_class, key, mode, image_dict):
        start_time = time.time()
        if mode == 'random':
            for counter in np.arange(10000):
                image = abs(sample[key][np.random.choice(np.where(labels==e_class)[0])])
                if np.max(image) !=0: break
        if mode == 'mean': image = np.mean(sample[key][labels==e_class], axis=0)
        if mode == 'std' : image = np.std (sample[key][labels==e_class], axis=0)
        print('plotting layer '+format(key,length+'s')+' for class '+str(e_class), end='', flush=True)
        print(' (', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
        image_dict[(e_class,key)] = image
    layers    = [layer for layer in layers if layer in sample.keys()]
    n_classes = max(labels)+1; length = str(max(len(n) for n in layers))
    manager   =  mp.Manager(); image_dict = manager.dict()
    processes = [mp.Process(target=get_image, args=(sample, labels, e_class, key, mode, image_dict))
                 for e_class in np.arange(n_classes) for key in layers]
    print('PLOTTING CALORIMETER IMAGES (mode='+mode+', scale='+str(scale)+')')
    for job in processes: job.start()
    for job in processes: job.join()
    file_name = '{}/cal_images{}.png'.format(output_dir, suffix)
    print('SAVING IMAGES TO:', file_name, '\n')
    fig = plt.figure(figsize=(7,14)) if n_classes == 2 else plt.figure(figsize=(18,14))
    for e_class in np.arange(n_classes):
        if scale == 'class': vmax = max([np.max(image_dict[(e_class,key)]) for key in layers])
        for key in layers:
            image_dict[(e_class,key)] -= min(0,np.min(image_dict[(e_class,key)]))
            #image_dict[(e_class,key)] = abs(image_dict[(e_class,key)])
            if scale == 'layer':
                vmax = max([np.max(image_dict[(e_class,key)]) for e_class in np.arange(n_classes)])
            if scale == 'free':
                vmax = np.max(image_dict[(e_class,key)])
            print("image dict type: ",type(image_dict[(e_class,key)]))
            plot_image(100*image_dict[(e_class,key)], n_classes, e_class, layers, key, 100*vmax, soft)
    wspace = -0.1 if n_classes == 2 else 0.2
    #fig.subplots_adjust(left=0.05, top=0.95, bottom=0.05, right=0.95, hspace=0.6, wspace=wspace)
    fig.savefig(file_name)

def plot_image(image, n_classes, e_class, layers, key, vmax, soft=True):
    class_dict = {0:'iso electron',  1:'charge flip' , 2:'photon conversion', 3:'b/c hadron',
                  4:'light flavor ($\gamma$/e$^\pm$)', 5:'light flavor (hadron)'}
    layer_dict = {'em_barrel_Lr0'     :'presampler'            , 'em_barrel_Lr1'  :'EM cal $1^{st}$ layer' ,
                  'em_barrel_Lr1_fine':'EM cal $1^{st}$ layer' , 'em_barrel_Lr2'  :'EM cal $2^{nd}$ layer' ,
                  'em_barrel_Lr3'     :'EM cal $3^{rd}$ layer' , 'tile_barrel_Lr1':'had cal $1^{st}$ layer',
                  'tile_barrel_Lr2'   :'had cal $2^{nd}$ layer', 'tile_barrel_Lr3':'had cal $3^{rd}$ layer'}
    if n_classes == 2: class_dict[1] = 'background'
    e_layer  = layers.index(key)
    n_layers = len(layers)
    plot_idx = n_classes*e_layer + e_class+1
    #plt.subplot(n_layers, n_classes, plot_idx)
    #title   = class_dict[e_class]+'\n('+layer_dict[key]+')'
    #title   = layer_dict[key]+'\n('+class_dict[e_class]+')'
    title   = class_dict[e_class]+'\n('+str(key)+')'
    limits  = [-0.13499031, 0.1349903, -0.088, 0.088]
    x_label = '$\phi$'                             if e_layer == n_layers-1 else ''
    x_ticks = [limits[0],-0.05,0.05,limits[1]]     if e_layer == n_layers-1 else []
    y_label = '$\eta$'                             if e_class == 0          else ''
    y_ticks = [limits[2],-0.05,0.0,0.05,limits[3]] if e_class == 0          else []
    plt.title(title,fontweight='normal', fontsize=12)
    plt.xlabel(x_label,fontsize=15); plt.xticks(x_ticks)
    plt.ylabel(y_label,fontsize=15); plt.yticks(y_ticks)
    plt.imshow(np.float32(image), cmap='Reds', interpolation='bilinear' if soft else None,
               extent=limits, vmax=1 if np.max(image)==0 else vmax) #norm=colors.LogNorm(1e-3,vmax))
    plt.colorbar(pad=0.02)

# VERIFYING ARGUMENTS
for key in ['n_train', 'n_eval', 'n_valid', 'batch_size']: vars(args)[key] = int(vars(args)[key])
if args.weight_type not in ['bkg_ratio', 'flattening', 'match2s', 'match2b', 'match2class', 'match2max', 'none']:
    print('\nweight_type', args.weight_type, 'not recognized --> resetting it to none\n')
    args.weight_type = 'none'
if '.h5' not in args.model_in and args.n_epochs < 1 and args.n_folds==1:
    print('\nERROR: weights file required with n_epochs < 1 --> aborting\n'); sys.exit()


# CNN PARAMETERS
CNN = {(56,11):{'maps':[100,100], 'kernels':[ (3,5) , (3,5) ], 'pools':[ (4,1) , (2,1) ]},
        (7,11):{'maps':[100,100], 'kernels':[ (3,5) , (3,5) ], 'pools':[ (1,1) , (1,1) ]},
        #(7,11):{'maps':[100,100], 'kernels':[(3,5,3),(3,5,3)], 'pools':[(1,1,1),(1,1,1)]},
      'tracks':{'maps':[200,200], 'kernels':[ (1,1) , (1,1) ], 'pools':[ (1,1) , (1,1) ]}}


# TRAINING VARIABLES
scalars = ['p_Eratio', 'p_Reta'   , 'p_Rhad'     , 'p_Rhad1' , 'p_Rphi'   , 'p_deltaPhiRescaled2'         ,
           'p_ndof'  , 'p_dPOverP', 'p_deltaEta1', 'p_f1'    , 'p_f3'     , 'p_sct_weight_charge'         ,
           'p_weta2' , 'p_d0'     , 'p_d0Sig'    , 'p_qd0Sig', 'p_nTracks', 'p_numberOfSCTHits'           ,
           'p_eta'   , 'p_et_calo', 'p_EptRatio' , 'p_EoverP', 'p_wtots1' , 'p_numberOfPixelHits'         ,
           'p_TRTPID', 'p_numberOfInnermostPixelHits'                                                     ]
images  = [ 'em_barrel_Lr0',   'em_barrel_Lr1',   'em_barrel_Lr2',   'em_barrel_Lr3', 'em_barrel_Lr1_fine',
                                'tile_gap_Lr1',
            'em_endcap_Lr0',   'em_endcap_Lr1',   'em_endcap_Lr2',   'em_endcap_Lr3', 'em_endcap_Lr1_fine',
           'lar_endcap_Lr0',  'lar_endcap_Lr1',  'lar_endcap_Lr2',  'lar_endcap_Lr3',
                             'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3', 'tracks'            ]
others  = ['mcChannelNumber', 'eventNumber', 'p_TruthType', 'p_iffTruth'   , 'p_TruthOrigin', 'p_LHValue' ,
           'p_LHTight'      , 'p_LHMedium' , 'p_LHLoose'  , 'p_ECIDSResult', 'p_eta'        , 'p_et_calo' ,
           'p_vertexIndex'  , 'p_charge'   , 'p_firstEgMotherTruthType'    , 'p_firstEgMotherTruthOrigin' ,
           'correctedAverageMu', 'p_firstEgMotherPdgId'                                                   ]


# DATASET AND TRAINING DICTIONARY
data_files = get_dataset(args.host_name, args.node_dir, args.eta_region)
keys       = set().union(*[h5py.File(data_file,'r').keys() for data_file in data_files])
images     = [key for key in images  if key in keys or key=='tracks']
scalars    = [key for key in scalars if key in keys or key=='tracks']
others     = [key for key in others  if key in keys]
if args.scalars != 'ON': scalars=[]
if args.images  != 'ON': images =[]
if args.feature_removal == 'ON':
    groups = [('em_barrel_Lr1','em_barrel_Lr1_fine'), ('em_barrel_Lr0','em_barrel_Lr2','em_barrel_Lr3')]
    scalars, images, removed_feature = feature_removal(scalars, images, groups=[], index=args.sbatch_var)
    args.output_dir += '/'+removed_feature
if images == []: args.NN_type = 'FCN'
train_data = {'scalars':scalars, 'images':images}
input_data = {**train_data, 'others':others}


# SAMPLES SIZES AND APPLIED CUTS ON PHYSICS VARIABLES
sample_size  = sum([len(h5py.File(data_file,'r')['eventNumber']) for data_file in data_files])
args.n_train = [0, min(sample_size, args.n_train)]
args.n_valid = [args.n_train[1], min(args.n_train[1]+args.n_valid, sample_size)]
if args.n_valid[0] == args.n_valid[1]: args.n_valid = args.n_train
if args.n_eval != 0: args.n_eval = [args.n_valid[0], min(args.n_valid[1],args.n_valid[0]+args.n_eval)]
else               : args.n_eval = args.n_valid
#args.train_cuts = '(abs(sample["eta"]) > 0.8) & (abs(sample["eta"]) < 1.15)'
#args.valid_cuts = '(sample["p_et_calo"] > 4.5) & (sample["p_et_calo"] < 20)'
#args.train_cuts = '((sample["mcChannelNumber"]==361106) | (sample["mcChannelNumber"]==423300)) & (sample["pt"]>=15)'
#args.valid_cuts = '((sample["mcChannelNumber"]==361106) | (sample["mcChannelNumber"]==423300)) & (sample["pt"]>=15)'


# OBTAINING PERFORMANCE FROM EXISTING VALIDATION RESULTS
if os.path.isfile(args.output_dir+'/'+args.results_in) or os.path.islink(args.output_dir+'/'+args.results_in):
    if args.eta_region in ['0.0-1.3', '1.3-1.6', '1.6-2.5']:
        eta_1, eta_2 = args.eta_region.split('-')
        valid_cuts   = '(abs(sample["eta"]) >= '+str(eta_1)+') & (abs(sample["eta"]) <= '+str(eta_2)+')'
        if args.valid_cuts == '': args.valid_cuts  = valid_cuts
        else                    : args.valid_cuts  = valid_cuts + '& ('+args.valid_cuts+')'
    inputs = {'scalars':scalars, 'images':[], 'others':others}
    validation(args.output_dir, args.results_in, args.plotting, args.n_valid, data_files,
               inputs, args.valid_cuts, args.sep_bkg, args.runDiffPlots)
elif args.results_in != '': print('\nOption --results_in not matching any file --> aborting\n')
if   args.results_in != '': sys.exit()


# MODEL CREATION AND MULTI-GPU DISTRIBUTION
n_gpus = min(args.n_gpus, len(tf.config.experimental.list_physical_devices('GPU')))
train_batch_size = max(1,n_gpus) * args.batch_size
valid_batch_size = max(1,n_gpus) * max(args.batch_size, int(5e3))
sample = make_sample(data_files[0], [0,1], input_data, args.n_tracks, args.n_classes)[0]
model  = create_model(args.n_classes, sample, args.NN_type, args.FCN_neurons, CNN,
                      args.l2, args.dropout, train_data, n_gpus)


# ARGUMENTS AND VARIABLES SUMMARY
args.scaling = args.scaling == 'ON' and list(set(scalars)-{'tracks'}) != []
args.t_scaling = args.t_scaling == 'ON' and 'tracks' in scalars+images
if args.NN_type == 'CNN':
    print('\nCNN ARCHITECTURE:')
    for shape in [shape for shape in CNN if shape in [sample[key].shape[1:] for key in sample]]:
        print(format(str(shape),'>8s')+':', str(CNN[shape]))
print('\nPROGRAM ARGUMENTS:'); print(tabulate(vars(args).items(), tablefmt='psql'))
print('\nTRAINING VARIABLES:')
headers = [           key  for key in train_data if train_data[key]!=[]]
table   = [train_data[key] for key in train_data if train_data[key]!=[]]
length  = max([len(n) for n in table])
table   = list(map(list, zip(*[n+(length-len(n))*[''] for n in table])))
print(tabulate(table, headers=headers, tablefmt='psql')); print()
args.model_in   = args.output_dir+'/'+args.model_in;   args.model_out   = args.output_dir+'/'+args.model_out
args.scaler_in  = args.output_dir+'/'+args.scaler_in;  args.scaler_out  = args.output_dir+'/'+args.scaler_out
args.t_scaler_in = args.output_dir+'/'+args.t_scaler_in ; args.t_scaler_out = args.output_dir+'/'+args.t_scaler_out
args.results_in = args.output_dir+'/'+args.results_in; args.results_out = args.output_dir+'/'+args.results_out


# GENERATING VALIDATION SAMPLE AND LOADING PRE-TRAINED WEIGHTS
if os.path.isfile(args.model_in):
    print('Loading pre-trained weights from', args.model_in, '\n')
    model.load_weights(args.model_in)
if args.scaling and os.path.isfile(args.scaler_in):
    print('Loading quantile transform from', args.scaler_in, '\n')
    scaler = pickle.load(open(args.scaler_in, 'rb'))
else: scaler = None
if args.t_scaling and os.path.isfile(args.t_scaler_in):
    print('Loading tracks scaler from ', args.t_scaler_in, '\n')
    t_scaler = pickle.load(open(args.t_scaler_in, 'rb'))
else: t_scaler = None
print('LOADING', np.diff(args.n_valid)[0], 'VALIDATION SAMPLES')
inputs = {'scalars':scalars, 'images':images, 'others':others} if args.generator == 'ON' else input_data
#valid_sample, valid_labels, _ = merge_samples(data_files, args.n_valid, inputs, args.n_tracks, args.n_classes,
#           args.valid_cuts, None if args.generator=='ON' else scaler, None if args.generator=='ON' else t_scaler)
jf17_sample, jf17_labels, _ = merge_samples(data_files[1:], args.n_valid, inputs, args.n_tracks, args.n_classes,
           args.valid_cuts, None if args.generator=='ON' else scaler, None if args.generator=='ON' else t_scaler)
mean_images(jf17_sample, jf17_labels, scalars, scaler, args.output_dir, 'jf17', args.pt_region, args.eta_region)
data17_sample, data17_labels, _ = merge_samples(data_files[:1], args.n_valid, inputs, args.n_tracks, args.n_classes,
           args.valid_cuts, None if args.generator=='ON' else scaler, None if args.generator=='ON' else t_scaler)
print("data17_labels",data17_labels)
mean_images(data17_sample, data17_labels, scalars, scaler, args.output_dir,'data17', args.pt_region, args.eta_region)
