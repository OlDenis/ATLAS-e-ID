import numpy           as np
import multiprocessing as mp
import sys, os, h5py, pickle, time, itertools, warnings
from   functools         import partial
from   sklearn           import metrics
from   scipy.spatial     import distance
from   matplotlib        import pylab
from   matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, FixedLocator
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def valid_accuracy(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    return np.sum(y_pred==y_true)/len(y_true)


def LLH_rates(sample, y_true, ECIDS=False):
    LLH_tpr, LLH_fpr = [],[]
    for wp in ['p_LHTight', 'p_LHMedium', 'p_LHLoose']:
        y_class0 = sample[wp][y_true == 0]
        y_class1 = sample[wp][y_true != 0]
        LLH_tpr.append( np.sum(y_class0 == 0)/len(y_class0) )
        LLH_fpr.append( np.sum(y_class1 == 0)/len(y_class1) )
    if ECIDS:
        ECIDS_cut    = -0.337671
        ECIDS_class0 = sample['p_ECIDSResult'][y_true == 0]
        ECIDS_class1 = sample['p_ECIDSResult'][y_true != 0]
        for wp in ['p_LHTight', 'p_LHMedium', 'p_LHLoose']:
            y_class0 = sample[wp][y_true == 0]
            y_class1 = sample[wp][y_true != 0]
            LLH_tpr.append( np.sum( (y_class0 == 0) & (ECIDS_class0 >= ECIDS_cut) ) / len(y_class0) )
            LLH_fpr.append( np.sum( (y_class1 == 0) & (ECIDS_class1 >= ECIDS_cut) ) / len(y_class1) )
    return LLH_fpr, LLH_tpr


def plot_history(history, output_dir, key='accuracy'):
    if history == None or len(history.epoch) < 2: return
    plt.figure(figsize=(12,8))
    pylab.grid(True)
    val = plt.plot(np.array(history.epoch)+1, 100*np.array(history.history[key]), label='Training')
    plt.plot(np.array(history.epoch)+1, 100*np.array(history.history['val_'+key]), '--',
             color=val[0].get_color(), label='Testing')
    min_acc = np.floor(100*min( history.history[key]+history.history['val_'+key] ))
    max_acc = np.ceil (100*max( history.history[key]+history.history['val_'+key] ))
    plt.xlim([1, max(history.epoch)+1])
    plt.xticks( np.append(1, np.arange(5, max(history.epoch)+2, step=5)) )
    plt.xlabel('Epochs',fontsize=25)
    plt.ylim( max(70,min_acc),max_acc )
    plt.yticks( np.arange(max(80,min_acc), max_acc+1, step=1) )
    plt.ylabel(key.title()+' (%)',fontsize=25)
    plt.legend(loc='lower right', fontsize=20, numpoints=3)
    file_name = output_dir+'/'+'train_history.png'
    print('Saving training accuracy history to:', file_name, '\n'); plt.savefig(file_name)


def plot_heatmaps(sample, labels, output_dir):
    n_classes = max(labels)+1
    label_dict = {0:'Iso electron', 1:'Charge flip'  , 2:'Photon conversion'    , 3  :'Heavy flavor (b/c)',
                  4:'Light flavor (e$^\pm$/$\gamma$)', 5:'Light flavor (hadron)'}
    pt  =     sample['pt']  ;  pt_bins = np.arange(0,81,1)
    eta = abs(sample['eta']); eta_bins = np.arange(0,2.55,0.05)
    extent = [eta_bins[0], eta_bins[-1], pt_bins[0], pt_bins[-1]]
    fig = plt.figure(figsize=(20,10)); axes = plt.gca()
    for n in np.arange(n_classes):
        plt.subplot(2, 3, n+1)
        heatmap = np.histogram2d(eta[labels==n], pt[labels==n], bins=[eta_bins,pt_bins], density=False)[0]
        plt.imshow(heatmap.T, origin='lower', extent=extent, cmap='Blues', interpolation='bilinear', aspect="auto")
        plt.title(label_dict[n]+' ('+format(100*np.sum(labels==n)/len(labels),'.1f')+'%)', fontsize=25)
        if n//3 == 1: plt.xlabel('abs('+'$\eta$'+')', fontsize=25)
        if n %3 == 0: plt.ylabel('$p_t$ (GeV)', fontsize=25)
    fig.subplots_adjust(left=0.05, top=0.95, bottom=0.1, right=0.95, wspace=0.15, hspace=0.25)
    file_name = output_dir+'/'+'heatmap.png'
    print('Saving heatmap plots to:', file_name)
    plt.savefig(file_name); sys.exit()


def var_histogram(sample, labels, n_etypes, weights, bins, output_dir, prefix, var,
                  density=True, separate_norm=False, log=True):
    n_classes = max(labels)+1
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    if var == 'pt':
        variable = sample[var]
        tag=''; plt.xlabel(tag+'$p_t$ (GeV)', fontsize=25)
        if bins == None or len(bins[var]) <= 2:
            #bins = [0, 10, 20, 30, 40, 60, 80, 100, 130, 180, 250, 500]
            bins = np.arange(np.min(variable), 102)
        else:
            bins = bins[var]
        if log:
            separate_norm=False
            if n_classes == 2: plt.xlim(4.5, 5000); plt.ylim(1e-5,1e2); plt.xscale('log'); plt.yscale('log')
            else             : plt.xlim(4.5, 5000); plt.ylim(1e-5,1e1); plt.xscale('log'); plt.yscale('log')
            plt.xticks( [4.5,10,100,1000,5000], [4.5,'$10^1$','$10^2$','$10^3$',r'$5\times10^3$'] )
        else:
            separate_norm=True
            axes.xaxis.set_minor_locator(FixedLocator(bins))
            plt.xticks( np.append([4.5],np.arange(0, 101, 20)), [4.5,0,20,40,60,80,100] )
            #xticks = np.append([4.5], plt.xticks()[0])
            plt.xlim(0, 100); plt.ylim(0,25)
            axes.xaxis.set_minor_locator(FixedLocator(np.arange(0,101,2)))
            axes.yaxis.set_minor_locator(AutoMinorLocator(5))
    if var == 'eta':
        variable = abs(sample[var])
        tag=''; plt.xlabel('abs($\eta$)', fontsize=25)
        if bins == None or len(bins[var]) <= 2:
            #bins = [0, 0.1, 0.6, 0.8, 1.15, 1.37, 1.52, 1.81, 2.01, 2.37, 2.47]
            step = 0.05; bins = np.arange(0, 2.5+step, step)
        else:
            bins = bins[var]
        axes.xaxis.set_minor_locator(FixedLocator(bins))
        plt.xticks(np.arange(0, 2.6, 0.5))
        #plt.xticks(bins, [str(n) for n in bins]); plt.xticks(bins, [format(n,'.1f') for n in bins])
        pylab.xlim(0, 2.5)
        if False:
            separate_norm = False
            pylab.ylim(1e-1, 1e2)
            plt.yscale('log')
        else:
            separate_norm = True
            pylab.ylim(0, 90)
            plt.yticks(np.arange(0, 100, 10))
            axes.yaxis.set_minor_locator(AutoMinorLocator(5))
    #bins[-1] = max(bins[-1], max(variable)+1e-3)
    if n_etypes == 5:
        label_dict = {0:'Iso electron', 1:'Charge flip'  , 2:'Photon conversion'    , 3  :'Heavy flavor',
                      4:'Light flavor'}
    if n_etypes == 6:
        label_dict = {0:'Iso electron', 1:'Charge flip'  , 2:'Photon conversion'    , 3  :'Heavy flavor (b/c)',
                      4:'Light flavor (e$^\pm$/$\gamma$)', 5:'Light flavor (hadron)'}
    color_dict = {0:'tab:blue'    , 1:'tab:orange'   , 2:'tab:green'            , 3  :'tab:red'   ,
                  4:'tab:purple'                     , 5:'tab:brown'            }
    if n_classes == 2: label_dict[1] = 'Background'
    #if n_classes != 2 and var == 'eta': separate_norm = True
    if np.all(weights) == None: weights = np.ones(len(variable))
    h = np.zeros((len(bins)-1,n_classes))
    for n in np.arange(n_classes):
        class_values  = variable[labels==n]
        class_weights =  weights[labels==n]
        class_weights = 100*class_weights/(np.sum(class_weights) if separate_norm else len(variable))
        if density:
            indices        = np.searchsorted(bins, class_values, side='right')
            class_weights /= np.take(np.diff(bins), np.minimum(indices, len(bins)-1)-1)
        h[:,n] = pylab.hist(class_values, bins, histtype='step', weights=class_weights, color=color_dict[n],
                            label=label_dict[n]+' ('+format(100*len(class_values)/len(variable),'.1f')+'%)', lw=2)[0]
    plt.ylabel('Distribution density (%)' if density else 'Distribution (%)', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=14)
    loc, ncol = ('upper right',1) if var=='pt' else ('upper center',2)
    plt.legend(loc=loc if var=='pt' else 'upper center', fontsize=16 if n_classes==2 else 14,
               ncol=ncol, facecolor='ghostwhite', framealpha=1).set_zorder(10)
    file_name = output_dir+'/'+str(var)+'_'+prefix+'.png'
    print('Saving', prefix, 'sample', format(var,'3s'), 'distributions to:', file_name)
    plt.savefig(file_name)


def plot_distributions_DG(sample, y_true, y_prob, n_etypes, output_dir, separation=False, bkg='bkg'):
    if n_etypes == 5:
        label_dict = {0:'Iso electron', 1:'Charge flip'  , 2:'Photon conversion'    ,   3  :'Heavy flavor',
                      4:'Light flavor', 'bkg':'Background'}
    if n_etypes == 6:
        label_dict = {0:'Iso electron', 1:'Charge flip'  , 2:'Photon conversion'    ,   3  :'Heavy flavor',
                      4:'Light flavor (e$^\pm$/$\gamma$)', 5:'Light flavor (hadron)', 'bkg':'Background'}
        #label_dict = {0:'Electron & charge flip'         , 1:'Photon conversion'    ,   2  :'Heavy flavor',
        #              3:'Light flavor (e$^\pm$/$\gamma$)', 4:'Light flavor (hadron)', 'bkg':'Background'}
    color_dict = {0:'tab:blue'    , 1:'tab:orange'   , 2:'tab:green'            ,   3  :'tab:red'   ,
                  4:'tab:purple'                     , 5:'tab:brown'            , 'bkg':'tab:orange'}
    if separation:
        label_dict.pop('bkg')
    else:
        label_dict={0:'Iso electron', 1:label_dict[bkg]}
        color_dict={0:'tab:blue'    , 1:color_dict[bkg]}
    n_classes = len(label_dict)
    def logit(x, delta=1e-16):
        x = np.clip(np.float64(x), delta, 1-delta)
        return np.log10(x) - np.log10(1-x)
    def inverse_logit(x):
        return 1/(1+10**(-x))
    def print_JSD(P, Q, idx, color, text):
        plt.text(0.945, 1.01-3*idx/100, 'JSD$_{0,\!'+text+'}$:',
                 {'color':'black', 'fontsize':10}, va='center', ha='right', transform=axes.transAxes)
        plt.text(0.990, 1.01-3*idx/100, format(distance.jensenshannon(P, Q), '.3f'),
                 {'color':color  , 'fontsize':10}, va='center', ha='right', transform=axes.transAxes)
    def class_histo(y_true, y_prob, bins, colors):
        h = np.full((len(bins)-1,n_classes), 0.)
        from utils import make_labels
        class_labels = make_labels(sample, n_classes)
        for n in label_dict:
            class_probs   = y_prob[class_labels==n]
            class_weights = len(class_probs)*[100/len(y_true)] #len(class_probs)*[100/len(class_probs)]
            h[:,n] = pylab.hist(class_probs, bins=bins, label=label_dict[n], histtype='step',
                                weights=class_weights, log=True, color=colors[n], lw=2)[0]
        if n_classes == 2:
            colors = len(colors)*['black']
        if False:
            for n in set(label_dict)-set([0]):
                new_y_true = y_true[np.logical_or(y_true==0, class_labels==n)]
                new_y_prob = y_prob[np.logical_or(y_true==0, class_labels==n)]
                fpr, tpr, threshold = metrics.roc_curve(new_y_true, new_y_prob, pos_label=0)
                sig_ratio = np.sum(y_true==0)/len(new_y_true)
                max_index = np.argmax(sig_ratio*tpr + (1-fpr)*(1-sig_ratio))
                axes.axvline(threshold[max_index], ymin=0, ymax=1, ls='--', lw=1, color=colors[n])
        for n in set(label_dict)-set([0]):
            print_JSD(h[:,0], h[:,n], n, colors[n], str(n))
        if n_classes > 2:
            print_JSD(h[:,0], np.sum(h[:,1:], axis=1), n_classes, 'black', '\mathrm{bkg}')
    plt.figure(figsize=(12,16))
    plt.subplot(2, 1, 1); pylab.grid(True); axes = plt.gca()
    pylab.xlim(0,100); pylab.ylim(1e-5 if n_classes>2 else 1e-5, 1e2)
    plt.xticks(np.arange(0,101,step=10))
    #pylab.xlim(0,10); pylab.ylim(1e-2 if n_classes>2 else 1e-2, 1e2)
    #plt.xticks(np.arange(0,11,step=1))
    bin_step = 0.5; bins = np.arange(0, 100+bin_step, bin_step)
    class_histo(y_true, 100*y_prob, bins, color_dict)
    plt.xlabel('$p_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('Distribution (%)', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper center', fontsize=16 if n_classes==2 else 14, numpoints=3,
               ncol=2, facecolor='ghostwhite', framealpha=1).set_zorder(10)
    plt.subplot(2, 1, 2); pylab.grid(True); axes = plt.gca()
    x_min=-10; x_max=5; pylab.xlim(x_min, x_max); pylab.ylim(1e-4 if n_classes>2 else 1e-3, 1e1)
    pos  =                   [  10**float(n)      for n in np.arange(x_min,0)       ]
    pos += [0.5]           + [1-10**float(n)      for n in np.arange(-1,-x_max-1,-1)]
    lab  =                   ['$10^{'+str(n)+'}$' for n in np.arange(x_min+2,0)     ]
    lab += [1,10,50,90,99] + ['99.'+n*'9'         for n in np.arange(1,x_max-1)     ]
    #x_min=-10; x_max=-1; pylab.xlim(x_min, x_max); pylab.ylim(1e-2 if n_classes>2 else 1e-4, 1e2)
    #pos  =                   [  10**float(n)      for n in np.arange(x_min,0)       ]
    #lab  =                   ['$10^{'+str(n)+'}$' for n in np.arange(x_min+2,0)     ] + [1,10]
    #lab += ['0.50   '] + ['$1\!-\!10^{'+str(n)+'}$' for n in np.arange(-1,-x_max-1,-1)]
    plt.xticks(logit(np.array(pos)), lab, rotation=15)
    bin_step = 0.1; bins = np.arange(x_min-1, x_max+1, bin_step)
    y_prob = logit(y_prob)
    class_histo(y_true, y_prob, bins, color_dict)
    #plt.xlabel('$p_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.xlabel('$d$ (%)', fontsize=25)
    plt.ylabel('Distribution (%)', fontsize=25)
    location = 'upper left' if n_classes==2 else 'upper center'
    axes.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc=location, fontsize=16 if n_classes==2 else 14, numpoints=3,
               ncol=2, facecolor='ghostwhite', framealpha=1).set_zorder(10)
    plt.subplots_adjust(top=0.95, bottom=0.1, hspace=0.2)
    file_name = output_dir+'/distributions.png'
    print('Saving test sample distributions to:', file_name); plt.savefig(file_name)


def performance_ratio(sample, y_true, y_prob, bkg, output_dir, ECIDS=False, output_tag='CNN2LLH'):
    #eta_bins = [0, 0.1, 0.6, 0.8, 1.15, 1.37, 1.52, 1.81, 2.01, 2.37, 2.47]
    #pt_bins  = [4, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 150, 250]
    #eta_bins = [0, 0.6, 1.15, 1.37, 1.52, 2.01, 2.47]
    eta_bins = [0, 2.47]
    pt_bins  = [4.5, 10, 20, 30, 40, 60, 80, 5000]
    ratios = {wp:ratio_meshgrid(sample, y_true, y_prob, wp, output_dir, eta_bins, pt_bins)
              for wp in ['loose','medium','tight']}
    pickle_file = 'CNN2LLH_inclusive.pkl' if len(eta_bins)==2 else 'CNN2LLH.pkl'
    if output_tag == 'CNN2CNN':
        input_dir = '6c_180m_match2class0/scalars_only'
        input_dir = output_dir.split('outputs')[0]+'outputs'+'/'+input_dir
        input_ratios = pickle.load(open(input_dir+'/'+'class_0_vs_'+str(bkg)+'/'+pickle_file,'rb'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for wp in ratios:
                ratios[wp] = np.nan_to_num(ratios[wp]/input_ratios[wp], nan=0, posinf=0, neginf=0)
    if output_tag == 'CNN2LLH':
        pickle.dump(ratios, open(output_dir+'/'+pickle_file,'wb'))
    vmin = min([np.min(n[n!=0]) for n in ratios.values()])
    vmax = max([np.max(n[n!=0]) for n in ratios.values()])
    if len(eta_bins) == 2:
        file_name = output_dir+'/'+output_tag+'.png'
        inclusive_meshgrid(eta_bins, pt_bins, ratios, file_name, output_tag, vmin, vmax)
    else:
        for wp in ratios:
            file_name = output_dir+'/'+output_tag+'_'+wp+'.png'
            bin_meshgrid(eta_bins, pt_bins, ratios[wp], file_name, vmin, vmax)
def ratio_meshgrid(sample, y_true, y_prob, wp, output_dir, eta_bins, pt_bins, ECIDS=False):
    bin_tuples = list(itertools.product(zip(eta_bins[:-1], eta_bins[1:]), zip(pt_bins[:-1], pt_bins[1:])))
    manager   = mp.Manager(); return_dict = manager.dict()
    arguments = [(sample, y_true, y_prob, wp, bin, ECIDS, return_dict) for bin in bin_tuples]
    processes = [mp.Process(target=bkg_eff_ratio, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    return np.array([return_dict[bin] for bin in bin_tuples]).reshape(-1,len(pt_bins)-1).T
def bkg_eff_ratio(sample, y_true, y_prob, wp, bin, ECIDS, return_dict):
    def CNN_fpr(tpr, fpr, LLH_tpr):
        """ Getting CNN bkg_eff for a given LLH sig_eff """
        return fpr[np.where(tpr <= LLH_tpr)[0][-1]]
    wp = {'tight':0, 'medium':1, 'loose':2}[wp]
    eta, pt = sample['eta'], sample['pt']
    cut_list = [abs(eta)>=bin[0][0], abs(eta)<bin[0][1], pt>=bin[1][0], pt<bin[1][1]]
    cuts = np.logical_and.reduce(cut_list)
    data = {key:sample[key][cuts] for key in sample}
    y_true = y_true[cuts]
    y_prob = y_prob[cuts]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob, pos_label=0)
        LLH_fpr, LLH_tpr = LLH_rates(data, y_true, ECIDS)
        LLH_fpr, LLH_tpr = LLH_fpr[-3:], LLH_tpr[-3:]
        ratio = LLH_fpr[wp]/CNN_fpr(tpr, fpr, LLH_tpr[wp])
    return_dict[bin] = np.nan_to_num(ratio, nan=0, posinf=0, neginf=0)
def inclusive_meshgrid(eta_bins, pt_bins, ratios, file_name, tag, vmin=None, vmax=None, color='black'):
    eta = np.arange(0, len(eta_bins))
    pt  = np.arange(0, len( pt_bins))
    if vmin is None: vmin = np.min(ratios[ratios!=0])
    if vmax is None: vmax = np.max(ratios[ratios!=0])
    plt.figure(figsize=(6 if tag=='CNN2LLH' else 5.2,7.5))
    def make_plot(wp, ratios, idx=None):
        if idx is None: idx = list(ratios.keys()).index(wp)+1
        ratios = ratios[wp]
        plt.subplot(1,3,idx); ax = plt.gca()
        plt.pcolormesh(eta, pt, ratios, cmap="Blues", vmin=vmin, vmax=vmax)
        plt.xticks(np.arange(len(eta_bins)), eta_bins)
        plt.xlabel(wp.title(), fontsize=18)
        if idx == 1 and tag == 'CNN2LLH':
            plt.yticks(np.arange(len( pt_bins)), (pt_bins[:-1]+[r'$\infty$']) if pt_bins[-1]>=1e3 else pt_bins)
            plt.ylabel('$p_t$ (GeV)', fontsize=25)
        for x in range(len(eta_bins)-1):
            for y in range(len( pt_bins)-1):
                text = 'Ind' if ratios[y,x]==0 else format(ratios[y,x],'.1f')
                plt.text(x+0.5, y+0.5, text, {'color':color, 'fontsize':18}, ha='center', va='center')
        plt.grid(True, color="grey", lw=1, alpha=1)
        left_axis = True if (idx==1 and tag=='CNN2LLH') else False
        plt.tick_params(axis='both', which='major', labelbottom=False, bottom=False,
                        labelleft=left_axis, left=left_axis, labelsize=14,)
        if pt_bins[-1] >= 1e3 and idx ==1:
            font_sizes = (len(pt_bins)-1)*[14]+[25]
            for tick, size in zip(plt.yticks()[-1], font_sizes): tick.set_fontsize(size)
        if idx ==2:
            plt.title('CNN / LLH' if tag=='CNN2LLH' else 'CNN / FCN', fontsize=25)
    for wp in ratios: make_plot(wp, ratios)
    cbar = plt.colorbar(fraction=0.05, pad=0.06, aspect=100, shrink=1)
    ticks = [val for val in cbar.get_ticks() if min(abs(val-vmin),abs(val-vmax))>0.02*(vmax-vmin)
             and round(val,1)!=round(vmin,1) and round(val,1)!=round(vmax,1)]
    ticks = [vmin] + ticks + [vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([format(n,'.1f') for n in ticks])
    cbar.ax.tick_params(labelsize=12)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=color)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.06)
    print('Saving test sample ratio meshgrid to:', file_name); plt.savefig(file_name)
def bin_meshgrid(eta_bins, pt_bins, ratios, file_name, vmin=None, vmax=None, color='black'):
    eta = np.arange(0, len(eta_bins))
    pt  = np.arange(0, len( pt_bins))
    plt.figure(figsize=(11,7.5)); ax = plt.gca()
    if vmin is None: vmin = np.min(ratios[ratios!=0])
    if vmax is None: vmax = np.max(ratios[ratios!=0])
    plt.pcolormesh(eta, pt, ratios, cmap="Blues", vmin=vmin, vmax=vmax)
    plt.xticks(np.arange(len(eta_bins)), eta_bins)
    plt.yticks(np.arange(len( pt_bins)), (pt_bins[:-1]+[r'$\infty$']) if pt_bins[-1]>=1e3 else pt_bins)
    for x in range(len(eta_bins)-1):
        for y in range(len( pt_bins)-1):
            text = 'Ind' if ratios[y,x]==0 else format(ratios[y,x],'.1f')
            plt.text(x+0.5, y+0.5, text, {'color':color, 'fontsize':18}, ha='center', va='center')
    plt.grid(True, color="grey", lw=1, alpha=1)
    plt.tick_params(axis='both', which='major', labelsize=14)
    if pt_bins[-1] >= 1e3:
        font_sizes = (len(pt_bins)-1)*[14]+[25]
        for tick, size in zip(plt.yticks()[-1], font_sizes):
            tick.set_fontsize(size)
    plt.xlabel('abs('+'$\eta$'+')', fontsize=25)
    plt.ylabel('$p_t$ (GeV)', fontsize=25)
    cbar = plt.colorbar(fraction=0.04, pad=0.02)
    ticks = [val for val in cbar.get_ticks() if min(abs(val-vmin),abs(val-vmax))>0.02*(vmax-vmin)
             and round(val,1)!=round(vmin,1) and round(val,1)!=round(vmax,1)]
    ticks = [vmin] + ticks + [vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([format(n,'.1f') for n in ticks])
    cbar.ax.tick_params(labelsize=12)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=color)
    plt.tight_layout()
    print('Saving test sample ratio meshgrid to:', file_name); plt.savefig(file_name)


def plot_ROC_curves(sample, y_true, y_prob, output_dir, ROC_type, ECIDS,
                    ROC_values=None, combine_plots=False):
    LLH_fpr, LLH_tpr = LLH_rates(sample, y_true, ECIDS)
    if ROC_values != None: fpr, tpr = ROC_values[0]
    #if ROC_values != None:
    #    index = output_dir.split('_')[-1]
    #    index = ROC_values[0].shape[1]-1 if index == 'bkg' else int(index)
    #    fpr_full, tpr_full = ROC_values[0][:,index], ROC_values[0][:,0]
    #    fpr     , tpr      = ROC_values[1][:,index], ROC_values[1][:,0]
    else:
        #if ECIDS: y_prob = sample['p_ECIDSResult']
        #y_prob = y_prob + (sample['p_ECIDSResult']/2 +0.5)
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_prob, pos_label=0)
        pickle.dump({'fpr':fpr, 'tpr':tpr}, open(output_dir+'/'+'pos_rates.pkl','wb'), protocol=4)
    signal_ratio       = np.sum(y_true==0)/len(y_true)
    accuracy           = tpr*signal_ratio + (1-fpr)*(1-signal_ratio)
    best_tpr, best_fpr = tpr[np.argmax(accuracy)], fpr[np.argmax(accuracy)]
    colors  = ['red', 'blue', 'green', 'red', 'blue', 'green']
    labels  = ['tight'      , 'medium'      , 'loose'       ,
               'tight+ECIDS', 'medium+ECIDS', 'loose+ECIDS']
    markers = 3*['o'] + 3*['D']
    sig_eff, bkg_eff = '$\epsilon_{\operatorname{sig}}$', '$\epsilon_{\operatorname{bkg}}$'
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    if ROC_type == 1:
        pylab.grid(False)
        len_0 = np.sum(fpr==0)
        x_min = min(80, 10*np.floor(10*min(LLH_tpr)))
        if fpr[np.argwhere(tpr >= x_min/100)[0]] != 0:
            y_max = 10*np.ceil( 1/min(np.append(fpr[tpr >= x_min/100], min(LLH_fpr)))/10 )
            if y_max > 200: y_max = 100*(np.ceil(y_max/100))
        else: y_max = 1000*np.ceil(max(1/fpr[len_0:])/1000)
        plt.xlim([x_min, 100]); plt.ylim([1, y_max])
        #LLH_scores = [1/fpr[np.argwhere(tpr >= value)[0]][0] for value in LLH_tpr]
        #LLH_scores = [1/fpr[np.argmin(abs(tpr-value))] for value in LLH_tpr]
        def interpolation(tpr, fpr, value):
            try:
                idx1 = np.where(tpr>value)[0][np.argmin(tpr[tpr>value])]
                idx2 = np.where(tpr<value)[0][np.argmax(tpr[tpr<value])]
                M = (1/fpr[idx2]-1/fpr[idx1]) / (tpr[idx2]-tpr[idx1])
                return 1/fpr[idx1] + M*(value-tpr[idx1])
            except ValueError:
                return np.max(1/fpr)
        LLH_scores = [interpolation(tpr, fpr, value) for value in LLH_tpr]
        for n in np.arange(len(LLH_scores)):
            axes.axhline(LLH_scores[n], xmin=(LLH_tpr[n]-x_min/100)/(1-x_min/100), xmax=1,
            ls='--', linewidth=0.5, color='tab:gray', zorder=10)
            axes.axvline(100*LLH_tpr[n], ymin=abs(1/LLH_fpr[n]-1)/(plt.yticks()[0][-1]-1),
            ymax=abs(LLH_scores[n]-1)/(plt.yticks()[0][-1]-1), ls='--', linewidth=0.5, color='tab:gray', zorder=5)
            plt.text(100.2, LLH_scores[n], str(int(LLH_scores[n])),
                     {'color':colors[n], 'fontsize':11 if ECIDS else 12}, va="center", ha="left")
        axes.xaxis.set_major_locator(MultipleLocator(10))
        axes.xaxis.set_minor_locator(AutoMinorLocator(10))
        yticks = plt.yticks()[0]
        axes.yaxis.set_ticks( np.append([1], yticks[1:]) )
        axes.yaxis.set_minor_locator(FixedLocator(np.arange(yticks[1]/5, yticks[-1], yticks[1]/5 )))
        plt.xlabel(sig_eff+' (%)', fontsize=25)
        plt.ylabel('1/'+bkg_eff, fontsize=25)
        #label = '{$w_{\operatorname{bkg}}$} = {$f_{\operatorname{bkg}}$}'
        P, = plt.plot(100*tpr[len_0:], 1/fpr[len_0:], color='#1f77b4', lw=2, zorder=10)
        if combine_plots:
            def get_legends(n_zip, output_dir):
                file_path, ls = n_zip
                file_path += '/' + output_dir.split('/')[-1] + '/pos_rates.pkl'
                fpr, tpr = pickle.load(open(file_path, 'rb')).values()
                len_0 = np.sum(fpr==0)
                P, = plt.plot(100*tpr[len_0:], 1/fpr[len_0:], color='tab:gray', lw=2, ls=ls, zorder=10)
                return P
            file_paths = ['outputs/6c_180m_match2class0/scalars_only/bkg_optimal',
                          'outputs/2c_180m_match2class0/scalars+images']
            linestyles = ['--', ':']
            leg_labels = ['HLV+images (6-class)', 'HLV only (6-class)', 'HLV+images (2-class)']
            Ps = [P] + [get_legends(n_zip, output_dir) for n_zip in zip(file_paths,linestyles)]
            L = plt.legend(Ps, leg_labels, loc='upper left', bbox_to_anchor=(0,1), fontsize=14,
                           facecolor='ghostwhite', framealpha=1); L.set_zorder(10)
        if ROC_values != None:
            tag = 'combined bkg'
            extra_labels = {1:'{$w_{\operatorname{bkg}}$} for best AUC',
                            2:'{$w_{\operatorname{bkg}}$} for best TNR @ 70% $\epsilon_{\operatorname{sig}}$'}
            extra_colors = {1:'tab:orange', 2:'tab:green'   , 3:'tab:red'     , 4:'tab:brown'}
            for n in [1,2]:#range(1,len(ROC_values)):
                extra_fpr, extra_tpr = ROC_values[n]
                extra_len_0 = np.sum(extra_fpr==0)
                plt.plot(100*extra_tpr[extra_len_0:], 1/extra_fpr[extra_len_0:],
                         label=extra_labels[n], color=extra_colors[n], lw=2, zorder=10)
        #if ROC_values != None:
        #    plt.rcParams['agg.path.chunksize'] = 1000
        #    plt.scatter(100*tpr_full[len_0:], 1/fpr_full[len_0:], color='silver', marker='.')
        if best_fpr != 0:
            plt.scatter( 100*best_tpr, 1/best_fpr, s=40, marker='D', c='tab:blue',
                         label="{0:<15s} {1:>3.2f}%".format('Best accuracy:',100*max(accuracy)), zorder=10 )
        for n in np.arange(len(LLH_fpr)):
            plt.scatter( 100*LLH_tpr[n], 1/LLH_fpr[n], s=40, marker=markers[n], c=colors[n], zorder=10,
                         label='$\epsilon_{\operatorname{sig}}^{\operatorname{LH}}$'
                         +'='+format(100*LLH_tpr[n],'.1f') +'%' +r'$\rightarrow$'
                         +'$\epsilon_{\operatorname{bkg}}^{\operatorname{LH}}$/'
                         +'$\epsilon_{\operatorname{bkg}}^{\operatorname{NN}}$='
                         +format(LLH_fpr[n]*LLH_scores[n], '>.1f')
                         +' ('+labels[n]+')' )
        axes.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(loc='upper right', fontsize=13 if ECIDS else 14, numpoints=3,
                   facecolor='ghostwhite', framealpha=1).set_zorder(10)
        if combine_plots: plt.gca().add_artist(L)
    if ROC_type == 2:
        plt.xlim([0.6, 1]); plt.ylim([0.9, 1-1e-4])
        plt.xticks([0.6, 0.7, 0.8, 0.9, 1], [60, 70, 80, 90, 100])
        plt.yscale('logit')
        plt.yticks([0.9, 0.99, 0.999, 0.9999], [90, 99, 99.9, 99.99])
        axes.xaxis.set_minor_locator(AutoMinorLocator(10))
        axes.xaxis.set_minor_formatter(plt.NullFormatter())
        axes.yaxis.set_minor_formatter(plt.NullFormatter())
        plt.xlabel('Signal Efficiency '+sig_eff+' (%)', fontsize=25)
        plt.ylabel('Background Rejection $1\!-\!$'+bkg_eff+' (%)', fontsize=25)
        plt.text(0.8, 0.67, 'AUC: '+format(metrics.auc(fpr,tpr),'.4f'),
                 {'color':'black', 'fontsize':22},  va='center', ha='center', transform=axes.transAxes)
        val = plt.plot(tpr, (1-fpr), label='Signal vs Background', color='#1f77b4', lw=2)
        plt.scatter( best_tpr, (1-best_fpr), s=40, marker='o', c=val[0].get_color(),
                     label="{0:<16s} {1:>3.2f}%".format('Best accuracy:', 100*max(accuracy)) )
        for LLH in zip(LLH_tpr, LLH_fpr, colors, labels, markers):
            plt.scatter(LLH[0], 1-LLH[1], s=40, marker=LLH[4], c=LLH[2], label='('+format(100*LLH[0],'.1f')
                        +'%, '+format(100*(1-LLH[1]),'.1f')+')'+r'$\rightarrow$'+LLH[3])
        axes.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(loc='upper right', fontsize=15, numpoints=3)
    if ROC_type == 3:
        def make_plot(location):
            plt.xlabel('Signal probability threshold (%)', fontsize=25); plt.ylabel('(%)',fontsize=25)
            val_1 = plt.plot(threshold[1:],   tpr[1:], color='tab:blue'  , label='Signal Efficiency'   , lw=2)
            val_2 = plt.plot(threshold[1:], 1-fpr[1:], color='tab:orange', label='Background Rejection', lw=2)
            val_3 = plt.plot(threshold[1:], accuracy[1:], color='black'  , label='Accuracy', zorder=10 , lw=2)
            for LLH in zip(LLH_tpr, LLH_fpr):
                p1 = plt.scatter(threshold[np.argwhere(tpr>=LLH[0])[0]], LLH[0],
                                 s=40, marker='o', c=val_1[0].get_color())
                p2 = plt.scatter(threshold[np.argwhere(tpr>=LLH[0])[0]], 1-LLH[1],
                                 s=40, marker='o', c=val_2[0].get_color())
            l1 = plt.legend([p1, p2], ['LLH '+sig_eff, 'LLH $1\!-\!$'+bkg_eff], loc='lower left', fontsize=13)
            plt.scatter( best_threshold, max(accuracy), s=40, marker='o', c=val_3[0].get_color(),
                         label='{0:<10s} {1:>5.2f}%'.format('Best accuracy:',100*max(accuracy)), zorder=10 )
            plt.legend(loc=location, fontsize=15, numpoints=3); plt.gca().add_artist(l1)
        best_threshold = threshold[np.argmax(accuracy)]
        plt.figure(figsize=(12,16))
        plt.subplot(2, 1, 1); pylab.grid(True); axes = plt.gca()
        plt.xlim([0, 1]);   plt.xticks(np.arange(0,1.01,0.1)   , np.arange(0,101,10))
        plt.ylim([0.6, 1]); plt.yticks(np.arange(0.6,1.01,0.05), np.arange(60,101,5))
        make_plot('lower center')
        plt.subplot(2, 1, 2); pylab.grid(True); axes = plt.gca()
        x_min=-2; x_max=3; y_min=0.1; y_max=1-1e-4;
        pylab.ylim(y_min, y_max); pylab.xlim(10**x_min, 1-10**(-x_max))
        pos  =                       [  10**float(n)  for n in np.arange(x_min,0)           ]
        pos += [0.5]               + [1-10**float(n)  for n in np.arange(-1,-x_max-1,-1)    ]
        lab  =                       [ '0.'+n*'0'+'1' for n in np.arange(abs(x_min)-3,-1,-1)]
        lab += [1, 10, 50, 90, 99] + ['99.'+n*'9'     for n in np.arange(1,x_max-1)         ]
        plt.xscale('logit'); plt.xticks(pos, lab)
        plt.yscale('logit'); plt.yticks([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999], [10, 50, 90, 99, 99.9, 99.99])
        axes.tick_params(axis='both', which='major', labelsize=12)
        axes.xaxis.set_minor_formatter(plt.NullFormatter())
        axes.yaxis.set_minor_formatter(plt.NullFormatter())
        make_plot('upper center')
    if ROC_type == 4:
        best_tpr = tpr[np.argmax(accuracy)]
        plt.xlim([60, 100.0])
        plt.ylim([80, 100.0])
        plt.xticks(np.arange(60,101,step=5))
        plt.yticks(np.arange(80,101,step=5))
        plt.xlabel('Signal Efficiency (%)',fontsize=25)
        plt.ylabel('(%)',fontsize=25)
        plt.plot(100*tpr[1:], 100*(1-fpr[1:]), label='Background rejection', color='darkorange', lw=2)
        val = plt.plot(100*tpr[1:], 100*accuracy[1:], label='Accuracy', color='black', lw=2, zorder=10)
        plt.scatter( 100*best_tpr, 100*max(accuracy), s=40, marker='o', c=val[0].get_color(),
                     label="{0:<10s} {1:>5.2f}%".format('Best accuracy:',100*max(accuracy)), zorder=10 )
        plt.legend(loc='lower center', fontsize=15, numpoints=3)
    file_name = output_dir+'/ROC'+str(ROC_type)+'_curve.png'
    print('Saving test sample ROC'+str(ROC_type)+' curve to   :', file_name); plt.savefig(file_name)


def combine_ROC_curves(output_dir, cuts=''):
    import multiprocessing as mp, pickle
    from scipy.interpolate import make_interp_spline
    from utils import NN_weights
    def mp_roc(idx, output_dir, return_dict):
        result_file = output_dir+'/'+'results_'+str(idx)+'.pkl'
        sample, labels, probs = pickle.load(open(result_file, 'rb'))
        #cuts = (sample["p_et_calo"] >= 0) & (sample["p_et_calo"] <= 500)
        if cuts == '': cuts = len(labels)*[True]
        sample, labels, probs = {key:sample[key][cuts] for key in sample}, labels[cuts], probs[cuts]
        fpr, tpr, threshold = metrics.roc_curve(labels, probs[:,0], pos_label=0)
        LLH_fpr, LLH_tpr = LLH_rates(sample, labels)
        print('LOADING VALIDATION RESULTS FROM', result_file)
        return_dict[idx] = fpr, tpr, threshold, LLH_fpr, LLH_tpr
    manager  = mp.Manager(); return_dict = manager.dict()
    idx_list = [1, 2, 3, 4]
    names    = ['no weight', 'flattening', 'match2s', 'match2max']
    colors   = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown']
    lines    = ['-', '-', '-', '-', '--', '--']
    processes = [mp.Process(target=mp_roc, args=(idx, output_dir, return_dict)) for idx in idx_list]
    for job in processes: job.start()
    for job in processes: job.join()
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    file_name = output_dir+'/'+'ROC_curve.png'
    x_min = []; y_max = []
    for n in np.arange(len(idx_list)):
        fpr, tpr, threshold, LLH_fpr, LLH_tpr = return_dict[idx_list[n]]; len_0  = np.sum(fpr==0)
        x_min += [min(60, 10*np.floor(10*LLH_tpr[0]))]
        y_max += [100*np.ceil(max(1/fpr[np.argwhere(tpr >= x_min[-1]/100)[0]], 1/LLH_fpr[0])/100)]
        plt.plot(100*tpr[len_0:], 1/fpr[len_0:], color=colors[n], label=names[n], linestyle=lines[n], lw=2)
        #label = str(bkg_class) if bkg_class != 0 else 'others'
        #val   = plt.plot(100*tpr[len_0:], 1/fpr[len_0:], label='class 0 vs '+label, lw=2)
        #for LLH in zip(LLH_tpr, LLH_fpr): plt.scatter(100*LLH[0], 1/LLH[1], s=40, marker='o', c=val[0].get_color())
    plt.xlim([min(x_min), 100]); plt.ylim([1, 250])  #plt.ylim([1, max(y_max)])
    axes.xaxis.set_major_locator(MultipleLocator(10))
    axes.yaxis.set_ticks( np.append([1], np.arange(50,300,50)) )
    plt.xlabel('Signal Efficiency (%)',fontsize=25)
    plt.ylabel('1/(Background Efficiency)',fontsize=25); #plt.yscale("log")
    plt.legend(loc='upper right', fontsize=15, numpoints=3)
    plt.savefig(file_name); sys.exit()
    '''
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    file_name = output_dir+'/'+'ROC1_curve.png'
    for LLH_tpr in [0.7, 0.8, 0.9]:
        bkg_rej  = [1/return_dict[idx][0][np.argwhere(return_dict[idx][1] >= LLH_tpr)[0]][0]
                    for idx in idx_list]
        #bkg_rej /= np.mean(bkg_rej)
        #n_weights = [NN_weights((5,13), CNN_dict, [200, 200], 2) for idx in idx_list]
        #bkg_rej   = [1e5*bkg_rej[n]/n_weights[n] for n in np.arange(len(idx_list))]
        plt.scatter(idx_list, bkg_rej, s=40, marker='o')
        idx_array    = np.linspace(min(idx_list), max(idx_list), 1000)
        spline       = make_interp_spline(idx_list, bkg_rej, k=2)
        plt.plot(idx_array, spline(idx_array), label=format(100*LLH_tpr,'.0f')+'% sig. eff.', lw=2)
    plt.xlim([min(idx_list)-1, max(idx_list)+1])
    axes.xaxis.set_major_locator(MultipleLocator(1))
    plt.ylim([0, 1500])
    axes.yaxis.set_major_locator(MultipleLocator(100))
    plt.xlabel('Maximum Number of Tracks',fontsize=25)
    plt.ylabel('1/(Background Efficiency)',fontsize=25)
    plt.legend(loc='lower center', fontsize=15, numpoints=3)
    plt.savefig(file_name)
    '''


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
    fig = plt.figure(figsize=(7,14)) if n_classes == 2 else plt.figure(figsize=(4,14))
    for e_class in np.arange(n_classes):
        if scale == 'class': vmax = max([np.max(image_dict[(e_class,key)]) for key in layers])
        for key in layers:
            image_dict[(e_class,key)] -= min(0,np.min(image_dict[(e_class,key)]))
            #image_dict[(e_class,key)] = abs(image_dict[(e_class,key)])
            if scale == 'layer':
                vmax = max([np.max(image_dict[(e_class,key)]) for e_class in np.arange(n_classes)])
            if scale == 'free':
                vmax = np.max(image_dict[(e_class,key)])
            plot_image(100*image_dict[(e_class,key)], n_classes, e_class, layers, key, 100*vmax, soft)
    wspace = -0.1 if n_classes == 2 else 0.2
    fig.subplots_adjust(left=0.05, top=0.95, bottom=0.05, right=0.95, hspace=0.6, wspace=wspace)
    fig.savefig(file_name)


def plot_image(image, n_classes, e_class, layers, key, vmax, soft=True):
    class_dict = {0:'electrons',  1:'charge flip' , 2:'photon conversion', 3:'b/c hadron',
                  4:'light flavor ($\gamma$/e$^\pm$)', 5:'light flavor (hadron)'}
    layer_dict = {'em_barrel_Lr0'     :'presampler'            , 'em_barrel_Lr1'  :'EM cal $1^{st}$ layer' ,
                  'em_barrel_Lr1_fine':'EM cal $1^{st}$ layer' , 'em_barrel_Lr2'  :'EM cal $2^{nd}$ layer' ,
                  'em_barrel_Lr3'     :'EM cal $3^{rd}$ layer' , 'tile_barrel_Lr1':'had cal $1^{st}$ layer',
                  'tile_barrel_Lr2'   :'had cal $2^{nd}$ layer', 'tile_barrel_Lr3':'had cal $3^{rd}$ layer'}
    if n_classes == 2: class_dict[1] = 'background'
    e_layer  = layers.index(key)
    n_layers = len(layers)
    plot_idx = n_classes*e_layer + e_class+1
    plt.subplot(n_layers, n_classes, plot_idx)
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


def plot_vertex(sample):
    bins = np.arange(0,50,1)
    fig = plt.figure(figsize=(12,8))
    pylab.xlim(-0.5,10.5)
    plt.xticks (np.arange(0,11,1))
    pylab.ylim(0,100)
    plt.xlabel('Track vertex value', fontsize=25)
    plt.ylabel('Distribution (%)', fontsize=25)
    weights = len(sample)*[100/len(sample)]
    pylab.hist(sample, bins=bins, weights=weights, histtype='bar', align='left', rwidth=0.5, lw=2)
    file_name = 'outputs/tracks_vertex.png'
    print('Printing:', file_name)
    plt.savefig(file_name)


def plot_scalars(sample, sample_trans, variable):
    bins = np.arange(-1,1,0.01)
    fig = plt.figure(figsize=(18,8))
    plt.subplot(1,2,1)
    pylab.xlim(-1,1)
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Number of Entries')
    #pylab.hist(sample_trans[variable], bins=bins, histtype='step', density=True)
    pylab.hist(sample      [variable], bins=bins, histtype='step', density=False)
    plt.subplot(1,2,2)
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Number of Entries')
    pylab.hist(sample_trans[variable], bins=bins)
    file_name = 'outputs/plots/scalars/'+variable+'.png'
    print('Printing:', file_name)
    plt.savefig(file_name)


def plot_tracks(tracks, labels, variable):
    tracks_var = {'efrac':{'idx':0, 'mean_lim':( 0,      3), 'max_lim':(0,    2), 'diff_lim':(0,    1)},
                  'deta' :{'idx':1, 'mean_lim':( 0, 0.0005), 'max_lim':(0, 0.03), 'diff_lim':(0, 0.04)},
                  'dphi' :{'idx':2, 'mean_lim':( 0,  0.001), 'max_lim':(0,  0.1), 'diff_lim':(0, 0.05)},
                  'd0'   :{'idx':3, 'mean_lim':( 0,    0.2), 'max_lim':(0,  0.1), 'diff_lim':(0,  0.3)},
                  'z0'   :{'idx':4, 'mean_lim':( 0,    0.5), 'max_lim':(0,  0.3), 'diff_lim':(0,   10)}}
    classes    = np.arange(max(labels)+1)
    n_e        = np.arange(len(labels)  )
    n_tracks   = np.sum(abs(tracks), axis=2)
    n_tracks   = np.array([len(np.where(n_tracks[n,:]!=0)[0]) for n in n_e])
    var        = tracks[..., tracks_var[variable]['idx']]
    var_mean   = np.array([np.mean(    var[n,:n_tracks[n]])  if n_tracks[n]!=0 else None for n in n_e])
    var_max    = np.array([np.max (abs(var[n,:n_tracks[n]])) if n_tracks[n]!=0 else None for n in n_e])
    var_diff   = np.array([np.mean(np.diff(np.sort(var[n,:n_tracks[n]])))
                           if n_tracks[n]>=2 else None for n in n_e])
    var_diff   = np.array([(np.max(var[n,:n_tracks[n]]) - np.min(var[n,:n_tracks[n]]))/(n_tracks[n]-1)
                           if n_tracks[n]>=2 else None for n in n_e])
    var_mean   = [var_mean[np.logical_and(labels==n, var_mean!=None)] for n in classes]
    var_max    = [var_max [np.logical_and(labels==n, var_max !=None)] for n in classes]
    var_diff   = [var_diff[np.logical_and(labels==n, var_diff!=None)] for n in classes]
    n_tracks   = [n_tracks[labels==n                                ] for n in classes]
    trk_mean   = [np.mean(n_tracks[n])                                for n in classes]
    fig  = plt.figure(figsize=(18,7))
    xlim = (0, 15)
    bins = np.arange(xlim[0], xlim[1]+2, 1)
    for n in [1,2]:
        plt.subplot(1,2,n); axes = plt.gca()
        plt.xlim(xlim)
        plt.xlabel('Number of tracks'      , fontsize=20)
        plt.xticks( np.arange(xlim[0],xlim[1]+1,1) )
        plt.ylabel('Normalized entries (%)', fontsize=20)
        title = 'Track number distribution (' + str(len(classes)) + '-class)'
        if n == 1: title += '\n(individually normalized)'
        weights = [len(n_tracks[n]) for n in classes] if n==1 else len(classes)*[len(labels)]
        weights = [len(n_tracks[n])*[100/weights[n]] for n in classes]
        plt.title(title, fontsize=20)
        label  =  ['class '+str(n)+' (mean: '+format(trk_mean[n],'3.1f')+')' for n in classes]
        plt.hist([n_tracks[n] for n in classes][::-1], bins=bins, lw=2, align='left',
                 weights=weights[::-1], label=label[::-1], histtype='step')
        plt.text(0.99, 0.05, '(sample: '+str(len(n_e))+' e)', {'color': 'black', 'fontsize': 12},
                 ha='right', va= 'center', transform=axes.transAxes)
        plt.legend(loc='upper right', fontsize=13)
    file_name = 'outputs/plots/tracks_number.png'; print('Printing:', file_name)
    plt.savefig(file_name)
    fig     = plt.figure(figsize=(22,6)); n = 1
    metrics = {'mean':(var_mean, 'Average'), 'max':(var_max, 'Maximum absolute'),
               'diff':(var_diff, 'Average difference')}
    #metrics = {'mean':(var_mean, 'Average'), 'max':(var_mean, 'Average'),
    #           'diff':(var_mean, 'Average')}
    for metric in metrics:
        plt.subplot(1, 3, n); axes = plt.gca(); n+=1
        n_e    = sum([len(metrics[metric][0][n]) for n in classes])
        x1, x2 = tracks_var[variable][metric+'_lim']
        bins   = np.arange(0.9*x1, 1.1*x2, (x2-x1)/100)
        plt.xlim([x1, x2])
        plt.title (metrics[metric][1] + ' value of ' + str(variable) + '\'s', fontsize=20)
        plt.xlabel(metrics[metric][1] + ' value'                            , fontsize=20)
        plt.ylabel('Normalized entries (%)'                                 , fontsize=20)
        #weights = [len(metrics[metric][0][n])*[100/len(metrics[metric][0][n])] for n in classes]
        weights = [len(metrics[metric][0][n])*[100/n_e] for n in classes]
        plt.hist([metrics[metric][0][n] for n in classes][::-1], weights=weights[::-1], stacked=False,
                 histtype='step', label=['class '+str(n) for n in classes][::-1], bins=bins, lw=2)
        plt.text(0.01, 0.97, '(sample: '+str(n_e)+' e)', {'color': 'black', 'fontsize': 12},
                 ha='left', va= 'center', transform=axes.transAxes)
        plt.legend(loc='upper right', fontsize=13)
    file_name = 'outputs/plots/tracks_'+str(variable)+'.png'; print('Printing:', file_name)
    plt.savefig(file_name)
