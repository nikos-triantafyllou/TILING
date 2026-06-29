
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch




# ██╗███╗   ███╗ █████╗  ██████╗ ███████╗███████╗
# ██║████╗ ████║██╔══██╗██╔════╝ ██╔════╝██╔════╝
# ██║██╔████╔██║███████║██║  ███╗█████╗  ███████╗
# ██║██║╚██╔╝██║██╔══██║██║   ██║██╔══╝  ╚════██║
# ██║██║ ╚═╝ ██║██║  ██║╚██████╔╝███████╗███████║
# ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝
                                               



def add_zoom_inset(ax, data_slice, cmap, extent, zoom_region,
                   inset_loc='upper right', zoom_percent=0.35,
                   box_color='white', mark=True, vmin=None, vmax=None, norm=None):

    x1, x2, y1, y2 = zoom_region

    # draw zoom box on the main axes
    rect = Rectangle((x1, y1), x2-x1, y2-y1,
                     linewidth=2, edgecolor=box_color, facecolor='none', linestyle='--')
    rect.set_in_layout(False)  # don't affect layout
    ax.add_patch(rect)

    # choose normalized position for the inset inside ax
    s = zoom_percent
    pad = 0.05
    loc_xy = {
        'upper right': (1 - s - pad, 1 - s - pad),
        'upper left' : (pad,        1 - s - pad),
        'lower left' : (pad,        pad),
        'lower right': (1 - s - pad, pad)
    }
    x0, y0 = loc_xy.get(inset_loc, loc_xy['upper right'])

    # create inset as a child of ax; keep it out of layout
    axins = ax.inset_axes([x0, y0, s, s], transform=ax.transAxes, zorder=5)
    axins.set_in_layout(False)          # ← crucial
    im = axins.imshow(data_slice, extent=extent, origin='lower',
                      cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks([])
    axins.set_yticks([])
    for sp in axins.spines.values():
        sp.set_edgecolor(box_color)

    # optional connectors (also keep them out of layout)
    # connectors: link both lower-left and upper-right corners
    if mark:
        corners_main = [(x1, y2), (x2, y1)]
        corners_inset = [(x1, y2), (x2, y1)]
        for (xm, ym), (xi, yi) in zip(corners_main, corners_inset):
            con = ConnectionPatch(xyA=(xm, ym), coordsA=ax.transData,
                                  xyB=(xi, yi), coordsB=axins.transData,
                                  color=box_color, lw=1)
            try:
                con.set_in_layout(False)
            except Exception:
                pass
            ax.add_artist(con)

    return axins, im




def add_colorbar_pdf(fig, im, ax, label=None):
    # Create a small axis just for the colorbar
    cax = inset_axes(ax,
                     width="3%",  # width relative to parent
                     height="100%", # same height
                     loc="center right",
                     borderpad=-3)  # adjust spacing
    cbar = fig.colorbar(im, cax=cax)
    if label:
        cbar.set_label(label)
    return cbar



from matplotlib.colors import LinearSegmentedColormap
primordia = LinearSegmentedColormap.from_list("primordia", [
    (0.0, "#000000"),   # black
    (0.25, "#1a085f"),  # deep indigo
    (0.5, "#5e3c99"),   # soft purple
    (0.75, "#b2abd2"),  # lavender
    (1.0, "#ffffff")    # white
])

starfall_bright = LinearSegmentedColormap.from_list("starfall_bright", [
    (0.0, "#000000"),
    (0.05, "#61094e"),
    (0.1, "#ba3f1d"),
    (0.15, "#ffc300"),
    (1.0, "#ffffff")
])


def get_eor_cmap(vmin=-150, vmax=30):
    from matplotlib import cm, colors
    name = f"EoR-{vmin}-{vmax}"
    negative_segments = 4
    positive_segments = 2
    neg_frac = abs(vmin) / (vmax - vmin)
    neg_seg_size = neg_frac / negative_segments
    pos_frac = abs(vmax) / (vmax - vmin)
    pos_seg_size = pos_frac / positive_segments
    eor_colour = colors.LinearSegmentedColormap.from_list(
        name,
        [
            (0, "white"),
            (neg_seg_size, "yellow"),
            (neg_seg_size * 2, "orange"),
            (neg_seg_size * 3, "maroon"),
            (neg_seg_size * 4, "black"),
            (neg_seg_size * 4 + pos_seg_size, "#1f77b4"),
            (1, "skyblue"),
        ],
    )
    print(neg_seg_size)
    try:
        plt.colormaps.register(cmap=eor_colour)
    except ValueError:
        plt.colormaps.unregister(name)
        plt.colormaps.register(cmap=eor_colour)

    return name

vmin_eor, vmax_eor = -120, 12.0
cmap_eor = get_eor_cmap(vmin_eor, vmax_eor)


def plot_all_slices(filename, pred_ics, true_ics, data, sample, added_unc, theta):
    
    zoom_region = (35, 65, 35, 65)
    z_slice_position = 100
    cmap=primordia
    vmin=-5
    vmax=5
    # cmap = 'BrBG'
    if theta.shape[1]>1:
        data2 = np.array(theta[0,1,:,:,:].cpu().detach())

    fig, axs = plt.subplots(2, 4, figsize=(20,8),  gridspec_kw={'hspace': 0.3, 'wspace': 0.0},)
    p1 = axs[0,0].imshow(true_ics[:,:,z_slice_position], extent=[0,300,0,300], origin='lower', cmap=cmap, vmin=vmin,vmax=vmax)
    
    # fig.colorbar(p1, ax=axs[0,0], label = r'$\rm \delta $')  
    # add_colorbar_pdf(fig, p1, axs[0,0], r'$\rm \delta$')
    divider = make_axes_locatable(axs[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(p1, cax=cax)
    cbar.set_label(r'$\rm \delta $')

    axs[0,0].set_title(label = 'True ICs')


    # pred, _ = model(theta)
    p2 = axs[0,1].imshow(pred_ics[:,:,z_slice_position], extent=[0,300,0,300], origin='lower',cmap=cmap, vmin=vmin,vmax=vmax)
    fig.colorbar(p2, ax=axs[0,1], label = r'$\rm \delta $')  
    
    axs[0,1].set_title(label = 'Mean of samples');


    p3 = axs[0,3].imshow(data[:,:,z_slice_position], extent=[0,300,0,300], origin='lower', cmap=cmap_eor, vmin=vmin_eor, vmax=vmax_eor)
    fig.colorbar(p3, ax=axs[0,3], label = r'$\rm \delta $')  
    axs[0,3].set_title(label = '21cm signal');


    p4 = axs[1,0].imshow(sample[:,:,z_slice_position], extent=[0,300,0,300], origin='lower',cmap=cmap, vmin=vmin,vmax=vmax)
    fig.colorbar(p4, ax=axs[1,0], label = r'$\rm \delta $')
    axs[1,0].set_title(label = 'Sample');



    p5 = axs[1,1].imshow(added_unc[:,:,z_slice_position], extent=[0,300,0,300], origin='lower',cmap=cmap, vmin=None,vmax=None)
    fig.colorbar(p5, ax=axs[1,1], label = r'$\rm \delta $')
    axs[1,1].set_title(label = r'$\sigma$ of samples');


    if theta.shape[1]>1:
        p6 = axs[1,3].imshow(data2[:,:,z_slice_position], extent=[0,300,0,300], origin='lower',cmap=starfall_bright, vmin=0, vmax=12)
        fig.colorbar(p6, ax=axs[1,3], label = r'$\rm num $')
        axs[1,3].set_title(label = 'Galaxy counts');
    else:
        fig.delaxes(axs[0,2])


    residual_MAP = true_ics[:,:,z_slice_position]-pred_ics[:,:,z_slice_position]
    p7 = axs[0,2].imshow(residual_MAP, extent=[0,300,0,300], origin='lower', cmap='BrBG', vmin=vmin,vmax=vmax)
    fig.colorbar(p7, ax=axs[0,2], label = r'$\rm \delta $')
    axs[0,2].set_title(label = 'True - MAP');

    residual_sample = true_ics[:,:,z_slice_position]-sample[:,:,z_slice_position]
    p8 = axs[1,2].imshow(residual_sample, extent=[0,300,0,300], origin='lower', cmap='BrBG', vmin=vmin,vmax=vmax)
    fig.colorbar(p8, ax=axs[1,2], label = r'$\rm \delta $')
    axs[1,2].set_title(label = 'True - Sample');


    axs[1,0].set_xlabel('x-axis [Mpc]')
    axs[1,1].set_xlabel('x-axis [Mpc]')
    axs[1,2].set_xlabel('x-axis [Mpc]')
    axs[1,3].set_xlabel('x-axis [Mpc]')

    axs[0,0].set_ylabel('y-axis [Mpc]')
    axs[1,0].set_ylabel('y-axis [Mpc]')

    
    add_zoom_inset(axs[0,0], true_ics[:, :, z_slice_position], cmap=cmap, extent=[0, 300, 0, 300], zoom_region=zoom_region, vmin=vmin, vmax=vmax)
    add_zoom_inset(axs[0,1], pred_ics[:, :, z_slice_position], cmap=cmap, extent=[0, 300, 0, 300], zoom_region=zoom_region,  vmin=vmin, vmax=vmax)
    add_zoom_inset(axs[0,3], data[:, :, z_slice_position], cmap=cmap_eor, extent=[0, 300, 0, 300], zoom_region=zoom_region, vmin=vmin_eor, vmax=vmax_eor)
    add_zoom_inset(axs[1,0], sample[:, :, z_slice_position], cmap=cmap, extent=[0, 300, 0, 300], zoom_region=zoom_region, vmin=vmin, vmax=vmax)
    add_zoom_inset(axs[1,1], added_unc[:, :, z_slice_position], cmap=cmap, extent=[0, 300, 0, 300], zoom_region=zoom_region, vmin=vmin, vmax=vmax)
    add_zoom_inset(axs[0,2], residual_MAP, cmap='BrBG', extent=[0, 300, 0, 300], zoom_region=zoom_region, vmin=vmin,vmax=vmax)
    add_zoom_inset(axs[1,2], residual_sample, cmap='BrBG', extent=[0, 300, 0, 300], zoom_region=zoom_region, vmin=vmin,vmax=vmax)
    if theta.shape[1]>1:
        add_zoom_inset(axs[1,3], data2[:, :, z_slice_position], cmap=starfall_bright, extent=[0, 300, 0, 300], zoom_region=zoom_region, vmin=0, vmax=12)

    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print('saved figure 1', flush = True)
    return dict(true_ics=true_ics, pred_ics=pred_ics, sample=sample, 
added_unc=added_unc, data=data, data2=data2 if theta.shape[1]>1 else None)





# ██████╗  ██████╗ ██╗    ██╗███████╗██████╗     ███████╗██████╗ ███████╗ ██████╗████████╗██████╗  █████╗ 
# ██╔══██╗██╔═══██╗██║    ██║██╔════╝██╔══██╗    ██╔════╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝██╔══██╗██╔══██╗
# ██████╔╝██║   ██║██║ █╗ ██║█████╗  ██████╔╝    ███████╗██████╔╝█████╗  ██║        ██║   ██████╔╝███████║
# ██╔═══╝ ██║   ██║██║███╗██║██╔══╝  ██╔══██╗    ╚════██║██╔═══╝ ██╔══╝  ██║        ██║   ██╔══██╗██╔══██║
# ██║     ╚██████╔╝╚███╔███╔╝███████╗██║  ██║    ███████║██║     ███████╗╚██████╗   ██║   ██║  ██║██║  ██║
# ╚═╝      ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝    ╚══════╝╚═╝     ╚══════╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝
                                                                                                        





def plot_all_power_spectra(filename, sample, pred_ics, true_ics, added_unc, BOX_SIDE=200, samples=None): # sample should be shape (n_samples, BOX_SIDE, BOX_SIDE, BOX_SIDE)
    # plt.rcParams.update({'font.size': 22})
    # plt.rcParams['lines.linewidth'] = plt.rcParamsDefault['lines.linewidth']
    # plt.rc('text', usetex=True)  # You need to install LaTeX on your system for this to work
    # plt.rc('font', family='serif',serif='Times')
    # plt.rcParams.update({
    # "font.size": 22,
    # "font.family": "serif",   # font family
    # "mathtext.fontset": "cm", # use Computer Modern (LaTeX-like)
    # "axes.unicode_minus": False
    # })
    # sample = samples[0] 



    figure, axs = plt.subplots(3, 1, figsize=(6,10), gridspec_kw={'hspace': 0.0, 'height_ratios': [2, 1, 1]}, sharex=True)
    import powerbox as pbox
    k_nyquist = np.pi*BOX_SIDE/300
    pk_true, k = pbox.get_power(true_ics, boxlength=300)

    # to plot the mean power spectrum of the samples and the std
    pk_samples = []
    pk_cross_samples = []
    for i in range(samples.shape[0]):
        pk_s, k = pbox.get_power(samples[i], boxlength=300)
        pk_samples.append(pk_s)
        pk_cs = pbox.get_power(deltax=true_ics, deltax2=samples[i], boxlength=300)[0]
        pk_cross_samples.append(pk_cs)
    pk_samples = np.array(pk_samples)
    pk_cross_samples = np.array(pk_cross_samples)
    pk_samples_mean = pk_samples.mean(axis=0)
    pk_samples_std = pk_samples.std(axis=0)
    T_samples = []
    r_samples = []
    for i in range(pk_samples.shape[0]):
        T_s = np.sqrt(pk_samples[i]/pk_true)
        T_samples.append(T_s)
        r_s = pk_cross_samples[i]/np.sqrt(pk_samples[i]*pk_true)
        r_samples.append(r_s)
    T_samples = np.array(T_samples)
    r_samples = np.array(r_samples)
    T_samples_mean = T_samples.mean(axis=0)
    T_samples_std = T_samples.std(axis=0)
    r_samples_mean = r_samples.mean(axis=0)
    r_samples_std = r_samples.std(axis=0)


    
    axs[0].plot(k, pk_true, color='black', label='Ground truth', linestyle='--')

    axs[0].plot(k, pk_samples_mean, color='purple', label='Mean of samples')
    axs[0].fill_between(k, pk_samples_mean - pk_samples_std, pk_samples_mean + pk_samples_std, color='purple', alpha=0.3)
    axs[0].fill_between(k, pk_samples_mean - 2*pk_samples_std, pk_samples_mean + 2*pk_samples_std, color='purple', alpha=0.2)

    pk_sample, k = pbox.get_power(sample, boxlength=300)
    axs[0].plot(k,pk_sample, color='purple', label='sample', linestyle='--')

    pk_pred, k = pbox.get_power(pred_ics, boxlength=300)
    axs[0].plot(k, pk_pred, color='teal', label='U-net')

    pk_unc, k = pbox.get_power(added_unc, boxlength=300)
    axs[0].plot(k, pk_unc, color='teal', label='Added uncertainty', linestyle='--')
    
    axs[0].fill_between(np.linspace(k_nyquist,6, 100), 1e-1, 1e6, label = r'$\rm >k_{Nyquist}$', color='black', alpha=0.3)

    

    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_ylim([1e-1,1e5])
    axs[0].set_xlim([0.025,5])
    axs[0].set_ylabel(r'P(k) [$\rm Mpc^3]$')
    axs[0].set_xlabel(r'k [$\rm  \; Mpc^{-1}$]')
    axs[0].legend()



    pk_cross_pred = pbox.get_power(deltax= true_ics, deltax2 = pred_ics, boxlength=300)[0]
    pk_cross_sample = pbox.get_power(deltax=true_ics, deltax2=sample, boxlength=300)[0]
    r_pred = pk_cross_pred/np.sqrt(pk_pred*pk_true)
    r_sample = pk_cross_sample/np.sqrt(pk_sample*pk_true)
    axs[1].axhline(1, color='black', linestyle='--')

    axs[1].plot(k, r_samples_mean, color='purple', label='Mean of samples')
    axs[1].fill_between(k, r_samples_mean - r_samples_std, r_samples_mean + r_samples_std, color='purple', alpha=0.3)
    axs[1].fill_between(k, r_samples_mean - 2*r_samples_std, r_samples_mean + 2*r_samples_std, color='purple', alpha=0.2)

    axs[1].plot(k, r_sample, color='purple', label='sample', linestyle='--')

    axs[1].plot(k, r_pred, color='teal', label='U-net')
    
    axs[1].fill_between(np.linspace(k_nyquist,6, 100), 0, 1.2, label = r'$\rm >k_{Nyquist}$', color='black', alpha=0.3)
    axs[1].axhspan(0.95, 1.05, color="teal", alpha=0.2, label = '±5%' ) 
    
    axs[1].set_xscale('log')
    axs[1].set_ylim([0,1.2])
    axs[1].set_xlim([0.025,5])
    axs[1].set_ylabel(r'CCC(k)')
    axs[1].set_xlabel(r'k [$\rm  \; Mpc^{-1}$]')
    # axs[1].legend()
    

    # Plot transfer function
    T_pred = np.sqrt(pk_pred/pk_true)
    T_sample = np.sqrt(pk_sample/pk_true)
    axs[2].axhline(1, color='black', linestyle='--')

    axs[2].plot(k, T_samples_mean, color='purple', label='Mean of samples')
    axs[2].fill_between(k, T_samples_mean - T_samples_std, T_samples_mean + T_samples_std, color='purple', alpha=0.3)
    axs[2].fill_between(k, T_samples_mean - 2*T_samples_std, T_samples_mean + 2*T_samples_std, color='purple', alpha=0.2)


    axs[2].plot(k, T_sample, color='purple', label='sample', linestyle='--')

    axs[2].plot(k, T_pred, color='teal', label='U-net')
    
    axs[2].fill_between(np.linspace(k_nyquist,6, 100), 0, 2, label = r'$\rm >k_{Nyquist}$', color='black', alpha=0.3)
    axs[2].axhspan(0.95, 1.05, color="teal", alpha=0.2, label = '±5%' ) 
    
    axs[2].set_xscale('log')
    axs[2].set_ylim([0.,2.0])
    axs[2].set_xlim([0.025,5])
    axs[2].set_ylabel(r'T(k)')
    axs[2].set_xlabel(r'k [$\rm  \; Mpc^{-1}$]')
    # plt.legend()

    for ax in axs:
            # Major ticks
        ax.tick_params(
            which="major",
            direction="in",
            top=True, right=True,
            length=6, width=1.2
        )
        # Minor ticks
        ax.tick_params(
            which="minor",
            direction="in",
            top=True, right=True,
            length=3, width=1.0
        )

    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print('saved figure all power spectra', flush = True)
    plt.close()
    return dict(k=k, pk_true=pk_true, pk_pred=pk_pred, pk_sample=pk_sample, 
    pk_samples_mean=pk_samples_mean, pk_samples_std=pk_samples_std, r_samples_mean=r_samples_mean, 
    r_samples_std=r_samples_std, T_samples_mean =T_samples_mean, T_samples_std=T_samples_std, T_pred=T_pred, T_sample=T_sample, r_pred=r_pred, r_sample=r_sample)







# ██████╗ ██╗      ███████╗██████╗ ███████╗ ██████╗████████╗██████╗  █████╗ 
# ██╔══██╗██║      ██╔════╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝██╔══██╗██╔══██╗
# ██████╔╝██║█████╗███████╗██████╔╝█████╗  ██║        ██║   ██████╔╝███████║
# ██╔══██╗██║╚════╝╚════██║██╔═══╝ ██╔══╝  ██║        ██║   ██╔══██╗██╔══██║
# ██████╔╝██║      ███████║██║     ███████╗╚██████╗   ██║   ██║  ██║██║  ██║
# ╚═════╝ ╚═╝      ╚══════╝╚═╝     ╚══════╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝


import Pk_library as PKL
from _my_funcs.ic_utils import calculate_bispectra_of_samples




def plot_mean_and_std(values, array_of_shape_samples_values, color='teal', label='mean', plot_mean=True, plot_std = True):
    mean = np.mean(array_of_shape_samples_values, axis=0)
    std = np.std(array_of_shape_samples_values, axis=0)
    if plot_mean:
        plt.plot(values, mean, label=label, color=color)
    if plot_std:
        plt.fill_between(values, mean - std, mean + std, alpha=0.5, color=color)
        plt.fill_between(values, mean - 2*std, mean + 2*std, alpha=0.25, color=color)
    # plt.legend()


def plot_reduced_bispectrum_vs_theta(filename, sample_boxes, k1, k2, BoxSize=300, true_ics=None): # exchnagnge black with true ics bispectrum relaization 
    fig, ax = plt.subplots(figsize=(8,6))
    theta, Q_samples = calculate_bispectra_of_samples(sample_boxes, k1, k2, BoxSize, equilateral=False)
    # plot mean and std of the samples
    plot_mean_and_std(theta, Q_samples, color='purple', label='MAP')
    # plot one of the samples
    plt.plot(theta, Q_samples[0,:], color='purple', label='sample', linestyle='--')
    # Add the gaussian sample prediction
    Q_gaussian = np.load('_save_space/_easy_access/Q_theta.npy')
    plot_mean_and_std(theta, Q_gaussian, color='black', plot_mean=False)

    # Add the true ics bispectrum realization
    theta, Q_true = calculate_bispectra_of_samples(true_ics, k1, k2, BoxSize, equilateral=False)
    plt.plot(theta, Q_true[0,:], color='black', label='Ground truth', linestyle='--')

    plt.xlabel(r'$\theta$ [rad]', fontsize=16)
    plt.ylabel(r'$Q(k_1,k_2,\theta)$', fontsize=16)
    plt.title(r'$k_1=$'+f'{k1}, '+r'$k_2=$'+f'{k2}', fontsize=16)
    # plt.ylim([-10,10])
    plt.axhline(0, color='black', label = 'Theoretical Gaussian')
    plt.xlim([0-0.1,np.pi+0.1])
    ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax.set_xticklabels(['0', r'$\pi/4$', '$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print('saved figure bispectrum', flush = True)
    return dict(theta=theta, Q_samples=Q_samples, Q_gaussian=Q_gaussian, Q_true=Q_true)

    theta = Q_vs_theta

def plot_reduced_bispectrum_vs_k(filename, sample_boxes, k1, k2, BoxSize=300, true_ics=None):  # exchnagnge black with true ics bispectrum relaization 
    fig, ax = plt.subplots(figsize=(8,6))
    k_arr, Q_samples = calculate_bispectra_of_samples(sample_boxes, k1, k2, BoxSize, equilateral=True)
    # plot mean and std of the samples
    plot_mean_and_std(k_arr, Q_samples, color='purple', label='MAP')
    # plot one of the samples
    plt.plot(k_arr, Q_samples[0,:], color='purple', label='sample', linestyle='--')
    # Add the gaussian sample prediction
    Q_k_gaussian = np.load('_save_space/_easy_access/Qeq_k_arr.npy')
    plot_mean_and_std(k_arr, Q_k_gaussian, color='black', plot_mean=False)

    # Add the true ics bispectrum realization
    k_arr, Q_true = calculate_bispectra_of_samples(true_ics, k1, k2, BoxSize, equilateral=True)
    plt.plot(k_arr, Q_true[0,:], color='black', label='Ground truth', linestyle='--')

    plt.xlabel(r'$\rm k [Mpc^{-1}]$', fontsize=16)
    plt.ylabel(r'$\rm Q(k_1=k, k_2=k, \theta=\pi /3)$', fontsize=16)
    plt.title(r'Equilateral', fontsize=16)
    # plt.ylim([-10,10])
    plt.axhline(0, color='black', label = 'Theoretical Gaussian')
    plt.xlim([0.9*k1, k2*1.1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xscale('log')
    plt.legend()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print('saved figure bispectrum', flush = True)
    return dict(k_arr=k_arr, Q_samples=Q_samples, Q_k_gaussian=Q_k_gaussian, Q_true=Q_true)




# ██████╗ ██████╗ ███████╗
# ██╔══██╗██╔══██╗██╔════╝
# ██████╔╝██║  ██║█████╗  
# ██╔═══╝ ██║  ██║██╔══╝  
# ██║     ██████╔╝██║     
# ╚═╝     ╚═════╝ ╚═╝     




def plot_pdf(filename, samples, true_ics):
    sample = samples[0]
    from scipy.stats import skew, kurtosis
    plt.figure(figsize=(8,6))
    # True 
    hist, bin_edges = np.histogram(true_ics.flatten(), 100, density=True);
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    plt.plot(bin_centers , hist, color='black', label='Ground truth')
    # Sample
    hist, _ = np.histogram(sample.flatten(), bin_edges, density=True);
    # bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    plt.plot(bin_centers , hist, color='purple', label='Sample', linestyle='--')
    
    # All samples
    hist_list = []
    skewness_list = []
    kyrtosis_list = []
    for i in range(len(samples)):
        hist, _ = np.histogram(samples[i].flatten(), bins=bin_edges, density=True);
        hist_list.append(hist)
        skewness_list.append(skew(samples[i].flatten()))
        kyrtosis_list.append(kurtosis(samples[i].flatten()))
    hist_array = np.array(hist_list)
    plot_mean_and_std(bin_centers, hist_array, color='purple', label='MAP')

    # plt.text(0.7, 0.8, f'skewness: {skew(sample.flatten()):.2f}', transform=plt.gca().transAxes, fontsize=12)
    # plt.text(0.7, 0.75, f'kurtosis: {kurtosis(sample.flatten()):.2f}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.9, f'skewness: {np.mean(skewness_list):.2f} ± {np.std(skewness_list):.2f}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.85, f'kurtosis: {np.mean(kyrtosis_list):.2f} ± {np.std(kyrtosis_list):.2f}', transform=plt.gca().transAxes, fontsize=12)
    plt.xlabel(r'$\rm \delta $')
    plt.ylabel(r'$\rm P(\delta)$')
    # plt.title('PDF of True vs Predicted ICs')
    plt.legend()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print('saved figure pdf', flush = True)
    return dict(bin_centers=bin_centers, hist_array=hist_array, skewness_list=skewness_list, kyrtosis_list=kyrtosis_list)


# ████████╗ █████╗ ██████╗ ██████╗ 
# ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗
#    ██║   ███████║██████╔╝██████╔╝
#    ██║   ██╔══██║██╔══██╗██╔═══╝ 
#    ██║   ██║  ██║██║  ██║██║     
#    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     
                               

from _my_funcs.coverage_tests import get_tarp_coverage
def plot_TARP(filename, supersamples, supertrue, bootstrap=True, sbc=False): # supersamples should have shape (num_samples, num_sims, num_dims) # supertrue has shape: (num_sims, num_dims)

    n_sims = supertrue.shape[0]

    ecp, alpha = get_tarp_coverage(supersamples, supertrue, references='random', metric='euclidean', norm = True, seed = 5, bootstrap=bootstrap, sbc=False)

    if bootstrap==True:
        std = ecp.std(axis=0)
        ecp = ecp.mean(axis=0)
        std=std[1:]
    else:
        std = None


    ecp = ecp[1:]
    alpha=alpha[1:]

    n_bins = alpha.shape[0]

    # Theoretical
    mean = n_sims/n_bins
    p = 1/n_bins
    sigma = np.sqrt(n_sims*p*(1-p))

    mean_norm = mean/n_sims
    sigma_norm = sigma/n_sims

    def cdf2pdf(cdf_array):
        pdf_array = np.zeros(cdf_array.shape)
        pdf_array[0] = cdf_array[0]
        for i in range(1,cdf_array.shape[0],1):
            pdf_array[i] = cdf_array[i]-cdf_array[i-1]
        return pdf_array

    sigmas_cdf = []
    for i in range(alpha.shape[0]):
        ii=i+1
        p = ii/n_bins
        sigma_i = np.sqrt(n_sims*p*(1-p))
        sigmas_cdf.append(sigma_i)
    sigmas_cdf = np.array(sigmas_cdf)/n_sims


    fig, axs = plt.subplots(3, 1, figsize=(4, 6),  gridspec_kw={ 'hspace': 0.0, 'wspace': 0.0, 'height_ratios': [1, 2,1]}, sharex=True)
    axs[0].set_ylabel("Coverage per bin")

    # 

    axs[0].axhline(mean, color='black', linestyle='--')

    # axs[0].fill_between(alpha, ecp_uniform - sigmas_cdf, ecp_uniform+sigmas_cdf, alpha=0.2)
    axs[0].axhspan(mean-sigma, mean+sigma, color="black", alpha=0.2,)
    axs[0].axhspan(mean-2*sigma, mean+2*sigma, color="black", alpha=0.2,)

    # axs[0].step(alpha, cdf2pdf(ecp*n_sims), where='pre') # unormalizes

    axs[0].step(alpha, cdf2pdf(ecp*n_sims), where='pre') # unormalizes


    if bootstrap==True:
        axs[0].fill_between(alpha, cdf2pdf(ecp*n_sims) - 1*cdf2pdf(std*n_sims), cdf2pdf(ecp*n_sims) + 1*cdf2pdf(std*n_sims), alpha=0.2, color='tab:blue',step='pre')
        axs[0].fill_between(alpha, cdf2pdf(ecp*n_sims) - 2*cdf2pdf(std*n_sims), cdf2pdf(ecp*n_sims) + 2*cdf2pdf(std*n_sims), alpha=0.2, color='tab:blue',step='pre')
    # axs[0].plot(alpha, cdf2pdf(ecp))  # normalized

    # axs[0].plot(alpha*n_sims, cdf2pdf(alpha*n_sims)) # essentially the mean

    #ADD 0 at first as a known theoretical result
    alpha_plot = np.concatenate(([0], alpha))
    ecp_plot   = np.concatenate(([0], ecp))
    sigmas_cdf_plot  = np.concatenate(([0], sigmas_cdf))
    if bootstrap==True:
        std_plot = np.concatenate(([0], std))


    axs[1].plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
    axs[1].fill_between(alpha_plot, alpha_plot - sigmas_cdf_plot, alpha_plot+sigmas_cdf_plot, alpha=0.2, color='black')
    axs[1].fill_between(alpha_plot, alpha_plot - 2*sigmas_cdf_plot, alpha_plot+2*sigmas_cdf_plot, alpha=0.2, color='black')
    # axs[1].plot(alpha_plot, [0, 1], ls='--', color='k', label = "Ideal case")
    axs[1].plot(alpha_plot, ecp_plot, label='TARP', color='tab:blue')
    if bootstrap==True:
        axs[1].fill_between(alpha_plot, ecp_plot - std_plot, ecp_plot + std_plot, alpha=0.2, color='tab:blue')
        axs[1].fill_between(alpha_plot, ecp_plot - 2*std_plot, ecp_plot + 2*std_plot, alpha=0.2, color='tab:blue')
        

    # axs[1].plot(alpha, ecp-alpha,  color='purple', label='SBC')
    axs[1].legend()
    axs[1].set_ylabel("ECDF")
    axs[1].set_xlabel("Credibility Level percentiles")



    axs[2].axhline(0, ls='--', color='k', label = "Ideal case")
    axs[2].fill_between(alpha_plot, - sigmas_cdf_plot, +sigmas_cdf_plot, alpha=0.2, color='black')
    axs[2].fill_between(alpha_plot, - 2*sigmas_cdf_plot, +2*sigmas_cdf_plot, alpha=0.2, color='black')

    axs[2].plot(alpha_plot, (ecp_plot-alpha_plot), label='TARP', color='tab:blue')
    if bootstrap==True:
        axs[2].fill_between(alpha_plot, ecp_plot-alpha_plot - std_plot, ecp_plot-alpha_plot + std_plot, alpha=0.2, color='tab:blue')
        axs[2].fill_between(alpha_plot, ecp_plot-alpha_plot - 2*std_plot, ecp_plot-alpha_plot + 2*std_plot, alpha=0.2, color='tab:blue')
        

    axs[2].set_ylabel("ECDF Difference")
    axs[2].set_xlabel("Percentiles")



    # in_ax = ax.inset_axes([0.7,-0.05,0.5,0.5])
    # in_ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
    for ax in axs:
            # Major ticks
        ax.tick_params(
            which="major",
            direction="in",
            top=True, right=True,
            length=4, width=1.
        )
        # Minor ticks
        ax.tick_params(
            which="minor",
            direction="in",
            top=True, right=True,
            length=3, width=1.0
        )


    dict_TARP = dict(mean_TARP=mean, sigma_TARP=sigma, n_sims_TARP=n_sims, ecp_TARP=ecp, std_TARP=std if bootstrap==True else None, 
    alpha_TARP=alpha, n_bins_TARP=n_bins, sigmas_cdf_TARP=sigmas_cdf, ecp_plot_TARP=ecp_plot, alpha_plot_TARP=alpha_plot, std_plot_TARP=std_plot if bootstrap==True else None, )


    if sbc==True:
        ecp, alpha = get_tarp_coverage(supersamples, supertrue, references='random', metric='euclidean', norm = True, seed = 5, bootstrap=bootstrap, sbc=sbc)
        if bootstrap==True:
            std = ecp.std(axis=0)
            ecp = ecp.mean(axis=0)
            std=std[1:]
        ecp = ecp[1:]
        alpha=alpha[1:]

        n_bins = alpha.shape[0]

        axs[0].step(alpha, cdf2pdf(ecp*n_sims), where='pre', color='darkred') # unormalizes
        if bootstrap==True:
            axs[0].fill_between(alpha, cdf2pdf(ecp*n_sims) - 1*cdf2pdf(std*n_sims), cdf2pdf(ecp*n_sims) + 1*cdf2pdf(std*n_sims), alpha=0.2, color='tab:blue',step='pre')
            axs[0].fill_between(alpha, cdf2pdf(ecp*n_sims) - 2*cdf2pdf(std*n_sims), cdf2pdf(ecp*n_sims) + 2*cdf2pdf(std*n_sims), alpha=0.2, color='tab:blue',step='pre')

        alpha_plot = np.concatenate(([0], alpha))
        ecp_plot   = np.concatenate(([0], ecp))
        if bootstrap==True:
            std_plot = np.concatenate(([0], std))

        axs[1].plot(alpha_plot, ecp_plot, label='SBC', color='darkred')
        if bootstrap==True:
            axs[1].fill_between(alpha_plot, ecp_plot - std_plot, ecp_plot + std_plot, alpha=0.2, color='darkred')
            axs[1].fill_between(alpha_plot, ecp_plot - 2*std_plot, ecp_plot + 2*std_plot, alpha=0.2, color='darkred')
            
        axs[2].plot(alpha_plot, (ecp_plot-alpha_plot), label='SBC', color='darkred')
        if bootstrap==True:
            axs[2].fill_between(alpha_plot, ecp_plot-alpha_plot - std_plot, ecp_plot-alpha_plot + std_plot, alpha=0.2, color='darkred')
            axs[2].fill_between(alpha_plot, ecp_plot-alpha_plot - 2*std_plot, ecp_plot-alpha_plot + 2*std_plot, alpha=0.2, color='darkred')

    dict_sbc = dict(mean_SBC=mean, sigma_SBC=sigma, n_sims_SBC=n_sims, ecp_SBC=ecp, std_SBC=std if bootstrap==True else None, 
    alpha_SBC=alpha, n_bins_SBC=n_bins, sigmas_cdf_SBC=sigmas_cdf, ecp_plot_SBC=ecp_plot, alpha_plot_SBC=alpha_plot, std_plot_SBC=std_plot if bootstrap==True else None, )
    # fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    # ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
    # ax.plot(alpha, ecp, label='TARP', color='purple')
    # ax.legend()
    # ax.set_ylabel("Expected Coverage (ECP)")
    # ax.set_xlabel("Credibility Level (1-a)")

    # plt.subplots_adjust(wspace=0.4)

    axs[1].legend()

    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print('saved figure pdf', flush = True)
    return dict(**dict_TARP, **dict_sbc)

































#  ██████╗  █████╗ ██████╗ ██████╗  █████╗  ██████╗ ███████╗
# ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝ ██╔════╝
# ██║  ███╗███████║██████╔╝██████╔╝███████║██║  ███╗█████╗  
# ██║   ██║██╔══██║██╔══██╗██╔══██╗██╔══██║██║   ██║██╔══╝  
# ╚██████╔╝██║  ██║██║  ██║██████╔╝██║  ██║╚██████╔╝███████╗
#  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
# GARBAGE




# def plot_all_field_stats(filename, sample, pred_ics, true_ics, added_unc, BOX_SIDE=200): # sample should be shape (n_samples, BOX_SIDE, BOX_SIDE, BOX_SIDE)
#     # plt.rcParams.update({'font.size': 22})
#     # plt.rcParams['lines.linewidth'] = plt.rcParamsDefault['lines.linewidth']
#     # plt.rc('text', usetex=True)  # You need to install LaTeX on your system for this to work
#     # plt.rc('font', family='serif',serif='Times')
#     # plt.rcParams.update({
#     # "font.size": 22,
#     # "font.family": "serif",   # font family
#     # "mathtext.fontset": "cm", # use Computer Modern (LaTeX-like)
#     # "axes.unicode_minus": False
#     # })


#     figure, axs = plt.subplots(3, 1, figsize=(6,10), gridspec_kw={'hspace': 0.0, 'height_ratios': [2, 1, 1]}, sharex=True)
#     counts, deltas = np.histogram(true_ics, bins=100, range=(-12,12), density=False)
#     bin_centers = 0.5 * (deltas[1:] + deltas[:-1])
#     axs[0].plot(bin_centers, counts/np.sum(counts), color='black', label='True')
#     counts, deltas = np.histogram(pred_ics

#     axs[0].plot(k, pk_true, color='black', label='True')
#     pk_pred, k = pbox.get_power(pred_ics, boxlength=300)
#     axs[0].plot(k, pk_pred, color='teal', label='MAP')


#     pk_sample, k = pbox.get_power(sample, boxlength=300)
#     axs[0].plot(k,pk_sample, color='purple', label='Sample')

#     pk_unc, k = pbox.get_power(added_unc, boxlength=300)
#     axs[0].plot(k, pk_unc, color='purple', label='Added uncertainty', linestyle='--')

#     axs[0].fill_between(np.linspace(k_nyquist,6, 100), 1e-1, 1e6, label = r'$\rm >k_{Nyquist}$', color='black', alpha=0.3)

#     axs[0].set_xscale('log')
#     axs[0].set_yscale('log')
#     axs[0].set_ylim([1e-1,1e5])
#     axs[0].set_xlim([0.025,5])
#     axs[0].set_ylabel(r'P(k) [$\rm Mpc^3]$')
#     axs[0].set_xlabel(r'k [$\rm  \; Mpc^{-1}$]')
#     # axs[0].legend()



#     pk_cross_pred = pbox.get_power(deltax= true_ics, deltax2 = pred_ics, boxlength=300)[0]
#     pk_cross_sample = pbox.get_power(deltax=true_ics, deltax2=sample, boxlength=300)[0]
#     r_pred = pk_cross_pred/np.sqrt(pk_pred*pk_true)
#     r_sample = pk_cross_sample/np.sqrt(pk_sample*pk_true)
#     axs[1].plot(k, r_pred, color='teal', label='MAP')
#     axs[1].plot(k, r_sample, color='purple', label='Sample')
#     axs[1].fill_between(np.linspace(k_nyquist,6, 100), 0, 1.2, label = r'$\rm >k_{Nyquist}$', color='black', alpha=0.3)
#     axs[1].axhline(1, color='black', linestyle='--')
#     axs[1].set_xscale('log')
#     axs[1].set_ylim([0,1.2])
#     axs[1].set_xlim([0.025,5])
#     axs[1].set_ylabel(r'CCC(k)')
#     axs[1].set_xlabel(r'k [$\rm  \; Mpc^{-1}$]')
#     axs[1].legend()

#     # Plot transfer function
#     T_pred = np.sqrt(pk_pred/pk_true)
#     T_sample = np.sqrt(pk_sample/pk_true)
#     axs[2].plot(k, T_pred, color='teal', label='MAP')
#     axs[2].plot(k, T_sample, color='purple', label='Sample')
#     axs[2].fill_between(np.linspace(k_nyquist,6, 100), 0, 2, label = r'$\rm >k_{Nyquist}$', color='black', alpha=0.3)
#     axs[2].axhline(1, color='black', linestyle='--')
#     axs[2].set_xscale('log')
#     axs[2].set_ylim([0.,2.0])
#     axs[2].set_xlim([0.025,5])
#     axs[2].set_ylabel(r'T(k)')
#     axs[2].set_xlabel(r'k [$\rm  \; Mpc^{-1}$]')
#     # plt.legend()

#     for ax in axs:
#             # Major ticks
#         ax.tick_params(
#             which="major",
#             direction="in",
#             top=True, right=True,
#             length=6, width=1.2
#         )
#         # Minor ticks
#         ax.tick_params(
#             which="minor",
#             direction="in",
#             top=True, right=True,
#             length=3, width=1.0
#         )

#     plt.savefig(filename, dpi=200, bbox_inches='tight')
#     print('saved figure all power spectra', flush = True)
#     plt.close()



# def plot_power_spectrum(filename, sample, pred_ics, true_ics, added_unc, BOX_SIDE=200): 
#     import powerbox as pbox
#     k_nyquist = np.pi*BOX_SIDE/300

#     pk_true, k = pbox.get_power(true_ics, boxlength=300)
#     plt.plot(k, pk_true, color='black', label='True')
#     pk_pred, k = pbox.get_power(pred_ics, boxlength=300)
#     plt.plot(k, pk_pred, color='teal', label='MAP')


#     pk_sample, k = pbox.get_power(sample, boxlength=300)
#     plt.plot(k,pk_sample, color='purple', label='Sample')

#     pk_unc, k = pbox.get_power(added_unc, boxlength=300)
#     plt.plot(k, pk_unc, color='purple', label='Added uncertainty', linestyle='--')

#     plt.fill_between(np.linspace(k_nyquist,6, 100), 1e-1, 1e6, label = r'$\rm >k_{Nyquist}$', color='black', alpha=0.3)

#     plt.xscale('log')
#     plt.yscale('log')
#     plt.ylim([1e-1,1e6])
#     plt.xlim([0.025,5])
#     plt.ylabel(r'P(|k|) [$\rm Mpc^3]$')
#     plt.xlabel(r'k [$\rm  \; Mpc^{-1}$]')
#     plt.legend()


#     plt.savefig(filename, dpi=200, bbox_inches='tight')
#     plt.close()
#     print('saved figure 3', flush = True)


#     # Plot transfer function
#     T_pred = np.sqrt(pk_pred/pk_true)
#     T_sample = np.sqrt(pk_sample/pk_true)
#     plt.plot(k, T_pred, color='teal', label='MAP')
#     plt.plot(k, T_sample, color='purple', label='Sample')
#     plt.fill_between(np.linspace(k_nyquist,6, 100), 0, 2, label = r'$\rm >k_{Nyquist}$', color='black', alpha=0.3)
#     plt.axhline(1, color='black', linestyle='--')
#     plt.xscale('log')
#     plt.ylim([0.,2.0])
#     plt.xlim([0.025,5])
#     plt.ylabel(r'T(|k|)')
#     plt.xlabel(r'k [$\rm  \; Mpc^{-1}$]')
#     plt.legend()
#     plt.savefig(filename.replace('.png','_T.png'), dpi=200, bbox_inches='tight')
#     plt.close()
#     print('saved figure 4', flush = True)

#     # Plot cross-correlation
#     pk_cross_pred = pbox.get_power(deltax= true_ics, deltax2 = pred_ics, boxlength=300)[0]
#     pk_cross_sample = pbox.get_power(deltax=true_ics, deltax2=sample, boxlength=300)[0]
#     r_pred = pk_cross_pred/np.sqrt(pk_pred*pk_true)
#     r_sample = pk_cross_sample/np.sqrt(pk_sample*pk_true)
#     plt.plot(k, r_pred, color='teal', label='MAP')
#     plt.plot(k, r_sample, color='purple', label='Sample')
#     plt.fill_between(np.linspace(k_nyquist,6, 100), 0, 1.2, label = r'$\rm >k_{Nyquist}$', color='black', alpha=0.3)
#     plt.axhline(1, color='black', linestyle='--')
#     plt.xscale('log')
#     plt.ylim([0,1.2])
#     plt.xlim([0.025,5])
#     plt.ylabel(r'r(|k|)')
#     plt.xlabel(r'k [$\rm  \; Mpc^{-1}$]')
#     plt.legend()
#     plt.savefig(filename.replace('.png','_r.png'), dpi=200, bbox_inches='tight')
#     plt.close()
#     print('saved figure 5', flush = True)






# def plot_slices(filename, pred_ics, true_ics, data, sample, added_unc, theta):
#     z_slice_position = 23
#     cmap='viridis'
#     vmin=-5
#     vmax=6

#     if theta.shape[1]==2:
#         data2 = np.array(theta[0,1,:,:,:].cpu().detach())

#     fig, axs = plt.subplots(2, 3, figsize=(20,8))
#     p1 = axs[0,0].imshow(true_ics[:,:,z_slice_position], extent=[0,300,0,300], origin='lower', cmap=cmap, vmin=vmin,vmax=vmax)
    
#     fig.colorbar(p1, ax=axs[0,0], label = r'$\rm \delta $')  
#     axs[0,0].set_title(label = 'True ICs')


#     # pred, _ = model(theta)
#     p2 = axs[1,1].imshow(pred_ics[:,:,z_slice_position], extent=[0,300,0,300], origin='lower',cmap=cmap, vmin=vmin,vmax=vmax)
#     fig.colorbar(p2, ax=axs[1,1], label = r'$\rm \delta $')  
#     axs[1,1].set_title(label = 'MAP');


#     p3 = axs[0,1].imshow(data[:,:,z_slice_position], extent=[0,300,0,300], origin='lower',cmap=cmap)
#     fig.colorbar(p3, ax=axs[0,1], label = r'$\rm \delta $')  
#     axs[0,1].set_title(label = 'Data');


#     p4 = axs[1,0].imshow(sample[:,:,z_slice_position], extent=[0,300,0,300], origin='lower',cmap=cmap, vmin=vmin,vmax=vmax)
#     fig.colorbar(p4, ax=axs[1,0], label = r'$\rm \delta $')
#     axs[1,0].set_title(label = 'Sample');



#     p5 = axs[1,2].imshow(added_unc[:,:,z_slice_position], extent=[0,300,0,300], origin='lower',cmap=cmap)
#     fig.colorbar(p5, ax=axs[1,2], label = r'$\rm \delta $')
#     axs[1,2].set_title(label = 'Added uncertainty');


#     if theta.shape[1]==2:
#         p6 = axs[0,2].imshow(data2[:,:,z_slice_position], extent=[0,300,0,300], origin='lower',cmap=cmap)
#         fig.colorbar(p5, ax=axs[0,2], label = r'$\rm num $')
#         axs[0,2].set_title(label = 'Galaxy counts');
#     else:
#         fig.delaxes(axs[0,2])


#     axs[1,0].set_xlabel('x-axis [Mpc]')
#     axs[1,1].set_xlabel('x-axis [Mpc]')
#     axs[1,2].set_xlabel('x-axis [Mpc]')

#     axs[0,0].set_ylabel('y-axis [Mpc]')
#     axs[1,0].set_ylabel('y-axis [Mpc]')


#     plt.savefig(filename, dpi=200, bbox_inches='tight')
#     plt.close()
#     print('saved figure 1', flush = True)




# def plot_pdf(filename, sample, pred_ics, true_ics):
#     plt.figure(figsize=(8,6))
#     plt.hist(true_ics.flatten(), bins=100, density=True, alpha=0.5, label='True ICs', color='black')
#     plt.hist(pred_ics.flatten(), bins=100, density=True, alpha=0.5, label='NN_PE', color='teal')
#     plt.hist(sample.flatten(), bins=100, density=True, alpha=0.5, label='Sample', color='purple')
#     plt.xlabel(r'$\rm \delta $')
#     plt.ylabel(r'$\rm P(\delta)$')
#     # plt.title('PDF of True vs Predicted ICs')
#     plt.legend()
#     plt.savefig(filename, dpi=200, bbox_inches='tight')
#     plt.close()
#     print('saved figure 2', flush = True)



# def add_zoom_inset(ax, data_slice, cmap, extent, zoom_region,
#                    inset_loc='upper right', zoom_percent=0.35, box_color='white', mark=True, vmin=None, vmax=None, norm=None):
#     """
#     Adds a zoomed-in inset to a matplotlib axis.

#     Parameters:
#     - ax: Main matplotlib axis to add inset to
#     - data_slice: 2D data to plot in the inset
#     - cmap: Colormap for the inset plot
#     - extent: [xmin, xmax, ymin, ymax] extent for imshow
#     - zoom_region: (x1, x2, y1, y2) region to zoom into
#     - inset_loc: Position of the inset in the main axis (e.g., 'upper right')
#     - zoom_percent: Size of the inset relative to main plot (default 0.35)
#     - box_color: Color of the rectangle and inset border (default white)
#     - mark: If True, draws connecting lines between main plot and inset
#     """
#     x1, x2, y1, y2 = zoom_region

#     # Draw zoom box on main plot
#     rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
#                              linewidth=2, edgecolor=box_color, facecolor='none', linestyle='--')
#     ax.add_patch(rect)

#     # Add inset axis
#     axins = inset_axes(ax, width=f"{int(zoom_percent*100)}%", height=f"{int(zoom_percent*100)}%",
#                        loc=inset_loc, borderpad=1)
#     axins.set_in_layout(False)          # ← crucial
#     axins.imshow(data_slice, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
#     axins.set_xlim(x1, x2)
#     axins.set_ylim(y1, y2)
#     axins.set_xticks([])
#     axins.set_yticks([])
#     for spine in axins.spines.values():
#         spine.set_edgecolor(box_color)

#     # Connect inset to rectangle
#     if mark:
#         mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec=box_color, lw=1)

# def add_colorbar(fig, im, ax, label=None):
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     cbar = fig.colorbar(im, cax=cax)
#     if label:
#         cbar.set_label(label)
#     return cbar





# import os, sys
# import contextlib

# @contextlib.contextmanager
# def suppress_output():
#     with open(os.devnull, "w") as devnull:
#         old_stdout, old_stderr = sys.stdout, sys.stderr
#         sys.stdout, sys.stderr = devnull, devnull
#         try:
#             yield
#         finally:
#             sys.stdout, sys.stderr = old_stdout, old_stderr




















# def plot_TARP(filename, supersamples, supertrue, bootstrap=True): # supersamples should have shape (num_samples, num_sims, num_dims) # theta has shape: (num_sims, num_dims)
#     from tarp import get_tarp_coverage
#     ecp, alpha = get_tarp_coverage(supersamples, supertrue, references='random', metric='euclidean', norm = True, seed = 5, bootstrap= bootstrap)

#     if bootstrap==True:
#         std = ecp.std(axis=0)
#         ecp = ecp.mean(axis=0)

#     # calculate the uniform distribution
#     ecp_uniform = ecp.copy()
#     for i in range(1, len(ecp),1):
#         ecp_uniform[i] = ecp[i]-ecp[i-1]
#     nsims = supertrue.shape[0]
#     bins = ecp.shape[0]
#     mean = nsims* (1/bins) # This is the mean but essentially we are bringing it to zero by subtracting
#     sigma2 = mean * (1-1/bins)
#     sigma = np.sqrt(sigma2)

#     fig, axs = plt.subplots(2, 1, figsize=(4, 6),  gridspec_kw={'hspace': 0.0, 'wspace': 0.0, 'height_ratios': [1, 2,]}, sharex=True)


#     axs[0].set_ylabel("Expected Coverage")
#     axs[0].axhline(0, color='black', linestyle='--')
#     # axs[0].set_ylim([-0.5,0.5])

    

#     axs[0].plot(alpha, ecp_uniform)
#     axs[0].axhspan(-sigma, sigma, color="black", alpha=0.2,)
#     axs[0].axhspan(-2*sigma, 2*sigma, color="black", alpha=0.2,)



#     if bootstrap==True:
#         ecp_bootstrap = ecp.copy()
#         axs[1].plot(alpha, ecp, label='TARP', color='purple')
#         k_sigma = [1,2]
#         for k in k_sigma:
#             axs[1].fill_between(alpha, ecp - k * std, ecp + k * std, alpha = 0.2, color='purple')
#     else:
#         axs[1].plot(alpha, ecp, label='TARP', color='purple')

#     axs[1].plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
#     axs[1].legend()
#     axs[1].set_ylabel("Expected Coverage (Empirical CDF)")
#     axs[1].set_xlabel("Credibility Level percentiles")


#     # in_ax = ax.inset_axes([0.7,-0.05,0.5,0.5])
#     # in_ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
#     for ax in axs:
#             # Major ticks
#         ax.tick_params(
#             which="major",
#             direction="in",
#             top=True, right=True,
#             length=4, width=1.
#         )
#         # Minor ticks
#         ax.tick_params(
#             which="minor",
#             direction="in",
#             top=True, right=True,
#             length=3, width=1.0
#         )

#     plt.subplots_adjust(wspace=0.4)



#     # fig, ax = plt.subplots(1, 1, figsize=(4, 4))

#     # ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
#     # ax.plot(alpha, ecp, label='TARP', color='purple')
#     # ax.legend()
#     # ax.set_ylabel("Expected Coverage (ECP)")
#     # ax.set_xlabel("Credibility Level (1-a)")

#     # plt.subplots_adjust(wspace=0.4)
#     plt.savefig(filename, dpi=200, bbox_inches='tight')
#     plt.close()
#     print('saved figure pdf', flush = True)




# from _my_funcs.coverage_tests import get_tarp_coverage
# def plot_TARP(filename, supersamples, supertrue, bootstrap=True): # supersamples should have shape (num_samples, num_sims, num_dims) # supertrue has shape: (num_sims, num_dims)

#     n_sims = supertrue.shape[0]

#     ecp, alpha = get_tarp_coverage(supersamples, supertrue, references='random', metric='euclidean', norm = True, seed = 5, bootstrap=bootstrap)

#     if bootstrap==True:
#         std = ecp.std(axis=0)
#         ecp = ecp.mean(axis=0)
#         std=std[1:]

#     ecp = ecp[1:]
#     alpha=alpha[1:]

#     n_bins = alpha.shape[0]

#     # Theoretical
#     mean = n_sims/n_bins
#     p = 1/n_bins
#     sigma = np.sqrt(n_sims*p*(1-p))

#     mean_norm = mean/n_sims
#     sigma_norm = sigma/n_sims

#     def cdf2pdf(cdf_array):
#         pdf_array = np.zeros(cdf_array.shape)
#         pdf_array[0] = cdf_array[0]
#         for i in range(1,cdf_array.shape[0],1):
#             pdf_array[i] = cdf_array[i]-cdf_array[i-1]
#         return pdf_array

#     sigmas_cdf = []
#     for i in range(alpha.shape[0]):
#         ii=i+1
#         p = ii/n_bins
#         sigma_i = np.sqrt(n_sims*p*(1-p))
#         sigmas_cdf.append(sigma_i)
#     sigmas_cdf = np.array(sigmas_cdf)/n_sims


#     fig, axs = plt.subplots(3, 1, figsize=(4, 6),  gridspec_kw={ 'hspace': 0.0, 'wspace': 0.0, 'height_ratios': [1, 2,1]}, sharex=True)
#     axs[0].set_ylabel("Coverage per bin")

#     # 

#     axs[0].axhline(mean, color='black', linestyle='--')

#     # axs[0].fill_between(alpha, ecp_uniform - sigmas_cdf, ecp_uniform+sigmas_cdf, alpha=0.2)
#     axs[0].axhspan(mean-sigma, mean+sigma, color="black", alpha=0.2,)
#     axs[0].axhspan(mean-2*sigma, mean+2*sigma, color="black", alpha=0.2,)

#     # axs[0].step(alpha, cdf2pdf(ecp*n_sims), where='pre') # unormalizes

#     axs[0].step(alpha, cdf2pdf(ecp*n_sims), where='pre') # unormalizes


#     if bootstrap==True:
#         axs[0].fill_between(alpha, cdf2pdf(ecp*n_sims) - 1*cdf2pdf(std*n_sims), cdf2pdf(ecp*n_sims) + 1*cdf2pdf(std*n_sims), alpha=0.2, color='tab:blue',step='pre')
#         axs[0].fill_between(alpha, cdf2pdf(ecp*n_sims) - 2*cdf2pdf(std*n_sims), cdf2pdf(ecp*n_sims) + 2*cdf2pdf(std*n_sims), alpha=0.2, color='tab:blue',step='pre')
#     # axs[0].plot(alpha, cdf2pdf(ecp))  # normalized

#     # axs[0].plot(alpha*n_sims, cdf2pdf(alpha*n_sims)) # essentially the mean

#     #ADD 0 at first as a known theoretical result
#     alpha_plot = np.concatenate(([0], alpha))
#     ecp_plot   = np.concatenate(([0], ecp))
#     sigmas_cdf_plot  = np.concatenate(([0], sigmas_cdf))
#     if bootstrap==True:
#         std_plot = np.concatenate(([0], std))


#     axs[1].plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
#     axs[1].fill_between(alpha_plot, alpha_plot - sigmas_cdf_plot, alpha_plot+sigmas_cdf_plot, alpha=0.2, color='black')
#     axs[1].fill_between(alpha_plot, alpha_plot - 2*sigmas_cdf_plot, alpha_plot+2*sigmas_cdf_plot, alpha=0.2, color='black')
#     # axs[1].plot(alpha_plot, [0, 1], ls='--', color='k', label = "Ideal case")
#     axs[1].plot(alpha_plot, ecp_plot, label='TARP', color='tab:blue')
#     if bootstrap==True:
#         axs[1].fill_between(alpha_plot, ecp_plot - std_plot, ecp_plot + std_plot, alpha=0.2, color='tab:blue')
#         axs[1].fill_between(alpha_plot, ecp_plot - 2*std_plot, ecp_plot + 2*std_plot, alpha=0.2, color='tab:blue')
        

#     # axs[1].plot(alpha, ecp-alpha,  color='purple', label='SBC')
#     axs[1].legend()
#     axs[1].set_ylabel("ECDF")
#     axs[1].set_xlabel("Credibility Level percentiles")



#     axs[2].axhline(0, ls='--', color='k', label = "Ideal case")
#     axs[2].fill_between(alpha_plot, - sigmas_cdf_plot, +sigmas_cdf_plot, alpha=0.2, color='black')
#     axs[2].fill_between(alpha_plot, - 2*sigmas_cdf_plot, +2*sigmas_cdf_plot, alpha=0.2, color='black')

#     axs[2].plot(alpha_plot, (ecp_plot-alpha_plot), label='TARP', color='tab:blue')
#     if bootstrap==True:
#         axs[2].fill_between(alpha_plot, ecp_plot-alpha_plot - std_plot, ecp_plot-alpha_plot + std_plot, alpha=0.2, color='tab:blue')
#         axs[2].fill_between(alpha_plot, ecp_plot-alpha_plot - 2*std_plot, ecp_plot-alpha_plot + 2*std_plot, alpha=0.2, color='tab:blue')
        

#     axs[2].set_ylabel("ECDF Difference")
#     axs[2].set_xlabel("Percentiles")



#     # in_ax = ax.inset_axes([0.7,-0.05,0.5,0.5])
#     # in_ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
#     for ax in axs:
#             # Major ticks
#         ax.tick_params(
#             which="major",
#             direction="in",
#             top=True, right=True,
#             length=4, width=1.
#         )
#         # Minor ticks
#         ax.tick_params(
#             which="minor",
#             direction="in",
#             top=True, right=True,
#             length=3, width=1.0
#         )





#     # fig, ax = plt.subplots(1, 1, figsize=(4, 4))

#     # ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
#     # ax.plot(alpha, ecp, label='TARP', color='purple')
#     # ax.legend()
#     # ax.set_ylabel("Expected Coverage (ECP)")
#     # ax.set_xlabel("Credibility Level (1-a)")

#     # plt.subplots_adjust(wspace=0.4)
#     plt.savefig(filename, dpi=200, bbox_inches='tight')
#     plt.close()
#     print('saved figure pdf', flush = True)

