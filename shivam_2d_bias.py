# ------------------ Shivam's code to compute 2d bias

import numpy as np
import matplotlib.pyplot as plot
from getdist import MCSamples, plots
import scipy.optimize as op
import scipy.special as spsp
from getdist import MCSamples, plots
from getdist.paramnames import escapeLatex, makeList, mergeRenames

def get_max_2dpost(g, root, param1='cosmological_parameters--omega_m', param2='cosmological_parameters--s8',param_pair = None):
    param_pair = g.get_param_array(root, param_pair or [param1, param2])

    density = g.sample_analyser.get_density_grid(root, param_pair[0], param_pair[1],
                                                                conts=g.settings.num_plot_contours,
                                                                likes=g.settings.shade_meanlikes)
    xyind = np.where(density.P == np.amax(density.P))
    return density.y[xyind[0][0]],density.x[xyind[1][0]]

def get_contour_line(sigma_contour,g, density):
    contours = spsp.erf(sigma_contour/np.sqrt(2))
    density.contours = density.getContourLevels([contours])
    contour_levels = density.contours
    fig1, ax1 = plot.subplots(1)
    cs = plot.contour(density.x, density.y, density.P, sorted(contour_levels))
    lines = []
    for line in cs.collections[0].get_paths():
        lines.append(line.vertices)
    plot.close()
    return lines

def get_dmin(sigma_contour, g, density, xref, yref): 
    line = get_contour_line(sigma_contour,g, density)

    linex = line[0][:,0]
    liney = line[0][:,1]
    d_all = np.sqrt((linex - xref)**2 + (liney - yref)**2)

    return np.amin(d_all)

diff = np.vectorize(get_dmin)

def root_find(init_x,g, density, xref, yref): 
    nll = lambda *args: diff(*args)
    args = (g, density, xref, yref)
    result = op.root(nll,np.array([init_x]),args=args,options={'maxfev':50})
    return result

def min_find(init_x,g, density, xref, yref): 
    nll = lambda *args: diff(*args)
    args = (g, density, xref, yref)
    result = op.fmin(nll,np.array([init_x]),args=args)
    return result

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_frac_angle(start_coord, end_coord,xlims, ylims):
    xmean12 = (0.5*(start_coord[0] + end_coord[0]))
    ymean12 = (0.5*(start_coord[1] + end_coord[1]))
    dx = xlims[1] - xlims[0]
    dy = ylims[1] - ylims[0]
    dxp = xmean12 - end_coord[0]
    dyp = ymean12 - end_coord[1]
    angle_deg = 180.+ np.arctan2((end_coord[1]-start_coord[1])/dy,( end_coord[0]-start_coord[0])/dx)*(180./np.pi)
    return xmean12-dxp/2, ymean12+dyp/3, angle_deg

def compute_2d_bias(baseline, contaminated, truth, param1, param2, label1, label2, guess=0.01, output=None):
    g = plots.getSinglePlotter()

    #xlims = [0.27,0.33]
    #ylims = [0.8,0.86]

    x1 = min([baseline.confidence(param1, 0.2, upper=False), contaminated.confidence(param1, 0.2, upper=False)])
    x2 = max([baseline.confidence(param1, 0.2, upper=True), contaminated.confidence(param1, 0.2, upper=True)])

    y1 = min([baseline.confidence(param2, 0.2, upper=False), contaminated.confidence(param2, 0.2, upper=False)])
    y2 = max([baseline.confidence(param2, 0.2, upper=True), contaminated.confidence(param2, 0.2, upper=True)])

    xlims = [x1, x2]
    ylims = [y1, y2]

    param2_cont, param1_cont = get_max_2dpost(g, contaminated, param1=param1, param2=param2)
    param2_base, param1_base = get_max_2dpost(g, baseline, param1=param1, param2=param2)

    # Truth values of various parameters
    param1_truth, param2_truth = truth[param1], truth[param2]

    param_pair = g.get_param_array(baseline, None or [param1, param2])

    density = g.sample_analyser.get_density_grid(baseline, param_pair[0], param_pair[1],
                                                                conts=g.settings.num_plot_contours,
                                                                likes=g.settings.shade_meanlikes)

    g = plots.getSinglePlotter()
    density_cont = g.sample_analyser.get_density_grid(contaminated, param_pair[0], param_pair[1],
                                                                conts=g.settings.num_plot_contours,
                                                                likes=g.settings.shade_meanlikes)

    # Get the 2D distance in terms of sigma. In case you get some weird getdist error, change the starting guess value (here 0.01)
    res_truth_base = min_find(guess, g, density, param1_truth, param2_truth)
    res_cont_base = min_find(guess, g, density_cont, param1_base, param2_base)
    res_base_cont = min_find(guess, g, density, param1_cont, param2_cont)
    res_truth_cont = min_find(guess, g, density_cont, param1_truth, param2_truth)

    line = get_contour_line(0.3,g, density)
    line_cont = get_contour_line(0.3,g, density_cont)

    fig, ax = plot.subplots(1, figsize=(8,6))
    ax.axvline(param1_truth,ls='--',alpha=0.3,color='k')
    ax.axhline(param2_truth,ls='--',alpha=0.3,color='k')

    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    ax.plot([param1_cont],[param2_cont], linestyle='', marker='o', color='b',label='Contaminated')
    ax.plot([param1_base],[param2_base], linestyle='', marker='s', color='r',label='Baseline')
    ax.plot(line[0][:,0],line[0][:,1], linestyle='--', marker='',lw=1, color='red',label=r'$0.3 \sigma$ Baseline')
    ax.plot(line_cont[0][:,0],line_cont[0][:,1], linestyle='--', marker='',lw=1, color='blue',label=r'$0.3 \sigma$ Contaminated')

    ax.annotate('', xy=(param1_truth, param2_truth), xytext=(param1_base, param2_base),
                arrowprops={'arrowstyle': '->'}, va='center')
    ax.annotate('', xy=(param1_truth, param2_truth), xytext=(param1_cont, param2_cont),
                arrowprops={'arrowstyle': '->'}, va='center')
    ax.annotate('', xy=(param1_base, param2_base), xytext=(param1_cont, param2_cont),
                arrowprops={'arrowstyle': '->'}, va='center')

    ax.set_xlabel(label1, size = 22)
    ax.set_ylabel(label2, size = 22)
    ax.legend(fontsize=17,loc='lower right')
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    # you might need to change the fracx, fracy by a little bit for the text to not overlap with the lines and show correctly around arrows
    # Also you might need to rotate the angle by 180 to have correct orientation

    fracx, fracy, angle = get_frac_angle(np.array([param1_cont, param2_cont]),np.array([param1_base,param2_base]),xlims, ylims)
    ax.text( fracx, fracy,str(np.round(res_cont_base[0],2)) + r'$\sigma$', rotation=angle, fontsize=16) 
    fracx, fracy, angle = get_frac_angle(np.array([param1_base,param2_base]), np.array([param1_truth, param2_truth]),xlims, ylims)
    ax.text( fracx, fracy,str(np.round(res_truth_base[0],2)) + r'$\sigma$',  rotation=angle, fontsize=16)    
    fracx, fracy, angle = get_frac_angle(np.array([param1_cont, param2_cont]),np.array([param1_truth,param2_truth]),xlims, ylims)
    ax.text( fracx, fracy,str(np.round(res_truth_cont[0],2)) + r'$\sigma$', rotation=angle, fontsize=16)    

    if output != None:
        fig.savefig(output)

    return {
            'truth to baseline': res_truth_base[0],
            'contaminated to baseline': res_cont_base[0],
            'baseline to contaminated': res_base_cont[0],
            'truth to contaminated': res_truth_cont[0],
            }

# ------------------ end of Shivam's code to compute 2d bias
