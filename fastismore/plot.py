# coding: utf-8

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import argparse, yaml
import itertools as itt

import getdist as gd
import getdist.plots

from . import chain as fchain
from . import parameters as fparams
from . import VERBOSE

__all__ = [
    'plot_posterior',
    'plot_1d',
    'plot_2d',
    'get_stats',
    'plot_triangle_getdist',
    'plot_triangle',
    'plot_weights',
    'get_markdown_stats',
    'setup_config',
    ]

figwidth = 440/72.27
plot_ratio = 1.5 #0.5*(1+5**0.5)

default_colors = ['#000000', '#3E89DA', '#F87A44', '#427B48', '#927FC3']
default_linestyles = ['-', ':', '--', '-.', (0, (3, 1, 1, 1, 1, 1))]
default_markers = ['o', '<', '>', 'v', '^']

def setup_config():

    plt.rcParams['figure.dpi']= 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.facecolor']= 'white'
    plt.rcParams['text.usetex']= True
    plt.rcParams['font.family']= 'serif'
    plt.rcParams['font.serif']= 'cm'
    plt.rcParams['font.size']= 10
    plt.rcParams['pgf.texsystem']= "pdflatex"
    plt.rcParams['pgf.rcfonts']= False
    plt.rcParams['lines.linewidth'] = 1.75
    plt.rcParams['figure.figsize'] = (figwidth, figwidth/plot_ratio)
    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#000000', '#3E89DA', '#FEFEFE', '#F87A44'])

def plot_posterior(chain, param1, param2, truth=None, equal_ratio=False, extent=None):
    fig, ax = plt.subplots()

    density = chain.get_density_grid(param1, param2)

    ax.imshow(density.P, extent=[min(density.x),
                                 max(density.x),
                                 min(density.y),
                                 max(density.y)],
            origin='lower',
            cmap='jet')

    if truth is not None:
        ax.axvline(truth[param1], ls='--',alpha=0.3,color='k')
        ax.axhline(truth[param2], ls='--',alpha=0.3,color='k')

    ax.plot(*chain.get_peak_2d(param1, param2),  ls='', c='white', marker='^', markersize=6,label='Peak', zorder=5)
    ax.plot(*chain.get_mean([param1, param2]),  ls='', c='white', marker='s', markersize=6,label='Mean', zorder=5)

    for sigma in np.arange(0,2.1,0.1)[1:]:
        for cv in chain.get_contour_vertices(sigma, param1, param2):
            ax.plot(*cv.T, marker='',lw=0.8, c=f'{sigma*0.5}', label=f'${sigma:.1f} \sigma$')

    if extent is None:
        range_points = chain.get_contour_vertices(2.2, param1, param2)[-1]
        range_points = np.array([np.min(range_points, axis=0), np.max(range_points, axis=0)]).T
    else:
        range_points = extent
    range_diffs = np.diff(range_points)

    if equal_ratio:
        mid = np.sum(range_points, axis=1)/2
        if range_diffs[1] < range_diffs[0]:
            ax.set_xlim(mid[0]-range_diffs[1]/2*plot_ratio, mid[0]+range_diffs[1]/2*plot_ratio)
            ax.set_ylim(mid[1]-range_diffs[1]/2, mid[1]+range_diffs[1]/2)
        else:
            ax.set_xlim(mid[0]-range_diffs[0]/2, mid[0]+range_diffs[0]/2)
            ax.set_ylim(mid[1]-range_diffs[0]/2/plot_ratio, mid[1]+range_diffs[0]/2/plot_ratio)
        ax.set_aspect(1)

    else:
        ax.set_xlim(*range_points[0])
        ax.set_ylim(*range_points[1])
        ax.set_aspect(range_diffs[0]/range_diffs[1]/plot_ratio)


    ax.set_xlabel(fparams.param_to_latex(param1))
    ax.set_ylabel(fparams.param_to_latex(param2))

    legend_handles, legend_labels, _, _ = mpl.legend._parse_legend_args([ax])
    ax.legend(legend_handles[:2], legend_labels[:2], frameon=False, labelcolor='linecolor')

    # ax.legend(loc=(1,0))
    return fig

def plot_2d(param1, param2, chains, truth, labels=None, sigma=0.3, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    _subplot_2d(ax, param1, param2, chains, truth, labels, sigma)
    if labels is not None:
        ax.legend(loc=(1.05,0))
    ax.set_xlabel(fparams.param_to_latex(param1))
    ax.set_ylabel(fparams.param_to_latex(param2))
    return fig

def plot_triangle(params, chains, truth, labels, sigma, show_peaks=True, param_labels=None, figsize=(20,20), show_1d=True, show_bands=True):

    list_chains = chains if isinstance(chains, (list, tuple)) else [chains]

    fig = plt.figure(figsize=figsize)

    axes = fig.add_gridspec(len(params), len(params), hspace=0, wspace=0).subplots(sharex='col', sharey='row')
        
    i, j = np.mgrid[0:len(params),0:len(params)]

    print('Plotting 2D')
    for ax,a,b in zip(axes[i > j], j[i > j], i[i > j]):
        _subplot_2d(ax=ax, param1=params[a], param2=params[b], chains=list_chains, truth=truth, labels=labels, sigma=sigma, show_peaks=show_peaks)
        ax.tick_params(direction='inout')
        ax.set_xlabel(fparams.param_to_latex(params[a]))
        ax.set_ylabel(fparams.param_to_latex(params[b]))

    if show_1d:
        if show_bands:
            print('Plotting bands')
            for ax,a,b in zip(axes[i > j], j[i > j], i[i > j]):
                left_a, right_a = list_chains[0].get_bounds(params[a], sigma=sigma, maxlike=False)
                left_b, right_b = list_chains[0].get_bounds(params[b], sigma=sigma, maxlike=False)

                ax.axvspan(left_a, right_a, facecolor="none", edgecolor='#000',alpha=0.1, hatch='/////')
                ax.axhspan(left_b, right_b, facecolor="none", edgecolor='#000',alpha=0.1, hatch='\\\\\\\\\\')
                
                # ax.fill_between([left_a, right_a], [left_b, left_b], [right_b, right_b], alpha = 0.2)

        plt.tight_layout() # Force plot range calcs
        
        for m,_ in enumerate(params[:-1]):
            axes[m,m].set_xlim(*axes[-1,m].get_xlim())
        # Set range of the rightmost 1D plot to its corresponding row y-range
        axes[-1,-1].set_xlim(*axes[-1,0].get_ylim())

        print('Plotting 1D')
        for ax,a in zip(np.diag(axes), np.diag(i)):
            ax.get_shared_y_axes().remove(ax)
            ax.yaxis.major = mpl.axis.Ticker()
            ax.yaxis.set_major_locator(mpl.ticker.AutoLocator())
            ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            ax.get_shared_x_axes().remove(ax)
            _subplot_1d(ax=ax, param=params[a], chains=list_chains, truth=truth, labels=labels, sigma=sigma, show_bands=show_bands)
            ax.set_xlabel(fparams.param_to_latex(params[a]))
            ax.set_yticks([])
            ax.margins(x=0)
            ax.set_ylim(0, ax.get_ylim()[1])

    for ax in axes[(i < j) if show_1d else (i <= j)]:
        ax.remove()
        
    for ax in axes.flatten():
        ax.label_outer()
        
    legend_handles, legend_labels, _, _ = mpl.legend._parse_legend_args([axes[1,0]])
    
    (axes[0,0] if show_1d else axes[1,1]).legend(legend_handles,
        legend_labels,
        loc=(1.05,0.2),
        ncol=(2 if (show_peaks and len(list_chains) > 1) else 1),
        frameon=False
    )

    return fig, axes

def _subplot_2d(ax, param1, param2, chains, truth, labels=None, sigma=0.3, show_peaks=True):
    istart = 1 if len(default_colors) - len(chains) == 1 else 0
    linestyles = default_linestyles[istart:]
    markers = default_markers[istart:]
    colors = default_colors[istart:]
    markersize=6
    lw=1.8
    
    ax.axvline(truth[param1], ls='--',alpha=0.5,color='k')
    ax.axhline(truth[param2], ls='--',alpha=0.5,color='k')

    if labels is None:
        labels = len(chains)*[None]
    
    if show_peaks:
        for chain,ls,m,c,l in zip(chains, linestyles, markers, colors, labels):
            ax.plot(*chain.get_peak_2d(param1, param2),  ls='', marker=m, markersize=markersize, c=c,label=('Peak ' + l) if l is not None else l, zorder=5)
    
    for chain,ls,m,c,l in zip(chains, linestyles, markers, colors, labels):
        for cv in chain.get_contour_vertices(sigma, param1, param2):
            ax.plot(*cv.T, ls=ls, marker='',lw=lw, c=c, label=(f'${sigma:.1f} \sigma$ ' + l) if l is not None else l)
    
    return ax

def _subplot_1d(ax, param, chains, truth, labels=None, sigma=0.3, show_bands=True):
    istart = 1 if len(default_colors) - len(chains) == 1 else 0
    linestyles = default_linestyles[istart:]
    markers = default_markers[istart:]
    colors = default_colors[istart:]
    markersize=6
    lw=1.8
    
    ax.axvline(truth[param], ls='--', alpha=0.5, color='k')

    if labels is None:
        labels = len(chains)*[None]
    
    for chain,ls,c,l in zip(chains, linestyles, colors, labels):

        mean = chain.get_mean([param])[0]
        std = chain.get_std([param])[0]

        x1, x2 = ax.get_xlim()

        x = np.linspace(x1,x2, 50)
        y = chain.get_density_1d(param).Prob(x)

        ax.plot(x, y, c=c, ls=ls, label=l)

        ax.set_xlabel(fparams.param_to_latex(param))

    if show_bands:
        left, right = chains[0].get_bounds(param, sigma=sigma, maxlike=False)
        x = np.linspace(left, right, 50)
        y = chains[0].get_density_1d(param).Prob(x)

        ax.fill_between(x, y, facecolor="none", edgecolor='#000',alpha=0.1, hatch='/////')
        # ax.axvspan(left_a, right_a, facecolor="none", edgecolor='#000',alpha=0.1, hatch='/////')

    return ax

def plot_1d(param, chains, labels, truth=None, sigma=0.3, baseline=True):

    fig, axes = plt.subplots()
    fig.set_size_inches(figwidth/3, figwidth/3)

    if baseline:
        l, r = chains[0].get_bounds(param, sigma, maxlike=False)
        plt.fill_betweenx([-0.7,0.3+len(chains)], l, r, color='#eee')
    
    if truth is not None:
        plt.axvline(truth[param], c='#aaa', ls='dashed')

    colors = default_colors if baseline else default_colors[1:]

    for i,(c, (l,m,r)) in enumerate(zip(colors, [(chain.get_bounds(param, sigma, maxlike=True)) for chain in chains])):
        plt.plot([l,r], [i,i], c=c)
        plt.scatter(m, i, zorder=2,c=c)

    plt.gca().set_yticks(np.arange(len(chains)))
    plt.gca().set_yticklabels(labels)
    plt.ylim(-0.3 + len(chains),-0.7)
    plt.xlabel(fparams.param_to_latex(param))
    
    return fig

def plot_1ds(chains, params, labels, sigma=1, figsize=(6,3)):

    fig = plt.figure(figsize=figsize)
    axes = fig.add_gridspec(1, len(params), hspace=0, wspace=0).subplots(sharey='row')

    colors = default_colors[1:]

    for param, ax in zip(params, axes):
        ax.tick_params(length=0)
        ax.axvline(0, c='#aaa', ls='dashed')    

        for i,(c, (l,m,r)) in enumerate(zip(colors, [(chain.get_bounds(param, sigma, maxlike=True)) for chain in chains])):
            ax.plot([l,r], [i,i], c=c)
            ax.scatter(m, i, zorder=2,c=c)

        ax.set_yticks(np.arange(len(chains)))
        ax.set_yticklabels(labels)
        ax.set_ylim(-0.3 + len(chains),-0.7)
        ax.set_xlabel(fparams.param_to_latex(param))
    
    return fig, axes

def plot_1d_post(param, chains, labels=None, truth=None, sigma=3, figsize=(2,2)):
    fig, ax = plt.subplots(figsize=(2,2))
    
    std = chains[0].get_std([param])[0]
    mean = chains[0].get_mean([param])[0]

    x = np.linspace(mean - sigma*std, mean + sigma*std, 50)

    for chain, c, ls in zip(chains, default_colors, default_linestyles):
        ax.plot(x, chain.get_density_1d(param).Prob(x), c=c, ls=ls)

    ax.axvline(truth[param], ls='dashed', alpha=0.3)
    ax.set_xlabel(fparams.param_to_latex(param))
    return fig, ax

def get_stats(chain, is_chains, params2plot):
    output_string = ''
    params2plot = tuple(params2plot)
    for chain in is_chains:
        ESS_base = chain.base.get_ESS_dict()
        ESS_IS = chain.get_ESS_dict()

        output_string += f'\n{"#"*95}\n\n# {"-"*10} File: {chain.filename}\n'

        base_mean, base_std = chain.base.get_mean(params2plot), chain.base.get_std(params2plot)
        is_mean, is_std = chain.get_mean(params2plot), chain.get_std(params2plot)

        output_string += '\nBaseline mean ± std\n'
        for p,m,s in zip(params2plot, base_mean, base_std):
            output_string += '\t{:<40} {:7n} ± {:7n}\n'.format(p, m, s)

        output_string += '\nImportance sampled mean ± std\n'
        for p,m,s in zip(params2plot, is_mean, is_std):
            output_string += '\t{:<40} {:7n} ± {:7n}\n'.format(p, m, s)

        output_string += '\nDelta parameter/std ± 1/sqrt(ESS)\n'
        for p, bm, bs, im in zip(params2plot, base_mean, base_std, is_mean):
            output_string += '\t{:<40} ({:7.4f} ± {:6.4f}) σ\n'.format(p, (im-bm)/bs, 1/np.sqrt(ESS_IS['Euclidean distance']))

        output_string += '\n1D bias in mean (n-sigma wrt specified posterior)\n'
        for p in params2plot:
            output_string += '\t{:<40} {:8.4f}σ (baseline), {:8.4f}σ (contaminated)\n'.format(p,
                *chain.get_1d_shift(p)
            )

        output_string += '\n2D contour overlap & bias in peak posterior \n'
        param_combinations = np.array(list())
        for p in itt.combinations(params2plot, 2):
            # output_string += '\n\t{:<40} {:6.1f}% (1σ contour), {:6.1f}% (2σ contour)\n\t{:<40} {:6.4f}σ (baseline),   {:6.4f}σ (contaminated)\n'.format(
            #     p[0],
            #     chain.get_jaccard_index(1, p[0], p[1])*100,
            #     chain.get_jaccard_index(2, p[0], p[1])*100,
            #     p[1],
            #     chain.get_2d_shift(p[0], p[1]),
            #     chain.get_2d_shift(p[0], p[1], base_posterior=False)
            # )

            output_string += '\n\t{:<40}\n\t{:<40} {:6.4f}σ (baseline),   {:6.4f}σ (contaminated)\n'.format(
                p[0],
                p[1],
                chain.get_2d_shift(p[0], p[1]),
                chain.get_2d_shift(p[0], p[1], base_posterior=False)
            )

        # output_string += '\nDelta loglike (= -2*<delta chi^2>, want shifts of <~O(1))\n'
        # dl = chain.get_dloglike_stats()
        # output_string += '\tAverage: {:10.4f}\n'.format(dl[0])
        # output_string += '\tRMS:     {:10.4f}\n'.format(dl[1])
        # output_string += '\nDelta logZ:     {:10.4f}\n'.format(chain.get_delta_logz())

        output_string += '\nEffective sample sizes\n'
        for key in ESS_base.keys():
            output_string += '\t{:<30}\t{:9.2f} / {:9.2f} = {:7.4f}\n'.format(key, ESS_IS[key], ESS_base[key], ESS_IS[key]/ESS_base[key])

        output_string += '\n\tTotal samples' + ' '*27 + '{}\n'.format(chain.N)
        
    return output_string

def plot_triangle_getdist(base_chain, is_chains, extra_chains, params2plot, labels, output, fig_format='pdf', base_plot=False, classic_kde=False):
    samples = []
    settings = {'smooth_scale_1D': 0.3, 'smooth_scale_2D': 0.4} if classic_kde else None
    if base_plot:
        samples.append(base_chain.get_MCSamples(settings=settings))
    samples.extend([is_chain.get_MCSamples(settings=settings) for is_chain in is_chains])
    samples.extend([c.get_MCSamples(settings=settings) for c in extra_chains])

    g = gd.plots.getSubplotPlotter()

    g.triangle_plot(
        samples,
        params=params2plot,
        legend_labels=['Baseline'] + labels if base_plot else labels
    )

    g.export(output + '_triangle.' + fig_format)

    return g

def plot_weights(is_chains, output, fig_format='pdf'):
    for chain in is_chains:
        fig, ax = plt.subplots()
        ax.plot(chain.get_weights())
        # ax.plot(chain.base.get_weights())
        plt.savefig('{}_{}_weights.{}'.format(output, chain.name, fig_format))

def get_markdown_stats(is_chains, labels, params2plot):
    pairs = list(itt.combinations(params2plot, 2))
    output_string = '\pagenumbering{gobble}\n\n'
    output_string += '| | ' + '| '.join(['$\Delta {}/\sigma$'.format(fparams.param_to_label(p)) for p in params2plot]) + ' | ' + ('| '.join(['2D bias ${} \\times {}$'.format(*fparams.param_to_label(p)) for p in pairs]) if len(pairs) > 1 else '2D bias') + ' |\n'
    output_string += '| -: |' + ' :-: |'*(len(params2plot)+1 + len(pairs)) + '\n'
    for chain, label in zip(is_chains, labels):
        ESS_base = chain.base.get_ESS_dict()
        ESS_IS = chain.get_ESS_dict()

        base_mean, base_std = chain.base.get_mean(params2plot), chain.base.get_std(params2plot)
        is_mean, is_std = chain.get_mean(params2plot), chain.get_std(params2plot)

        biases_1d = (is_mean - base_mean)/base_std
        error_1d = 1/np.sqrt(ESS_IS['Euclidean distance'])
        biases_2d = ['${:.3f}$'.format(chain.get_2d_shift(p)) for p in pairs]

        output_string += '| {} | '.format(label) + ' |'.join(['${:+.3f} \\pm {:.3f}$'.format(b, error_1d) for b in biases_1d]) + ' | ' + ' |'.join(biases_2d) + ' |\n'

    return output_string

# def plot_summary(base_chain, is_chains, params2plot, output, fig_format='pdf'):
#     chains = [base_chain]
#     chains.extend(is_chains)
#     c = ChainConsumer()
#     for i, chain in zip(range(len(chains)), chains):
#         c.add_chain(chain.on_params(params2plot), parameters=fparams.param_to_latex(params2plot).tolist(), weights=chain.get_weights(), name='chain{}'.format(i))
#     c.plotter.plot_summary(errorbar=True, truth='chain0', include_truth_chain=True, filename='{}_summary.{}'.format(output, fig_format))

def main():
    parser = argparse.ArgumentParser(description = '')

    parser.add_argument('chain', help = 'base chain filename.')
    parser.add_argument('importance_weights', nargs='+', help = 'importance sampling weights filename.')
    parser.add_argument('output', help = 'output root.')

    parser.add_argument('--fig-format', dest = 'fig_format',
                    default = 'pdf', required = False,
                    help = 'export figures in specified format.')

    parser.add_argument('--triangle-plot', dest = 'triangle_plot', action='store_true',
                    help = 'generate triangle plots.')

    parser.add_argument('--base-plot', dest = 'base_plot', action='store_true',
                    help = 'include base chain in triangle plots.')

    parser.add_argument('--plot-weights', dest = 'plot_weights', action='store_true',
                    help = 'plot importance weights.')

    parser.add_argument('--stats', dest = 'stats', action='store_true',
                    help = 'compute importance sampling statistics.')

    parser.add_argument('--markdown-stats', dest = 'markdown_stats', action='store_true',
                    help = 'output short summary in markdown.')

    # parser.add_argument('--summary', dest = 'summary', action='store_true',
    #                 help = 'do summary plot.')

    parser.add_argument('--shift-2d', dest = 'shift_2d', action='store_true',
                    help = "compute 2d bias.")

    parser.add_argument('--plot-shifts', dest = 'plot_shifts', action='store_true',
                    help = "plot shifts.")

    parser.add_argument('--all', dest = 'all', action='store_true',
                    help = 'same as --stats --triangle-plot --base-plot.')

    parser.add_argument('--classic-kde', dest = 'classic_kde', action='store_true',
                    help = 'Use a constant KDE kernel instead of getdist optimized kernel.')

    parser.add_argument('--extra-chains', dest = 'extra_chains', nargs='*', required = False,
                    help = 'Use this to include more chains in the plots.')

    parser.add_argument('--labels', dest = 'labels', nargs='*', required = False,
                    help = 'IS chain labels.')
    
    parser.add_argument('--boosted', dest = 'boosted',  action='store_true',
                    help = 'Load the baseline chain from the polychord output files rather than cosmosis output (useful if boost_posterior=T).')
    
    parser.add_argument('--base-weight', dest = 'base_weight',
                    default = 'weight', required = False,
                    help = 'define how the baseline weights will be determined ("weight": weight, "log_weight": exp(log_weight)*old_weight, "old_weight": old_weight.')
    
    parser.add_argument('--config', dest = 'config', required = True,
                    help = 'Loads config yaml file.')

    parser.add_argument('--debug', dest = 'debug', action='store_true', default=False, required=False,
                    help = 'Increases verbosity.')
    
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    fparams.label_dict.update(config.get('param_labels', {}))

    if args.debug:
        global VERBOSE
        VERBOSE = True

    if args.all:
        args.stats = True
        args.triangle_plot = True
        args.base_plot = True
        # args.markdown_stats = True
        args.plot_shifts = True

    setup_config()
        
    base_chain = fchain.Chain(args.chain, args.boosted, args.base_weight)
    is_chains = [fchain.ImportanceChain(iw_filename, base_chain) for i, iw_filename in enumerate(args.importance_weights)]
    extra_chains = [fchain.Chain(f) for f in args.extra_chains] if args.extra_chains else []

    if args.stats:
        output_string = get_stats(base_chain, is_chains, config['params2plot'])
        if VERBOSE:
            print(output_string.replace('\n', '\r\n'))
        with open(args.output + '_stats.txt', 'w') as f:
            f.write(output_string)

    # Plot IS weights
    if args.plot_weights:
        plot_weights(is_chains, output, fig_format='pdf')

    # Make triangle plot
    if args.triangle_plot:
        plot_triangle_getdist(
            base_chain,
            is_chains,
            extra_chains,
            config['params2plot'],
            args.labels,
            args.output,
            args.fig_format,
            args.base_plot,
            args.classic_kde,
            )
    
    if args.plot_shifts:
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(args.output + '_shift_plots.pdf') as pdf:
            for param1, param2, sigma in config['pairs_plot_2d']:
                fig = plot_2d(param1, param2, [base_chain, *is_chains, *extra_chains], config['truth'], ['Baseline'] + args.labels, sigma)
                pdf.savefig(fig, bbox_inches = "tight")
                plt.close()
            for param in config['params2plot']:
                fig = plot_1d(param, [base_chain, *is_chains, *extra_chains], ['Baseline'] + args.labels, config['truth'])
                fig.set_size_inches(figwidth/3, figwidth/3)
                fig.gca().set_title(r'mean $\pm\; 0.3\sigma$', fontsize=10)
                pdf.savefig(fig, bbox_inches = "tight")
                plt.close()
            

    if args.markdown_stats:
        output_string = get_markdown_stats(is_chains, args.labels, config['params2plot'])
        with open(args.output + '_stats.md', 'w') as f:
            f.write(output_string)

    # if args.summary:
    #     plot_summary(base_chain, is_chains, config['params2plot'], args.output, args.fig_format)

if __name__ == "__main__":
    main()
