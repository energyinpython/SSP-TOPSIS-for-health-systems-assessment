import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ssp_topsis import SSP_TOPSIS

from pyrepo_mcda import normalizations as norms
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import weighting_methods as mcda_weights

from pyrepo_mcda.mcda_methods import TOPSIS

import copy
import seaborn as sns
from matplotlib.pyplot import cm

def plot_barplot(df_plot, legend_title='Weighting methods'):
    """
    Visualization method to display column chart of alternatives rankings obtained with 
    different methods.

    Parameters
    ----------
        df_plot : DataFrame
            DataFrame containing rankings of alternatives obtained with different methods.
            The particular rankings are included in subsequent columns of DataFrame.

        title : str
            Title of the legend (Name of group of explored methods, for example MCDA methods or Distance metrics).

    Examples
    ----------
    >>> plot_barplot(df_plot, legend_title='MCDA methods')
    """
    

    ax = df_plot.plot(kind='bar', width = 0.8, stacked=False, edgecolor = 'black', figsize = (9,4))
    ax.set_xlabel('Evaluation criteria', fontsize = 12)
    ax.set_ylabel('Weight', fontsize = 12)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 12)
    y_ticks = ax.yaxis.get_major_ticks()
    

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=3, mode="expand", borderaxespad=0., edgecolor = 'black', title = legend_title, fontsize = 12)

    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    legend_title = legend_title.replace("$", "")
    legend_title = legend_title.replace("{", "")
    legend_title = legend_title.replace("}", "")
    plt.savefig('./results/' + 'bar_chart_' + legend_title + '.pdf', bbox_inches = 'tight')
    plt.show()


def main():

    # Load decision matrix with performance values
    dataset = pd.read_csv('dataset.csv', index_col='Country')

    df = dataset.iloc[:len(dataset) - 1, :]

    types = dataset.iloc[len(dataset) - 1, :].to_numpy()

    matrix = df.to_numpy()
    types = np.ones(matrix.shape[1])

    # weights = mcda_weights.entropy_weighting(matrix)
    # saved_weights = copy.deepcopy(weights)
    # weights = mcda_weights.gini_weighting(matrix)
    # saved_weights = np.vstack((saved_weights, weights))
    # weights = mcda_weights.idocriw_weighting(matrix, types)
    # saved_weights = np.vstack((saved_weights, weights))
    # weights = mcda_weights.cilos_weighting(matrix, types)
    # saved_weights = np.vstack((saved_weights, weights))
    # weights = mcda_weights.angle_weighting(matrix, types)
    # saved_weights = np.vstack((saved_weights, weights))

    weights = mcda_weights.critic_weighting(matrix)

    # saved_weights = np.vstack((saved_weights, weights))
    # wm =  ['Entropy', 'Gini', 'IDOCRIW', 'CILOS', 'Angular', 'CRITIC']
    # list_cols = [r'$C_{' + str(i + 1) + '}$' for i in range(15)]
    # df_saved_weights = pd.DataFrame(data = saved_weights, columns = list_cols)
    # df_saved_weights.index = wm
    # df_saved_weights.index.name = 'Weighting methods'
    # plot_barplot(df_saved_weights.T)

    names = list(df.index)

    results_pref = pd.DataFrame(index=names)
    results_rank = pd.DataFrame(index=names)

    topsis = TOPSIS(normalization_method=norms.minmax_normalization)
    pref_t = topsis(matrix, weights, types)
    results_pref['TOPSIS'] = pref_t
    rank_t = rank_preferences(pref_t, reverse=True)
    results_rank['TOPSIS'] = rank_t


    toss = SSP_TOPSIS(normalization_method=norms.minmax_normalization)

    # sustainability coefficient from matrix calculated based on standard deviation from normalized matrix
    n_matrix = norms.minmax_normalization(matrix, types)
    s = np.sqrt(np.sum(np.square(np.mean(n_matrix, axis = 0) - n_matrix), axis = 0) / n_matrix.shape[0])

    pref = toss(matrix, weights, types, s_coeff = s)
    results_pref['SSP-TOPSIS std'] = pref
    rank = rank_preferences(pref, reverse = True)
    results_rank['SSP-TOPSIS std'] = rank

    
    # analysis with sustainability coefficient modification
    model = [
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8],
        [9, 10],
        [11, 12, 13, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    ]


    #
    # analysis performed for table
    for el, mod in enumerate(model):
        new_s = np.zeros(matrix.shape[1])
        new_s[mod] = s[mod]

        pref = toss(matrix, weights, types, s_coeff = new_s)
        results_pref['SSP-TOPSIS ' + r'$G_{' + str(el + 1) + '}$'] = pref
        rank = rank_preferences(pref, reverse = True)
        results_rank['SSP-TOPSIS ' + r'$G_{' + str(el + 1) + '}$'] = rank

    results_pref = results_pref.rename_axis('Country')
    results_rank = results_rank.rename_axis('Country')
    results_pref.to_csv('./results/df_pref_G' + '.csv')
    results_rank.to_csv('./results/df_rank_G' + '.csv')

    #
    # color = []
    # for i in range(9):
    #     color.append(cm.Set1(i))
    # for i in range(8):
    #     color.append(cm.Set2(i))
    # for i in range(10):
    #     color.append(cm.tab10(i))
    # for i in range(8):
    #     color.append(cm.Pastel2(i))
    # plt.style.use('seaborn')
    # analysis performed for figures
    sust_coeff = np.arange(0, 1.1, 0.1)

    for el, mod in enumerate(model):
        results_pref = pd.DataFrame(index=names)
        results_rank = pd.DataFrame(index=names)

        for sc in sust_coeff:

            s = np.zeros(matrix.shape[1])
            s[mod] = sc

            pref = toss(matrix, weights, types, s_coeff=s)
            rank = rank_preferences(pref, reverse = True)

            results_pref[str(sc)] = pref
            results_rank[str(sc)] = rank


        results_pref = results_pref.rename_axis('Country')
        results_rank = results_rank.rename_axis('Country')
        results_pref.to_csv('./results/df_pref_sust_G' + str(el + 1) + '.csv')
        results_rank.to_csv('./results/df_rank_sust_G' + str(el + 1) + '.csv')

        # plot results of analysis with sustainabiblity coefficient modification
        ticks = np.arange(1, matrix.shape[0])

        x1 = np.arange(0, len(sust_coeff))

        plt.figure(figsize = (10, 6))
        for i in range(results_rank.shape[0]):
            plt.plot(x1, results_rank.iloc[i, :], '.-', linewidth = 2)
            ax = plt.gca()
            y_min, y_max = ax.get_ylim()
            x_min, x_max = ax.get_xlim()
            plt.annotate(names[i], (x_max, results_rank.iloc[i, -1]),
                            fontsize = 12, #style='italic',
                            horizontalalignment='left')

        plt.xlabel("Sustainability coeffcient", fontsize = 12)
        plt.ylabel("Rank", fontsize = 12)
        plt.xticks(x1, np.round(sust_coeff, 2), fontsize = 12)
        plt.yticks(ticks, fontsize = 12)
        plt.xlim(x_min - 0.2, x_max + 1.8)
        plt.gca().invert_yaxis()
        
        plt.grid(True, linestyle = '--')
        if el < 4:
            plt.title(r'$G_{' + str(el + 1) + '}$')
        else:
            plt.title('All criteria')
        plt.tight_layout()
        plt.savefig('./results/rankings_sust_G' + str(el + 1) + '.pdf')
        plt.show()
    


if __name__ == '__main__':
    main()