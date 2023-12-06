import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats


def group_main_levels(data, df, target):
    tmp = dict(data.groupby(df)[target].mean())
    return len(tmp), list(tmp.keys()), list(tmp.values())


def plot_main_effects(data, cols, target, moi):
    fig, axes = plt.subplots(ncols=len(cols), figsize=(6*len(cols), 6))
    all_yvals = []
    for i, col in enumerate(cols):
        n, xticks, yvals = group_main_levels(data, col, target)
        all_yvals += yvals
        axes[i].plot(range(n), yvals, '-o')
        axes[i].set_xlim(-0.5, n-0.5)
        axes[i].set_xticks(range(n))
        axes[i].set_xticklabels(xticks, fontsize=12)
        # axes[i].set_ylim(5.5, 8.5)
        axes[i].set_xlabel(f"Design Factor [{col}]", fontsize=16)
        axes[i].set_ylabel(moi, fontsize=16)
    for i in range(len(cols)):
        axes[i].set_ylim(min(all_yvals)*0.98, max(all_yvals)*1.02)
    plt.show()


def interaction_combinations(cols):
    combinations = []
    for _ in range(len(cols)):
        target = cols.pop()
        for val in cols:
            combinations.append([target, val])
        cols.insert(0, target)
    return combinations


def plot_interaction_effects(data, cols, target, moi):
    n = len(cols)
    fig, axes = plt.subplots(nrows=n, ncols=n-1, figsize=(4 * n, 5 * (n-1)))
    axes = axes.flatten()
    linestyles = ['-', '--', ':']
    vals = []
    for j, vars in enumerate(interaction_combinations(cols)):
        var1, var2 = vars
        grouped = data.groupby([var1, var2])[target] \
            .mean().unstack().T
        vals += list(grouped.values.flatten())
        for i, val in enumerate(grouped.index):
            axes[j].plot(grouped.loc[val], ls=linestyles[i], label=val)
        axes[j].set_xlabel(var1, fontsize=12)
        axes[j].set_ylabel(moi, fontsize=12)
        axes[j].legend(title=var2)
    for ax in axes:
        ax.set_ylim(min(vals)*0.98, max(vals)*1.02)
    plt.tight_layout()
    plt.show()


def create_reg_str(reg_cols, y_col='time', interactions=False):
    sep = ' * ' if interactions else ' + '
    return f"{y_col} ~ " + sep.join([f"C({col})" for col in reg_cols])


def students_t_test(mu1, mu2, n1, n2, s1, s2):
    """ Function to compute the students t stat
        and corresponding degrees of freedom """
    print("Used Student's t-test to compute p-value\n")
    df = n1 + n2 - 2
    s = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    t_stat = (mu1 - mu2) / np.sqrt(s / n1 + s / n2)
    return t_stat, df


def welch_t_test(mu1, mu2, n1, n2, s1, s2):
    """ Function to compute the welch's t stat
        and corresponding degrees of freedom """
    print("Used Welch's t-test to compute p-value\n")
    df_num = np.power(s1 / n1 + s2 / n2, 2)
    df_den = (np.power(s1 / n1, 2) / (n1 - 1)
              + np.power(s2 / n2, 2) / (n2 - 1))
    df = df_num / df_den
    t_stat = (mu1 - mu2) / np.sqrt(s1 / n1 + s2 / n2)
    return t_stat, df


def determine_t_test(v1, v2, n1, n2, alpha=0.05):
    """ Function that determines if a students t-test is appropriate
        using an variance F-test. Returns appropriate function """
    t = v1 / v2 if v1 > v2 else v2 / v1
    df = {'dfn': n1, 'dfd': n2}
    pv = stats.f.cdf(1 / t, **df) + stats.f.sf(t, **df)
    print(f"Variance F-test p-value = {pv}")
    return students_t_test if pv >= alpha else welch_t_test


def hypothesis_test(data, col, val1, val2, alternate,
                    alpha=0.05, target='time'):
    """ Function to perform t-test hypothesis testing on
        provided column and unique column values. """
    # Print diagnostic message
    print(f"Info:\t[{col}, {val1}, {val2}, {alternate}]")
    # Extract relevant mu, sigma, n values from data
    grouped = data.groupby(col)[target] \
                  .agg(['mean', 'count', np.var]).T.to_dict()
    mu1, n1, s1 = grouped[val1].values()
    mu2, n2, s2 = grouped[val2].values()
    # Determine appropriate t-test & compute t-stat / df
    t_test = determine_t_test(s1, s2, n1, n2, alpha)
    t_stat, df = t_test(mu1, mu2, n1, n2, s1, s2)
    # Compute the appropriate p-value from the alternate
    if alternate == 'greater':
        pv = stats.t.sf(t_stat, df)
    elif alternate == 'less':
        pv = stats.t.cdf(t_stat, df)
    elif alternate == 'equal':
        pv = (stats.t.cdf(-abs(t_stat), df)
              + stats.t.sf(abs(t_stat), df))
    # Return the t-stat and p-value
    return t_stat, pv


def add_bonferroni_adjusted_pv(results):
    """ Function to add bonferroni adjusted p-values """
    M = len(results)
    def bonferroni(p): return p * M
    results['p* [Bonferroni]'] = results['p-value'].apply(bonferroni)


def add_sidak_adjusted_pv(results):
    """ Function to add sidak adjusted p-values """
    M = len(results)
    def sidak(p): return 1 - (1 - p) ** M
    results['p* [Sidak]'] = results['p-value'].apply(sidak)


def add_holm_adjusted_pv(results):
    """ Function to add holm adjusted p-values """
    pvs = results['p-value'].values
    M = len(pvs)
    results['p* [Holm]'] = [
        max(pvs[:i + 1] * (M - np.arange(i + 1)))
        for i in range(M)
    ]


def pairwise_hypothesis_testing(data, col, pairs, alternates, target, alpha=0.05):
    """ Function to perform pairwise hypothesis testing
        on a column given a list of pair values and alternatives.
        Computes various adjusted p-values as well """
    # Perform pairwise hypothesis tests and compute t / p-vals
    results = []
    result_cols = ['Column', 'Value 1', 'Value 2',
                   'Alternate', 't-stat', 'p-value']
    for pair, alternate in zip(pairs, alternates):
        t, pv = hypothesis_test(data, col, *pair, alternate, alpha, target)
        results.append([col, *pair, alternate, t, pv])
    # Convert results to df an calc adjusted p-values
    results = pd.DataFrame(results, columns=result_cols) \
                .sort_values('p-value').reset_index(drop=True)
    add_bonferroni_adjusted_pv(results)
    add_sidak_adjusted_pv(results)
    add_holm_adjusted_pv(results)
    return results


def format_data_for_pairwise(data, cols, target):
    conditions = []
    for i in range(len(data)):
        condition = []
        for val in data.loc[i, cols].items():
            condition.append(f"[{': '.join([str(v) for v in val])}]")
        conditions.append(' '.join(condition))
    return pd.DataFrame.from_dict({
        'Condition': conditions,
        target: data[target]
    })


def run_min_condition_validation(data, cols, target, n=None):
    pairwise_data = format_data_for_pairwise(data, cols, target)
    pairs = []
    alternates = []
    conditions = pairwise_data.groupby('Condition').mean() \
        .sort_values(target).index.values
    n = n if n is not None else len(np.unique(pairwise_data['Condition']))
    for condition in conditions[1: n + 1]:
        pairs.append([conditions[0], condition])
        alternates.append('less')
    return pairwise_hypothesis_testing(
        pairwise_data, 'Condition', pairs, alternates, target
    ).iloc[::-1].reset_index(drop=True)
