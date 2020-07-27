# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# %%
def bland_altman_plot(
    predicted, truth, percent=False, title="Bland-Altman Plot"
):
    predicted = np.asarray(predicted)
    truth = np.asarray(truth)
    diff = truth - predicted

    if percent:
        diff = np.divide(diff, truth)

    fig, ax = plt.subplots()
    ax = sns.scatterplot(truth, diff)
    ax.set(
        xlabel="truth", ylabel="difference from truth", title=title,
    )

    # Plot a horizontal line at 0
    ax.axhline(0, ls=":", c=".2")

    return ax


# %%
def scatter_plot(metric, truth, title="Scatter Plot"):
    metric = np.asarray(metric)
    truth = np.asarray(truth)

    fig, ax = plt.subplots()
    ax = sns.scatterplot(truth, metric)
    ax.set(
        xlabel="GT", ylabel="Metric", title=title,
    )

    # Plot a horizontal line at 0
    ax.axhline(0, ls=":", c=".2")

    return ax


# %%
def sample_plot():
    sample_x = np.random.rayleigh(scale=10, size=201)
    sample_y = np.random.normal(size=len(sample_x)) + 10 - sample_x / 10.0

    bland_altman_plot(sample_y, sample_x)

    plt.show()


# %%
