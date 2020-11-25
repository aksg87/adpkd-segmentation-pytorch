# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score

# %%
def bland_altman_plot(
    predicted, truth, percent=True, title="Bland-Altman Plot"
):
    predicted = np.asarray(predicted)
    truth = np.asarray(truth)
    diff = (truth - predicted)
    diff = np.divide(diff, truth)

    fig, ax = plt.subplots()
    ax = sns.scatterplot(truth, 100*diff)
    ax.set(
        xlabel="ground truth (mL)", ylabel="%Difference from truth", title=title
    )

    avg = 100*np.mean(diff)
    sd_1_96 = 100*(diff.std(ddof=1) * 1.96)
    print(f"mid {avg}  sd {sd_1_96}")
    ax.axhline(avg, c=".2", ls='dashed', label=f"mean ({avg:.1f}%)")
    ax.axhline(avg + sd_1_96, ls=":", c=".2", label=f"± 1.96 std (±{sd_1_96:.1f}%)")
    ax.axhline(avg - sd_1_96, ls=":", c=".2")
    ax.set_ylim(-20, 20)

    ax.legend()
    plt.legend(loc='lower right', fontsize='10')

    return ax


# %%
def scatter_plot(metric, truth, title="Scatter Plot"):
    metric = np.asarray(metric)
    truth = np.asarray(truth)

    fig, ax = plt.subplots()
    ax = sns.scatterplot(truth, metric)
    ax.set(
        xlabel="ground truth (mL)", ylabel="Dice Metric", title=title,
    )

    avg = np.mean(metric)
    # Plot a horizontal line at 0
    ax.axhline(avg, c=".2", ls='dashed', label=f"mean ({avg:.2f})")

    sd_1_96 = metric.std(ddof=1) * 1.96
    ax.axhline(avg+sd_1_96, ls=":", c=".2", label=f"± 1.96 std (±{sd_1_96:.2f})")
    ax.axhline(avg-sd_1_96, ls=":", c=".2")
    ax.set_ylim(0, 1)

    ax.legend()
    plt.legend(loc='lower right', fontsize='10')

    return ax


# %%

def linreg_plot(pred, truth, title="Predicted vs Ground truth"):
    truth = truth.astype(float)
    pred = pred.astype(float)

    model = np.polyfit(truth, pred, 1)
    predict = np.poly1d(model)
    r2 = r2_score(pred, predict(truth))

    fig, ax = plt.subplots()
    ax = sns.regplot(truth, pred)
    ax.set(
        xlabel="ground truth (mL)", ylabel="predicted (mL)", title=title,
    )

    ax.legend()
    ax.set_xlim(0, 4e6)
    ax.set_ylim(0, 4e6)
    plt.legend(loc='lower right', fontsize='10', labels=[f'R-Sq={r2:.3f}'])
# %%
def sample_plot():
    sample_x = np.random.rayleigh(scale=10, size=201)
    sample_y = np.random.normal(size=len(sample_x)) + 10 - sample_x / 10.0

    bland_altman_plot(sample_y, sample_x)

    plt.show()


# %%
