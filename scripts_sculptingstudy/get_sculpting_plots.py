import numpy as np
from matplotlib import pyplot as plt
from os.path import join
import matplotlib
from matplotlib.cm import get_cmap

viridis = get_cmap("viridis")
blues = get_cmap("Blues")
oranges = get_cmap("Oranges")
colors = viridis(np.linspace(0, 1, 5))
colors_data = blues(np.linspace(0.3, 1, 5)[::-1])
colors_samps = oranges(np.linspace(0.3, 1, 5)[::-1])

base_dir = "../results_sculptingstudy"

eff_list = [1.0, 0.2, 0.05, 0.01, 0.001]
data_dict = dict()
samps_dict = dict()
data_mjj_dict = dict()
samps_mjj_dict = dict()

for eff in eff_list:
    data_dict[eff] = []
    samps_dict[eff] = []

masses = np.arange(3200, 4600, 200)

for mass in masses:
    for idx, eff in enumerate(eff_list):
        if eff == 1.0:
            mjj_data = np.load(join(base_dir, f"sr_m{mass}",
                                    "data_full.npy"))[:, 0]
            mjj_samps = np.load(join(base_dir, f"sr_m{mass}",
                                     "samps_full.npy"))[:, 0]
        else:
            mjj_data = np.load(join(base_dir, f"sr_m{mass}",
                                    f"mjj_data_eff{eff}.npy"))
            mjj_samps = np.load(join(base_dir, f"sr_m{mass}",
                                     f"mjj_samps_eff{eff}.npy"))

        data_dict[eff].append(mjj_data.shape[0])
        samps_dict[eff].append(mjj_samps.shape[0])
        data_mjj_dict[eff] = mjj_data
        samps_mjj_dict[eff] = mjj_samps

for eff in eff_list:
    data_dict[eff] = np.array(data_dict[eff])
    samps_dict[eff] = np.array(samps_dict[eff])

for idx, eff in enumerate(eff_list):
    if idx == 0:
        a1 = plt.axes([0.0, 0.22, 1.0, 0.8])
        a2 = plt.axes([0.0, 0.0, 1.0, 0.16], sharex=a1)
        tmp_label = "expected"
    else:
        tmp_label = None

    plt.sca(a1)
    plt.errorbar(masses, data_dict[eff], color="black", ds='steps-mid',
                 linestyle="dashed", label=tmp_label)

    plt.errorbar(masses, samps_dict[eff], fmt="o",
                 label=f"{eff*100:.1f}%", color=colors[idx],
                 yerr=np.sqrt(samps_dict[eff]))

    plt.sca(a2)
    ratio_samps_data = ((samps_dict[eff] - data_dict[eff])
                        / np.sqrt(data_dict[eff]))

    plt.errorbar(masses, ratio_samps_data, fmt='o', yerr=None, capthick=0.4,
                 capsize=0, elinewidth=0.8, markersize=4, color=colors[idx])

plt.sca(a1)
plt.tick_params(direction="in", which="both")
plt.yscale("log")
plt.ylabel("Events")
plt.ylim(0.1, 1.0e5)
y_major = matplotlib.ticker.LogLocator(base=10.0, numticks=10)
a1.yaxis.set_major_locator(y_major)
y_minor = matplotlib.ticker.LogLocator(base=10.0,
                                       subs=np.arange(1.0, 10.0)*0.1,
                                       numticks=10)

a1.yaxis.set_minor_locator(y_minor)
a1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a1.set_xlim(np.min(masses)-50, np.max(masses)+50)
a1.set_ylim(1, 1e6)
plt.legend(ncol=2, frameon=False, loc="upper right")

plt.sca(a2)
one_sigma_x = np.arange(np.min(masses), np.max(masses)+200, 200)
plt.plot(one_sigma_x, np.full_like(one_sigma_x, 1), color="black",
         linestyle="dashed")

plt.plot(one_sigma_x, np.full_like(one_sigma_x, -1), color="black",
         linestyle="dashed")

a2.set_ylabel(
    r"$\frac{\mathrm{Samples} - \mathrm{Data}}{\sigma_{\mathrm{Data}}}$")

a2.set_ylim(-5, 5)
ticks = a1.get_xticks()
a2.set_xticks(ticks)
a1.set_xlim(np.min(masses) - 50, np.max(masses) + 50)
a2.set_xlim(a1.get_xlim())
a1.tick_params(axis='x', which='both', bottom=False, top=False,
               labelbottom=False)

a2.tick_params(axis="both", which="both", direction="in")
plt.grid(True, linestyle='dashed')
a2.set_xlabel(r"$m_{jj}$")
plt.savefig(join(base_dir, "data_samps_cut_compare.pdf"),
            dpi=300, bbox_inches="tight")

plt.close()
print("Done!")
