"""
Playground for feature selection.

This script is used to generate various plot regardin the feature selection
pipeline (correlation matrix plot, histogram of correlations, explained
variance for different values of retained components in PCA, etc).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ddm_project.ml.feature_generation import FeatureGenerator
from ddm_project.readers.nab_dataset_reader import NABReader

register_matplotlib_converters()

reader = NABReader()
reader.load_data()
reader.load_labels()

# dataset_names = ["iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
#                  "machine_temperature_system_failure.csv",
#                  "ec2_cpu_utilization_fe7f93.csv",
#                  "rds_cpu_utilization_e47b3b.csv",
#                  "grok_asg_anomaly.csv",
#                  "elb_request_count_8c0756.csv"]
dataset_name = "grok_asg_anomaly.csv"

df = reader.data.get(dataset_name)
labels = reader.labels.get(dataset_name)

scaler = StandardScaler()

feature_generator = FeatureGenerator(
    df, ts_col='value', scaler=scaler,
    fname=dataset_name.replace(".csv", ".pkl")
)

X = feature_generator.get(read=True)

print(X.shape)
# raise SystemExit

# prefix = "value__"
# for c in X.columns:
#     if type(c) == str and c.startswith(prefix):
#         c = c[len(prefix):]
#     print(c)
plot_correlation = False
if plot_correlation:
    # Plot histogram of correlations
    corrmat = X.corr(method='pearson')
    tril_idx = np.tril_indices_from(corrmat, -1)
    # hist, bins = np.histogram(
    #   corrmat.values[tril_idx], range=(-1, 1), bins=100)
    # plt.bar(bins[:-1], hist)  <- occhio non funziona
    values = corrmat.values[tril_idx].copy()

    ret = plt.hist(values, bins=500,
                   range=(-1, 1), label="before pca")
    # plt.figure()
    # values = values[~np.isnan(values)]
    # ret = plt.hist(values, bins=500, density=True, <- non cambia nulla
    #                range=(-1, 1), label="before pca")
    plt.show()

    # Plot correlation histogram after pca
    # for n in [30, 40, 50, 60, 80, 100, 150]:
    #     plt.figure()
    #     pca = PCA(n_components=n)
    #     X_pca = pd.DataFrame(pca.fit_transform(X), index=X.index)
    #     corrmat = X_pca.corr(method='pearson')
    #     tril_idx = np.tril_indices_from(corrmat, -1)
    #     plt.hist(corrmat.values[tril_idx], bins=100,
    #              label="after pca({})".format(n))
    #     plt.legend()
    #     plt.show()

    raise SystemExit

generate_high_corr_features = False
# Generate a list of tuples with names of features with high correlation, in
# the form of [(feat, other_feat, corr_value), ...]
if generate_high_corr_features:
    corrmat = np.abs(corrmat) * 100
    corrmat.fillna(-1., inplace=True)
    corrmat = corrmat.astype(int)
    tril = np.tril(corrmat, -1)
    tril = np.where(tril > 0, tril, np.nan)
    tril = pd.DataFrame(tril, index=corrmat.index, columns=corrmat.columns)
    thresh = 90
    high_corr_features = []
    for i, (name, col) in enumerate(tril.iteritems()):
        hc = col[col >= thresh]
        for feat_name, value in hc.iteritems():
            if name != feat_name:
                if name < feat_name:
                    high_corr_features.append((name, feat_name, value))
                else:
                    high_corr_features.append((feat_name, name, value))

    # This was to make sure that no duplicate were considered.
    # counts = []
    # for f1, f2, value in high_corr_features:
    #     c = 0
    #     for other in high_corr_features:
    #         if other[0] == f1 and other[1] == f2:
    #             c += 1
    #     counts.append((f1, f2, c))


# These were to plot a tringular verison of the correlation matrix.
# plt.plot(corrmat.columns, col.where(col > 0.9))
# plt.xticks(rotation=90, size=5)

# plt.show()

# corrmat = np.abs(corrmat) * 100
# corrmat.fillna(-1., inplace=True)
# corrmat = corrmat.astype(int)
# high_corr = corrmat[corrmat > 50]
# tril = np.tril(high_corr)
# tril_mask = np.where(tril > 0, False, True)
# sns.heatmap(high_corr, square=True, mask=tril_mask,
#             xticklabels=False, yticklabels=False)
# plt.show()

correlation_matrix = False
if correlation_matrix:
    corrmat = X.corr(method='pearson')
    corrmat = np.abs(corrmat)
    #     sns.heatmap(
    #     corrmat * 100,
    #     cbar=False,
    #     annot=True,
    #     xticklabels=False,
    #     yticklabels=False,
    #     square=False,
    #     fmt=".0f",
    #     annot_kws={"size": 4},
    # )
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # plt.show()
    # plt.savefig("heatmap.svg", transparent=True,
    #             frameon=False, bbox_inches='tight', pad_inches=0)
    ax = sns.heatmap(corrmat, square=True,
                     xticklabels=False, yticklabels=False)
    nan_mask = corrmat.where(corrmat.isna())  # df with all elements = nan
    nan_mask = nan_mask.mask(corrmat.isna(), 1)
    sns.heatmap(nan_mask, square=True, xticklabels=False, cbar=False,
                cmap=['#757575'], yticklabels=False, ax=ax)
    plt.show()

# kpca = KernelPCA(kernel='rbf')
# kpca_fit = kpca.fit(X)
# lambdas = kpca_fit.lambdas_
# kpca_var = lambdas.cumsum() / lambdas.sum()
# for i, v in enumerate(kpca_var):
#     print("{:3d}: {:.4f}".format(i, v))

pca = PCA()
pca_fit = pca.fit(X)
print("Explained Variance:")
var = pca_fit.explained_variance_ratio_
cum_var = var.cumsum()
max_idx = len(var)
var = var[:max_idx]
cum_var = cum_var[:max_idx]
fig, ax = plt.subplots()
ax.bar(range(len(var)), var)
ax.plot(cum_var)

var_thresholds = [0.90, 0.93, 0.95, 0.97, 0.99]
for i, thresh in enumerate(var_thresholds):
    try:
        n_comp = np.argwhere(cum_var > thresh).ravel()[0]
    except IndexError:
        import warnings
        warnings.warn("IndexError")
        continue
    ax.axhline(thresh, linestyle='--', alpha=0.4, color="C{}".format(i),
               label="{:.0f}%-{}".format(thresh * 100, n_comp))

plt.legend()
plt.show()


# print(pca_fit.components_)

# Plot 3D surface
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# l = corrmat.shape[0]
# print(corrmat.shape)
# x = y = np.arange(l)
# X, Y = np.meshgrid(x, y)
# print(X.shape)
# from matplotlib import cm
# surf = ax.plot_surface(X, Y, corrmat, cmap='viridis',
#                        linewidth=0, antialiased=False)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()
