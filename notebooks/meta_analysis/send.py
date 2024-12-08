# %%
import json
import os
import scipy

import pandas as pd
import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
dataset_path = "../../dataset/new"

# %%
def check_dataset(dataset_folder_path):
    folders_in_dataset = set(os.listdir(dataset_folder_path))

    # info print
    print(f"Dataset has {len(folders_in_dataset)} repos in total")
    return None

check_dataset(dataset_path)

# %%
df = pd.read_csv(r"..\..\dataset\new\python-poetry_poetry_metrics.csv")
def df_transformation(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(["Unnamed: 0", "Name", "LongName", "Parent", "Component", "Path", "Line", "Column", "EndLine", "EndColumn", "ID"],
            axis=1, inplace=True)
    return df
trans = df_transformation(df)
columns_list = list(df.columns)

# exclude the 0 variance columns that happen in all repos
global_columns_with_variance = set()
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        df = pl.read_csv(file_path, schema_overrides={col: pl.Float32 for col in columns_list})
        df = df.drop(["", "ID","Name","LongName","Parent","Component","Path","Line","Column","EndLine","EndColumn"])
        for col in columns_list:
            if df[col].var() > 0:
                global_columns_with_variance.add(col)
METHOD_METRICS_LIST = list(global_columns_with_variance)
zero_col_list = [metric for metric in columns_list if metric not in METHOD_METRICS_LIST]

# %%
def get_datasets_stats(folder_path, class_stats=True):
    if class_stats:
        suffix = "-Class.csv"
    else:
        suffix = "_metrics.csv"
    results = []
    columns_list = [x for x in (CLASS_METRICS_LIST if class_stats else METHOD_METRICS_LIST)]
    print(columns_list)
    num_columns = len(columns_list)
    print(num_columns)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(suffix):
                file_path = os.path.join(root, file)
                df = pl.read_csv(file_path, columns=columns_list, dtypes=[pl.Float32]*num_columns)
                if len(df) == 0:
                    print(f"Skipping empty {file}")
                    continue
                repo_name = file.split('_metrics')[0] # only repo name
                nulls_list = df.null_count().row(0)
                try:
                    nans_list = df.select(pl.all().is_nan().sum()).row(0)
                except Exception as e:
                    print(repo_name)
                    print(df.schema)
                    print(e)
                results.append((repo_name, *nulls_list, *nans_list))

    null_colnames = [x+"_nulls" for x in columns_list]
    nan_colnames = [x+"_NANs" for x in columns_list]
    df_schema = ["repo_name", *null_colnames, *nan_colnames]
    res_df = pl.DataFrame(results, schema=df_schema)
    return res_df

# %% [markdown]
# ### Method metrics null analysis

# %%
df_null_stats_methods = get_datasets_stats(dataset_path, class_stats=False)

# %%
df_null_stats_methods

# %%
# sum over columns, exclude first reponame column, sum
max_sum = df_null_stats_methods.select(
    pl.all().exclude("repo_name").sum()
).row(0)[0]
max_sum

# %% [markdown]
# ### Calculating correlations

# %%
def calculate_and_save_per_repo_correlations(folder_path):
    suffix = "_metrics.csv"
    columns_list = [x for x in METHOD_METRICS_LIST]
    num_columns = len(columns_list)
    correlations = []
    lengths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(suffix):
                file_path = os.path.join(root, file)
                df = pl.read_csv(file_path, columns=columns_list, schema_overrides={col: pl.Float32 for col in columns_list})
                df = df.with_columns(
                    [pl.col(col).fill_nan(0).alias(col) for col in df.columns]
                )
                df = df.with_columns(
                    [pl.col(col).fill_null(0).alias(col) for col in df.columns]
                )
                # df = df.select([col for col in df.columns if df[col].var() > 0])
                df = df.with_columns(
                    [pl.col(col) + np.random.normal(0, 1e-6, len(df)) for col in df.columns]
                )
                if len(df) == 0:
                    print(f"Skipping empty {file}")
                    continue
                if len(df) == 1:
                    print(f"Skipping oneline {file}")
                    continue
                repo_name = file.split('_metrics')[0] # only repo name
                try:
                    df_corr = df.corr().to_pandas() #.fill_nan(0).cast(pl.Float32) # XXX
                    filepath = f"{root}/correlations{suffix}"
                    df_corr.to_csv(filepath)
                    correlations.append(df_corr)
                    lengths.append(df.shape[0])
                except Exception as e:
                    print(repo_name)
                    print(df_corr)
                    # print(df.schema)
                    print(e)

    return correlations, lengths

# %%
correlations_df_list, lengths_list = calculate_and_save_per_repo_correlations(dataset_path)

# %%
cols = correlations_df_list[0].columns

for df in correlations_df_list:
    for col in df.columns:
        if col not in cols:
            print(col)
            print(df.head(5))
            break

# %%
correlations_df_list[5].head()


# %%
correlations_df_list[0].columns

# %%
lengths_list[0]

# %%
def dataset_correlations(folder_path):
    suffix = "_metrics.csv"
    columns_list = [x for x in METHOD_METRICS_LIST]
    num_columns = len(columns_list)
    correlations = []
    lengths = []
    df_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(suffix):
                file_path = os.path.join(root, file)
                df = pl.read_csv(file_path, columns=columns_list, schema_overrides={col: pl.Float32 for col in columns_list})
                if len(df) == 0:
                    print(f"Skipping empty {file}")
                    continue
                if len(df) == 1:
                    print(f"Skipping oneline {file}")
                    continue
                df_list.append(df)

    df_all = pl.concat(df_list)
    return df_all

# %%
df_all_dataset = dataset_correlations(dataset_path)
df_all_dataset

# %%
[col for col in df_all_dataset.columns if df_all_dataset[col].n_unique() == 1]

# %%
df_all_dataset = df_all_dataset.select([col for col in df_all_dataset.columns if df_all_dataset[col].std() > 0])

df_all_corr = df_all_dataset.corr()

# %%
output_dir = 'outputs'
file_path = os.path.join(output_dir, 'method-all-correlations.csv')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_all_corr.write_csv(file_path)
print(f"File saved successfully to {file_path}")

# %%
df_all_corr

# %%
correlations_rows = (
    df_all_dataset.corr()
    .with_columns(pl.Series(name="index", values=df_all_dataset.columns))
    .unpivot(index = "index")
    .filter(pl.col('index') != pl.col('variable'))
)
correlations_rows

# %%
correlations_rows_important = correlations_rows.filter(pl.col('value').abs() > 0.7).sort(by='value', descending=True).gather_every(2)
correlations_rows_important.write_csv('outputs/method-all-important.csv')
correlations_rows_important

# %% [markdown]
# ### Meta Analysis

# %%
def r_to_z(r_colname) -> pl.Expr:
    return (0.5 * pl.Expr.log( (1 + pl.col(r_colname)) / (1 - pl.col(r_colname))))


def variance_z(n_colname) -> pl.Expr:
    return 1/(pl.col(n_colname) - 3)

def z_to_r(z):
    return (np.exp(2*z)-1)/(np.exp(2*z)+1)


def std_z(n_colname) -> pl.Expr:
    return pl.Expr.sqrt(variance_z(n_colname))


def making_fixed_effect_df(df):
    df_fixed = df.with_columns(
        effect_size_Y=r_to_z('correlation'),
        variance_within_V=variance_z('n'),
    ).with_columns(
        raw_weight_W=1/pl.col('variance_within_V')
    ).with_columns(
        WY=(pl.col('raw_weight_W') * pl.col('effect_size_Y')),
        WY_2=(pl.col('raw_weight_W') * (pl.col('effect_size_Y'))**2),
        W_2=(pl.col('raw_weight_W')**2),
    )
    return df_fixed

def fixed_effect(df, alpha=0.05):
    """
    Fixed-effect meta-analysis on given correlations polars.DataFrame.
    """
    fixed_effect_df = making_fixed_effect_df(df)
    fixed_effect_z = fixed_effect_df.select(pl.sum('WY')/pl.sum('raw_weight_W')).item()
    fixed_variance_z = fixed_effect_df.select(1/pl.sum('raw_weight_W')).item()
    fixed_std_z = np.sqrt(fixed_variance_z)
    std_interval = scipy.stats.norm.isf(alpha/2)
    fixed_int_z = (fixed_effect_z-std_interval*fixed_std_z, fixed_effect_z+std_interval*fixed_std_z)
    z_value = fixed_effect_z/fixed_std_z
    p_value = 2*scipy.stats.norm.sf(z_value)
    fixed_effect_cor = z_to_r(fixed_effect_z)
    fixed_int_corr = (z_to_r(fixed_int_z[0]), z_to_r(fixed_int_z[1]))
    return (p_value, fixed_effect_cor, fixed_int_corr[0], fixed_int_corr[1], p_value <= alpha)

def making_random_effects_df(df):
    fixed_effect_df = making_fixed_effect_df(df)
    Q = fixed_effect_df.select(pl.sum('WY_2') - pl.sum('WY')**2 / pl.sum('raw_weight_W')).item()
    C = fixed_effect_df.select(pl.sum('raw_weight_W') - pl.sum('W_2')/pl.sum('raw_weight_W')).item()
    T_2 = (Q - (len(fixed_effect_df)-1))/C
    fixed_effect_df = (
        fixed_effect_df.with_columns(V_star=pl.col('variance_within_V') + T_2)
        .with_columns(W_star=1/pl.col('V_star'))
        .with_columns(W_star_Y=pl.col('W_star')*pl.col('effect_size_Y'))
    )
    return fixed_effect_df

def random_effects(df, alpha=0.05):
    random_effects_df = making_random_effects_df(df)
    random_effect_z = random_effects_df.select(pl.sum('W_star_Y')/pl.sum('W_star')).item()
    random_variance_z = random_effects_df.select(1/pl.sum('W_star')).item()
    random_std_z = np.sqrt(random_variance_z)
    std_interval = scipy.stats.norm.isf(alpha/2)
    random_int_z = (random_effect_z-std_interval*random_std_z, random_effect_z+std_interval*random_std_z)
    z_value = random_effect_z/random_std_z
    p_value = 2*scipy.stats.norm.sf(z_value)
    random_effect_cor = z_to_r(random_effect_z)
    random_int_corr = (z_to_r(random_int_z[0]), z_to_r(random_int_z[1]))
    return (p_value, random_effect_cor, random_int_corr[0], random_int_corr[1], p_value <= alpha)

# %%
from tqdm import trange

def meta_analysis_correlations(correlations_list, lengths_list, fixed=True):
    columns_names = correlations_list[0].columns
    results = []
    for i in trange(len(columns_names)):
        for j in range(i + 1, len(columns_names)):
            # analyzing correlations between columns i and j
            data = []
            for corr, length in zip(correlations_list, lengths_list):
                if not corr.isna().iloc[i,j] and abs(corr.iloc[i, j]) != 1 and length > 3:
                    data.append((corr.iloc[i, j], length))
            # print(data)
            df_data = pl.DataFrame(data, schema=['correlation', 'n'], orient='row')
            if fixed:
                result = fixed_effect(df_data)
            else:
                result = random_effects(df_data)
            results.append((columns_names[i], columns_names[j], *result))

    return pl.DataFrame(results, schema=['Metric A', 'Metric B', 'p-value', 'effect', 'interval_l', 'interval_r', 'significant'], orient='row')

# %%
meta_analysis_fixed_results = meta_analysis_correlations(correlations_df_list, lengths_list, fixed=True)
meta_analysis_fixed_results.write_csv("outputs/method-meta-analysis-fixed.csv")

# %%
meta_analysis_fixed_results.filter(
    (pl.col('significant') == 1) & (pl.col('effect').abs() > 0.7)
).sort(pl.col('effect').abs(), descending=True).write_csv('outputs/method-fixed-important.csv')

# %%
meta_analysis_random_results = meta_analysis_correlations(correlations_df_list, lengths_list, fixed=False)
meta_analysis_random_results.write_csv("outputs/method-meta-analysis-random.csv")

# %%
meta_analysis_random_results.filter(
    (pl.col('significant') == 1) & (pl.col('effect').abs() > 0.7)
).sort(pl.col('effect').abs(), descending=True).write_csv('outputs/method-random-important.csv')

# %%
meta_analysis_fixed_results["p-value"].unique()

# %%
meta_analysis_fixed_results.sort(by='p-value', descending=True).gather_every(100).head(10)

# %%
i = 2
j = -6
# analyzing correlations between columns i and j
data = []
for corr, length in zip(correlations_df_list, lengths_list):
    if not corr.isna().iloc[i,j] and abs(corr.iloc[i, j]) != 1 and length > 3:
        data.append((corr.iloc[i, j], length))
# print(data)
df_data = pl.DataFrame(data, schema=['correlation', 'n'], orient='row')
print(df_data)

# %%
result = fixed_effect(df_data)
result

# %%
import pandas as pd

def find_columns_to_drop(file_path, correlation_threshold=0.9):
    df = pd.read_csv(file_path)
    correlations = df[["Metric A", "Metric B", "effect"]]
    high_correlations = correlations[correlations["effect"].abs() >= correlation_threshold]
    columns_to_drop = set()
    processed_columns = set()
    
    for _, row in high_correlations.iterrows():
        col_a, col_b = row["Metric A"], row["Metric B"]
        if col_a not in processed_columns and col_b not in processed_columns:
            columns_to_drop.add(col_b)
            processed_columns.add(col_a)
            processed_columns.add(col_b)
        elif col_a in processed_columns:
            columns_to_drop.add(col_b)
            processed_columns.add(col_b)
        elif col_b in processed_columns:
            columns_to_drop.add(col_a)
            processed_columns.add(col_a)
    
    return list(columns_to_drop)

file_path = "outputs/method-random-important.csv"
columns_to_drop = find_columns_to_drop(file_path, correlation_threshold=0.9)
print("Columns to drop:", columns_to_drop)


# %%
empty_cols = zero_col_list

final_list_90 = empty_cols.copy()
final_list_80 = empty_cols.copy()
final_list_70 = empty_cols.copy()

final_list_90.extend(find_columns_to_drop(file_path, 0.9))
final_list_80.extend(find_columns_to_drop(file_path, 0.8))
final_list_70.extend(find_columns_to_drop(file_path, 0.7))

# %%
final_list_90

# %%
final_list_80

# %%
final_list_70

# %%
print(f"With threshold 0.9 we drop {len(final_list_90)} columns, with threshold 0.8 we drop {len(final_list_80)} columns, with threshold 0.7 we drop {len(final_list_70)} columns.")

# %%
[x for x in final_list_80 if x not in final_list_90]

# %%
[x for x in final_list_70 if x not in final_list_80]


