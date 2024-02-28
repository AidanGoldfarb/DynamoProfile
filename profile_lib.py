import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,pickle

def plot(dataframes,only_unexpected=False,title=None,filename=None):
    for df in dataframes:
        percent_difference = np.abs((df.iloc[:, 1] - df.iloc[:, 2]) / df.iloc[:, 1]) * 100
        percent_margin_values = np.linspace(0.1, percent_difference.max(), 100)

        if only_unexpected:
            #Mask to apply if only plotting unexpected values where the first column is less than the second
            mask = df.iloc[:, 1] < df.iloc[:, 2]
            filtered_percent_difference = percent_difference[mask]
            counts_filtered = [np.sum(filtered_percent_difference > margin) for margin in percent_margin_values]
            counts_unfiltered = [np.sum(percent_difference > margin) for margin in percent_margin_values]

            plt.plot(percent_margin_values, counts_filtered, marker='o', linestyle='-', label=df.columns[1][5:-4])

            plt.plot(percent_margin_values, counts_unfiltered, color='0.75', linestyle='--')

            # for x, y1, y2 in zip(percent_margin_values, counts_filtered, counts_unfiltered):
            #     plt.plot([x, x], [y1, y2], color='grey', linestyle=':')
        else:
            counts = [np.sum(percent_difference > margin) for margin in percent_margin_values]
            plt.plot(percent_margin_values, counts, marker='o', linestyle='-', label=df.columns[1][5:-4])

    plt.title(title)
    plt.xlabel('Runtime difference (%)')
    plt.ylabel('Number of layers')
    plt.xscale('log')
    #plt.yscale('log')
    plt.grid(True)
    plt.legend(title='Column')
    plt.savefig(filename)

def get_dataframes():
    df_res,df_google,df_dense,df_squeeze,df_alex,df_mobile = _get_cuda_triton()

    df_res,df_google,df_dense,df_squeeze,df_alex,df_mobile = _get_interp_cpp()

    df_res,df_google,df_dense,df_squeeze,df_alex,df_mobile = _get_interp_gpu()

def filter_similar_differences(dataframes, threshold=1.0):
    name_counts = {}  # Dictionary to track the counts of names

    # Process each DataFrame individually
    for df in dataframes:
        # Calculate percent difference where column 1 < column 2
        percent_difference = np.where(df.iloc[:, 1] < df.iloc[:, 2],
                                      100 * (df.iloc[:, 2] - df.iloc[:, 1]) / df.iloc[:, 1], np.nan)

        # Iterate through the DataFrame
        for index, row in df.iterrows():
            name = row.iloc[0].split('(')[0]
            diff = percent_difference[index]

            # Skip if NaN
            if pd.isna(diff):
                continue

            # Check if difference is within the threshold
            if diff > threshold:
                if name not in name_counts:
                    name_counts[name] = 1
                else:
                    name_counts[name] += 1

    # Convert the name_counts dictionary to a DataFrame
    matching_df = pd.DataFrame(list(name_counts.items()), columns=['Name', 'Appearances'])
    return matching_df.sort_values('Appearances',ascending=False)

def count_percentage_of_significant_differences(dataframes, threshold=1.0):
    name_counts = {}  # Dictionary to track the counts of significant differences
    total_appearances = {}  # Dictionary to track the total counts of names

    # Process each DataFrame individually
    for df in dataframes:
        # Calculate percent difference where column 1 < column 2
        percent_difference = np.where(df.iloc[:, 1] < df.iloc[:, 2],
                                      100 * (df.iloc[:, 2] - df.iloc[:, 1]) / df.iloc[:, 1], np.nan)

        # Iterate through the DataFrame
        for index, row in df.iterrows():
            # Process the name to take the part before '('
            name = row.iloc[0].split('(')[0]

            diff = percent_difference[index]

            # Update total appearances count
            if name not in total_appearances:
                total_appearances[name] = 1
            else:
                total_appearances[name] += 1

            # Skip if NaN
            if pd.isna(diff):
                continue

            # Update significant appearances count
            if diff > threshold:
                if name not in name_counts:
                    name_counts[name] = 1
                else:
                    name_counts[name] += 1

    # Calculate the percentage of significant differences
    percentages = {name: (name_counts.get(name, 0) / total_appearances[name]) * 100 for name in total_appearances}

    # Convert the percentages dictionary to a DataFrame
    percentages_df = pd.DataFrame(list(percentages.items()), columns=['Name', 'Percentage'])
    return percentages_df.sort_values('Percentage',ascending=False)

"""
arr0 should model which is expected to be slower

[('layer0',.424),(layer1,.202),...(layern,.953)]
[('layer0',.244),(layer1,.829),...(layern,.053)]

"""
def find_slow_layers_helper(config0,config1):
    #_plot(_get_interp_gpu(),only_unexpected=True,title="Runtime difference of layers (interp vs GPU)",filename="InterpVGPU")
    # res_raw = filter_similar_differences(_get_interp_cpp())
    # res_freq = count_percentage_of_significant_differences(_get_interp_cpp())
    # res = pd.merge(res_raw,res_freq, on='Name')
    # print(res[0:50])
    # exit()
    arr0 = np.array(unpickle_lst(config0+".pkl"),dtype=DATA_DTYPE)
    arr1 = np.array(unpickle_lst(config1+".pkl"),dtype=DATA_DTYPE)
    slow_layers = []

    for ((ln0,tm0),(ln1,tm1)) in zip(arr0,arr1):
        assert ln0 == ln1
        if tm0 < tm1: 
            slow_layers.append((ln0,diff(tm0,tm1)))
    slow_layers = np.sort(np.array(slow_layers,dtype=DATA_DTYPE), order='Time')
    tlayers = arr0.shape[0]
    nslow_layers = slow_layers.shape[0]
    difference = (float(slow_layers.shape[0])/arr0.shape[0])*100

    config0rt = np.sum(arr0['Time'])
    config1rt = np.sum(arr1['Time'])

    print(f"\t{nslow_layers} of {tlayers} ({difference:.3}%)\n\twere faster in {config0}")
    print(f"\tTotal runtime of {config0}: {BOLD} {config0rt:.4} {ENDC}\n\tTotal runtime of {config1}: {BOLD} {config1rt:.4} {ENDC}")
    if config0rt > config1rt:
        print(f"\t{BOLD}{GREEN}-{diff(config0rt,config1rt):.4}{ENDC}%")
    else:
        print(f"\t{BOLD}{RED}+{diff(config0rt,config1rt):.4}{ENDC}%")