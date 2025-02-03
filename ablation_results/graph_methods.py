import pandas as pd
import matplotlib.pyplot as plt
import os
# modify plt rcParam

plt.rcParams.update({'font.size': 20, 'axes.linewidth': 1.5, 'lines.linewidth': 2.5})

filenamelist = [
    ('result_graphs/llama7b_results', ["L2_7B_2k.csv", "L2_7B_2k_oracle.csv", "L2_7B_2k_h2o_true.csv", "L2_7B_2k_quest.csv", "L2_7B_2k_streamingLLM.csv"]),
    ('result_graphs/llama8b_results', ["L3_8B_1k.csv", "L3_8B_1k_oracle.csv", "L3_8B_1k_h2o_true.csv", "L3_8B_1k_quest.csv", "L3_8B_1k_streamingLLM.csv"]),
    ('result_graphs/llama3b_1pc_results', ["L3_3B_2k_1PC_oracle.csv", "L3_3B_2k_1PC_h2o_true.csv", "L3_3B_2k_1PC_quest.csv", "L3_3B_2k_1PC.csv", "L3_3B_2k_1PC_streamingLLM.csv"]),
    ('result_graphs/llama3b_results', ["L3_3B_2k_oracle.csv", "L3_3B_2k_h2o_true.csv", "L3_3B_2k_quest.csv", "L3_3B_2k.csv", "L3_3B_2k_streamingLLM.csv"]),
    ('result_graphs/llama1b_1pc_results', ["L3_1B_2k_1PC_oracle.csv", "L3_1B_2k_1PC_h2o_true.csv", "L3_1B_2k_1PC_quest.csv", "L3_1B_2k_1PC.csv", "L3_1B_2k_1PC_streamingLLM.csv"]),
    ('result_graphs/llama1b_results', ["L3_1B_2k_oracle.csv", "L3_1B_2k_h2o_true.csv", "L3_1B_2k_quest.csv", "L3_1B_2k.csv", "L3_1B_2k_streamingLLM.csv"]),
    ('result_graphs/mistral7b_results', ["M7B_1k.csv","M7B_1k_oracle.csv","M7B_1k_h2o_true.csv","M7B_1k_quest.csv","M7B_1k_streamingLLM.csv"]),
    ('result_graphs/phi35mini_1pc_results', ["P35mini_1k_1PC.csv", "P35mini_1k_1PC_oracle.csv", "P35mini_1k_1PC_h2o_true.csv", "P35mini_1k_1PC_quest.csv", "P35mini_1k_1PC_streamingLLM.csv"]),
    ('result_graphs/phi3mini_1pc_results', ["P3mini_1k_1PC.csv", "P3mini_1k_1PC_oracle.csv", "P3mini_1k_1PC_h2o_true.csv", "P3mini_1k_1PC_quest.csv", "P3mini_1k_1PC_streamingLLM.csv"])

]

for output_dir, filenames in filenamelist:
    try:
        filenames = ["evalresults/" + x for x in filenames]
        dataframes = []

        # Loop through each file
        for filename in filenames:
            try:
                with open(filename, 'r') as file:
                    lines = file.readlines()

                # Filter out the duplicated header rows
                filtered_lines = []
                header_found = False
                for line in lines:
                    if line.startswith("seed,model_path"):
                        if not header_found:  # Keep the first header only
                            filtered_lines.append(line)
                            header_found = True
                    else:
                        filtered_lines.append(line)

                # Read the filtered data into a dataframe, skip extra headers
                df = pd.read_csv(pd.io.common.StringIO(''.join(filtered_lines)))
                dataframes.append(df)
            except Exception as e:
                print(e)
                pass
        # import pdb; pdb.set_trace()
        # Combine all dataframes into one
        combined_data = pd.concat(dataframes, ignore_index=True)

        # Save the combined dataframe into a new CSV file
        combined_data.to_csv('combined_results.csv', index=False)

        # Optionally, display the combined dataframe
        print(combined_data.head()) 
        print(combined_data.shape)

        # Set up a function to generate each graph
        def plot_graph(y_column, title, output_filename):
            plt.figure(figsize=(12, 8))  # Set figure size

            for wname in combined_data['wname'].unique():
                if 'quest' in wname and 'perplexity' in y_column:
                    continue
                print(wname)
                subset = combined_data[combined_data['wname'] == wname]
                use_val = 'true_token_sparsity' if 'quest' not in wname else 'true_token_sparsity'
                # sort by true_token_sparsity
                subset = subset.sort_values(by=use_val)
                plt.plot(subset[use_val], subset[y_column], label=wname.replace('_tests', ''), marker='o')

            plt.xlabel('Net Sparsity')  # X-axis label
            plt.ylabel(title)  # Y-axis label
            plt.title(title)  # Title
            plt.legend()  # Legend with "Method" label
            plt.grid(True)  # Enable grid for better readability
            plt.tight_layout()  # Adjust layout to fit everything nicely
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f'{output_dir}/' + output_filename, format='pdf')  # Save as PDF
            plt.close()  # Close the plot to avoid overlapping

        # Plot the three graphs
        plot_graph('perplexity', 'Perplexity', 'perplexity.pdf')
        plot_graph('hellaswag_acc', 'HellaSwag', 'hellaswag_acc.pdf')
        plot_graph('piqa_acc', 'PIQA', 'piqa_acc.pdf')
        plot_graph('winogrande_acc', 'Winogrande', 'winogrande_acc.pdf')
        plot_graph('arc_easy_acc', 'ARC_Easy', 'arceasy_acc.pdf')


        import pandas as pd
        import matplotlib.pyplot as plt
        import os

        # Initialize an empty list to hold dataframes
        dataframes = []

        # Loop through each file
        for filename in filenames:
            try:
                with open(filename, 'r') as file:
                    lines = file.readlines()

                # Filter out the duplicated header rows
                filtered_lines = []
                header_found = False
                for line in lines:
                    if filename == 'mse_ble_bnhll.csv':
                        line = line.replace('oracle_tests', 'predictor_tests')
                    if line.startswith("seed,model_path"):
                        if not header_found:
                            filtered_lines.append(line)
                            header_found = True
                    else:
                        filtered_lines.append(line)

                # Read the filtered data into a dataframe
                df = pd.read_csv(pd.io.common.StringIO(''.join(filtered_lines)))
                dataframes.append(df)
            except Exception as e:
                print(e)
                pass

        # Combine all dataframes into one
        combined_data = pd.concat(dataframes, ignore_index=True)

        # Save the combined dataframe into a new CSV file
        combined_data.to_csv('combined_results.csv', index=False)

        # Extract the baseline accuracies from Oracle.csv at true_token_sparsity=0
        baseline = 'oracle'
        use_val = 'true_token_sparsity' if 'quest' not in baseline else 'true_token_sparsity'
        # oracle_baseline = combined_data[(combined_data['wname'] == baseline) & (combined_data[use_val] == 0)]
        oracle_baseline = combined_data[
            combined_data['wname'].str.contains(baseline, na=False) & 
            (combined_data[use_val] < 10)
        ]
        # Function to calculate percentage difference relative to the baseline
        def calculate_percentage_diff(df, metric, baseline_df):
            for wname in df['wname'].unique():
                baseline_value = baseline_df[baseline_df['wname'].str.contains(baseline, na=False)][metric].values[0]
                df.loc[df['wname'] == wname, f'{metric}_pct_diff'] = (
                    (df.loc[df['wname'] == wname, metric] - baseline_value) / baseline_value * 100
                )

        # Metrics to compute percentage differences for
        metrics = ['hellaswag_acc', 'piqa_acc', 'winogrande_acc', 'arc_easy_acc']

        # Apply percentage difference calculation for all metrics
        for metric in metrics:
            calculate_percentage_diff(combined_data, metric, oracle_baseline)

        # Set up a function to generate each graph
        def plot_percentage_diff_graph(metric, title, output_filename):
            plt.figure(figsize=(12, 8))
            for wname in combined_data['wname'].unique():
                subset = combined_data[combined_data['wname'] == wname]
                use_val = 'true_token_sparsity' if 'quest' not in wname else 'true_token_sparsity'
                subset = subset.sort_values(by=use_val)
                plt.plot(
                    subset[use_val], subset[f'{metric}_pct_diff'], 
                    label=wname.replace('_tests', ''), marker='o'
                )

            plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)  # Baseline line at 0%
            plt.xlabel('Net Sparsity')
            plt.ylabel('Accuracy % Difference')
            plt.title(f'{title} % Difference vs Oracle')
            # find min-max
            # poffs = 0.5
            # maxlim = 10
            # uplim = (max(maxlim, combined_data[f'{metric}_pct_diff'].max() + poffs) + maxlim)/3
            # plt.ylim(-maxlim, uplim)
            plt.legend(ncol=2, loc='upper center')
            plt.grid(True)
            plt.tight_layout()
            if not os.path.exists('results/results_perc'):
                os.makedirs('results/results_perc', exist_ok=True)
            plt.savefig(f'results/results_perc/{output_filename}', format='pdf')
            plt.close()

        # Plot percentage difference graphs for all metrics
        plot_percentage_diff_graph('hellaswag_acc', 'HellaSwag', 'hellaswag_pct_diff.pdf')
        plot_percentage_diff_graph('piqa_acc', 'PIQA', 'piqa_pct_diff.pdf')
        plot_percentage_diff_graph('winogrande_acc', 'Winogrande', 'winogrande_pct_diff.pdf')
        plot_percentage_diff_graph('arc_easy_acc', 'ARC_Easy', 'arceasy_pct_diff.pdf')

        average_of = ['hellaswag_acc', 'piqa_acc', 'winogrande_acc', 'arc_easy_acc']

        # Compute the average accuracy
        combined_data['average_acc'] = combined_data[average_of].mean(axis=1)

        # Compute the average percentage differences
        pct_diff_cols = [f'{m}_pct_diff' for m in average_of]
        combined_data['average_acc_pct_diff'] = combined_data[pct_diff_cols].mean(axis=1)

        def plot_graph_generic(y_column, title, output_filename):
            plt.figure(figsize=(12, 8))
            for wname in combined_data['wname'].unique():
                subset = combined_data[combined_data['wname'] == wname]
                use_val = 'true_token_sparsity'  # consistent with previous logic
                subset = subset.sort_values(by=use_val)
                plt.plot(subset[use_val], subset[y_column], label=wname.replace('_tests', ''), marker='o')

            plt.xlabel('Net Sparsity')
            plt.ylabel(title)
            plt.title(title)
            plt.legend(ncol=2)
            plt.grid(True)
            plt.tight_layout()
            # Ensure output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f'{output_dir}/{output_filename}', format='pdf')
            plt.close()

        # Plot the average accuracy
        plot_graph_generic('average_acc', 'Average Accuracy', 'average.pdf')

        # Plot the average accuracy percentage difference
        plot_graph_generic('average_acc_pct_diff', 'Average Accuracy % Difference vs Oracle', 'average_pct_diff.pdf')


    # filenames = ["L2_7B_2k.csv", "L2_7B_2k_oracle.csv", "L2_7B_2k_h2o_true.csv", "L2_7B_2k_quest.csv"]
    # filenames = ["evalresults/" + x for x in filenames]
    # output_dir = 'result_graphs/llama7b_results'

    # filenames = ["L3_8B_1k.csv", "L3_8B_1k_oracle.csv", "L3_8B_1k_h2o_true.csv", "L3_8B_1k_quest.csv"]
    # filenames = ["evalresults/" + x for x in filenames]
    # output_dir = 'result_graphs/llama8b_results'

    # filenames = ["L3_3B_2k_1PC_oracle.csv", "L3_3B_2k_1PC_h2o_true.csv", "L3_3B_2k_1PC_quest.csv", "L3_3B_2k_1PC.csv"]
    # filenames = ["evalresults/" + x for x in filenames]
    # output_dir = 'result_graphs/llama3b_1pc_results'

    # filenames = ["L3_3B_2k_oracle.csv", "L3_3B_2k_h2o_true.csv", "L3_3B_2k_quest.csv", "L3_3B_2k.csv"]
    # filenames = ["evalresults/" + x for x in filenames]
    # output_dir = 'result_graphs/llama3b_results'

    # filenames = ["L3_1B_2k_1PC_oracle.csv", "L3_1B_2k_1PC_h2o_true.csv", "L3_1B_2k_1PC_quest.csv", "L3_1B_2k_1PC.csv"]
    # filenames = ["evalresults/" + x for x in filenames]
    # output_dir = 'result_graphs/llama1b_1pc_results'

    # filenames = ["L3_1B_2k_oracle.csv", "L3_1B_2k_h2o_true.csv", "L3_1B_2k_quest.csv", "L3_1B_2k.csv"]
    # filenames = ["evalresults/" + x for x in filenames]
    # output_dir = 'result_graphs/llama1b_results'
    except:
        continue