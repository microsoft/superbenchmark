import json
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def cal_arithmatic_intensity(m,n,k,precision):
    if precision == 'fp16':
        return cal_ai_fp16(m,n,k)
    elif precision == 'fp8':
        return cal_ai_fp8(m,n,k)

def cal_ai_fp16(m,n,k):
    return m*n*k/(m*n+n*k+k*m)

def cal_ai_fp8(m,n,k):
    bytes_per_element = 1  # FP8 = 1 byte
    flops = 2 * m * n * k  # 2 FLOPs per MAC
    bytes_moved = (m*n + n*k + k*m) * bytes_per_element
    return flops / bytes_moved

def theoretical_gemm_flops(m,n,k, spec_mem_bandwidth, spec_precision_gemm, precision):
    ai = cal_arithmatic_intensity(m,n,k,precision)
    return min(spec_mem_bandwidth*ai, spec_precision_gemm)

# Define the Roofline model function (replace with your own model)
def roofline_model(computational_intensity, peak_performance, memory_bandwidth):
    return np.minimum(peak_performance, memory_bandwidth * computational_intensity)



def analyze_roof_line(data, target, baseline, target_nick, baseline_nick, precision='fp16'):
    #restriction: the sku keys in 'data' must be exactly the same as the value of 'target' and 'baseline'

    target = target.lower()
    baseline = baseline.lower()
    
    # modify values, automatically pass in, - pending
    #                    H200    H100    MI300x     GB200
    #spec mem bw         4.8     3.35    5.2        8
    #spec gemm fp16      989     989     1300       2500
    #projected gemm fp16 975.9   815.1   763.8      2???
    #projected mem bw    4.26    3.05    3.9        7.?
    
    # spec_mem_bandwidth_dict = {
    #     target:8,
    #     baseline:3.35
    # }
    # spec_precision_gemm_dict = {
    #     target:{
    #         'fp16': 2500
    #     },
    #     baseline:{
    #         'fp16': 989
    #     }
    # }
    # projected_precision_gemm_dict = {
    #     target:{
    #         'fp16': 2000
    #     },
    #     baseline:{
    #         'fp16': 815.1
    #     }
    # }
    # projected_mem_bandwidth_dict = {
    #     target: 7.0,
    #     baseline: 3.05
    # }
    
    # Define the dictionary
    spec_dict = {
        'h200': {'spec_mem_bw': 4.8, 'spec_gemm_fp16': 989, 'projected_gemm_fp16': 975.9, 'projected_mem_bw': 4.26, 'spec_gemm_fp8': 1979, 'projected_gemm_fp8': 1892},
        'h100': {'spec_mem_bw': 3.35, 'spec_gemm_fp16': 989, 'projected_gemm_fp16': 815.1, 'projected_mem_bw': 3.05, 'spec_gemm_fp8': 1979, 'projected_gemm_fp8': 1630},
        'h100vm': {'spec_mem_bw': 3.35, 'spec_gemm_fp16': 989, 'projected_gemm_fp16': 815.1, 'projected_mem_bw': 3.05, 'spec_gemm_fp8': 1979, 'projected_gemm_fp8': 1630},
        'h100bm': {'spec_mem_bw': 3.35, 'spec_gemm_fp16': 989, 'projected_gemm_fp16': 815.1, 'projected_mem_bw': 3.05, 'spec_gemm_fp8': 1979, 'projected_gemm_fp8': 1630},
        'mi300x': {'spec_mem_bw': 5.2, 'spec_gemm_fp16': 1300, 'projected_gemm_fp16': 763.8, 'projected_mem_bw': 3.9, 'spec_gemm_fp8': 2600, 'projected_gemm_fp8': 1527.6},
        'mi300xhf': {'spec_mem_bw': 5.2, 'spec_gemm_fp16': 1300, 'projected_gemm_fp16': 763.8, 'projected_mem_bw': 3.9, 'spec_gemm_fp8': 2600, 'projected_gemm_fp8': 1527.6},
        'mi300x2410': {'spec_mem_bw': 5.2, 'spec_gemm_fp16': 1300, 'projected_gemm_fp16': 763.8, 'projected_mem_bw': 3.9, 'spec_gemm_fp8': 2600, 'projected_gemm_fp8': 1527.6},
        'gb200': {'spec_mem_bw': 8, 'spec_gemm_fp16': 2500, 'projected_gemm_fp16': 2250, 'projected_mem_bw': 7.2, 'spec_gemm_fp8': 5000, 'projected_gemm_fp8': 4500},
        'beta': {'spec_mem_bw': 4.8, 'spec_gemm_fp16': 989, 'projected_gemm_fp16': 975.9, 'projected_mem_bw': 4.26, 'spec_gemm_fp8': 1979, 'projected_gemm_fp8': 1892},
        'alpha': {'spec_mem_bw': 4.8, 'spec_gemm_fp16': 989, 'projected_gemm_fp16': 975.9, 'projected_mem_bw': 4.26, 'spec_gemm_fp8': 1979, 'projected_gemm_fp8': 1892},
        'eta': {'spec_mem_bw': 8, 'spec_gemm_fp16': 2500, 'projected_gemm_fp16': 2250, 'projected_mem_bw': 7.2, 'spec_gemm_fp8': 5000, 'projected_gemm_fp8': 4500},
        'delta': {'spec_mem_bw': 8, 'spec_gemm_fp16': 2500, 'projected_gemm_fp16': 2250, 'projected_mem_bw': 7.2, 'spec_gemm_fp8': 5000, 'projected_gemm_fp8': 4500},
        'other': {'spec_mem_bw': 1, 'spec_gemm_fp16': 100, 'projected_gemm_fp16': 100, 'projected_mem_bw': 1, 'spec_gemm_fp8': 100, 'projected_gemm_fp8': 100}
    }

    # Assign the values from the dictionary
    spec_mem_bandwidth_dict = {
        target: spec_dict.get(target_nick.lower(), spec_dict["other"]).get('spec_mem_bw'),
        baseline: spec_dict.get(baseline_nick.lower(), spec_dict["other"]).get('spec_mem_bw')
    }
    spec_precision_gemm_dict = {
        target: {
            'fp16': spec_dict.get(target_nick.lower(), spec_dict["other"]).get('spec_gemm_fp16'),
            'fp8': spec_dict.get(target_nick.lower(), spec_dict["other"]).get('spec_gemm_fp8')
        },
        baseline: {
            'fp16': spec_dict.get(baseline_nick.lower(), spec_dict["other"]).get('spec_gemm_fp16'),
            'fp8': spec_dict.get(baseline_nick.lower(), spec_dict["other"]).get('spec_gemm_fp8')
        }
    }
    projected_precision_gemm_dict = {
        target: {
            'fp16': spec_dict.get(target_nick.lower(), spec_dict["other"]).get('projected_gemm_fp16'),
            'fp8': spec_dict.get(target_nick.lower(), spec_dict["other"]).get('projected_gemm_fp8')
        },
        baseline: {
            'fp16': spec_dict.get(baseline_nick.lower(), spec_dict["other"]).get('projected_gemm_fp16'),
            'fp8': spec_dict.get(baseline_nick.lower(), spec_dict["other"]).get('projected_gemm_fp8')
        }
    }
    projected_mem_bandwidth_dict = {
        target: spec_dict.get(target_nick.lower(), spec_dict["other"]).get('projected_mem_bw'),
        baseline: spec_dict.get(baseline_nick.lower(), spec_dict["other"]).get('projected_mem_bw')
    }

    #precision = 'fp16'

    computational_intensity = {}
    experiment_performance = {}
    theorical_performance = {}
    
    output_line_math = {}
    output_line_mem = {}
    output_line_math['Kernel'] = 'Math-Limited (TFLOPS)'
    output_line_mem['Kernel'] = 'Memory-Limited (TB/s)'

    memory_bound = {}
    compute_bound = {}
    num_skus = len(list(data.keys()))
    fig, axs = plt.subplots(num_skus, figsize=(12, 16))
    plt.rcParams['font.size'] = 16
    makecell = [f"\\makecell[l]{{", f" \\\\ ", f"}}"] # make a two line cell
    
    for index, sku in enumerate(data):
        spec_mem_bandwidth = spec_mem_bandwidth_dict[sku]
        spec_precision_gemm = spec_precision_gemm_dict[sku][precision]
        
        computational_intensity[sku] = []
        experiment_performance[sku] = []
        theorical_performance[sku] = []
        metrics = data[sku].keys()
        for metric in metrics:
            pattern = f'(hipblaslt|cublaslt)-gemm:*.*/{precision}.*_(\d+)_(\d+)_(\d+)_(\d+)_flops'
            # Match the pattern
            match = re.match(pattern, metric)
            if match:
                group_values = match.groups()
                m = int(group_values[2])
                n = int(group_values[3])
                k = int(group_values[4])
                computational_intensity[sku].append(cal_arithmatic_intensity(m,n,k, precision))
                experiment_performance[sku].append(data[sku][metric])
                theorical_performance[sku].append(theoretical_gemm_flops(m,n,k, spec_mem_bandwidth, spec_precision_gemm, precision))
        computational_intensity[sku] = np.array(computational_intensity[sku])
        experiment_performance[sku] = np.array(experiment_performance[sku])
        theorical_performance[sku] = np.array(theorical_performance[sku])

        if len(experiment_performance[sku]) == 0:
            empty_output = []
            empty_output.append({(target_nick + makecell[1] + 'Avg. Perf.'): '0'})
            empty_output.append({(target_nick + makecell[1] + 'Avg. Perf.'): '0'})
            return empty_output, fig

        # Fit the model to the experimental data
        params, covariance = curve_fit(roofline_model, computational_intensity[sku], experiment_performance[sku])

        y_fit = roofline_model(computational_intensity[sku], *params)
        # Find the x-value corresponding to the minimum y-value
        # Calculate the x-value corresponding to params[0]
        target_performance = params[0]
        min_computational_intensity = target_performance / params[1]
        if sku not in memory_bound:
            memory_bound[sku] = min_computational_intensity
        memory_bound[sku] = min(memory_bound[sku], min_computational_intensity)
        
        mask = computational_intensity[sku] > min_computational_intensity
        experiment_performance_filtered = experiment_performance[sku][mask]
        min_compute_bound_performance = np.percentile(experiment_performance_filtered, 50)

        mask = experiment_performance[sku] > min_compute_bound_performance
        max_computational_intensity = np.min(computational_intensity[sku][mask])
        
        if sku not in compute_bound:
            compute_bound[sku] = max_computational_intensity
        compute_bound[sku] = max(max_computational_intensity, compute_bound[sku])

        print(f'SKU {sku} - memory_bound: {memory_bound[sku]}, compute_bound: {compute_bound[sku]}')

        # Plot the experimental data and fitted model
        if sku == target:
            nickname = target_nick
        else:
            nickname = baseline_nick
        
        axs[index].plot(computational_intensity[sku], experiment_performance[sku], 'o', label=f'{nickname} {precision} Experimental')
        axs[index].plot(computational_intensity[sku], theorical_performance[sku], 'o', label=f'{nickname} {precision} Theoretical')

        axs[index].set_xlabel('Computational Intensity (FLOP/s per Byte)')
        axs[index].set_ylabel('Performance (TFLOP/s)')
        axs[index].set_title(f'Roofline Plot with Fitted Data for {nickname}')
        axs[index].legend()

        # Display the fitted parameters
        print("Fitted Parameters - Peak Performance:", params[0])
        print("Fitted Parameters - Memory Bandwidth:", params[1])
    
    plt.subplots_adjust(hspace = 0.3)

    memory_bound_data = {}
    math_bound_data = {}
    for sku in data:
        mask = computational_intensity[sku] < memory_bound[sku]
        memory_bound_compute_intensity = computational_intensity[sku][mask]
        memory_bound_data[sku] = experiment_performance[sku][mask] 
        memory_bound_data[sku] = [ memory_bound_data[sku][i] / memory_bound_compute_intensity[i] for i in range(len(memory_bound_data[sku]))]
        mask = computational_intensity[sku] > compute_bound[sku]
        math_bound_data[sku] = experiment_performance[sku][mask]

    ratio  = np.mean(memory_bound_data[target]) /  np.mean(memory_bound_data[baseline])
    output_line_mem[makecell[0] + target_nick + makecell[1] + 'Avg. Perf.' + makecell[2]] = f'{round(np.mean(memory_bound_data[target]), 2)}'
    output_line_mem[makecell[0] + baseline_nick + makecell[1] + 'Avg. Perf.' + makecell[2]] = f'{round(np.mean(memory_bound_data[baseline]), 2) }'
    output_line_mem[makecell[0] + 'Speedup' + makecell[1] + 'Ratio' + makecell[2]] = round(ratio, 2)
    output_line_mem[makecell[0] + 'Projected' + makecell[1] + 'Ratio' + makecell[2]] = f'{round(projected_mem_bandwidth_dict[target] / projected_mem_bandwidth_dict[baseline], 2)} ({projected_mem_bandwidth_dict[target]} / {projected_mem_bandwidth_dict[baseline]})'
    output_line_mem[makecell[0] + 'Ideal' + makecell[1] + 'Ratio' + makecell[2]] = f'{round(spec_mem_bandwidth_dict[target] / spec_mem_bandwidth_dict[baseline], 2) } ({spec_mem_bandwidth_dict[target]} / {spec_mem_bandwidth_dict[baseline]})'
    print(np.mean(memory_bound_data[target]))
    print(np.mean(memory_bound_data[baseline]))
    print(f"Memory bound ratio: {ratio}")
    
    ratio  = np.mean(math_bound_data[target]) /  np.mean(math_bound_data[baseline])
    output_line_math[makecell[0] + target_nick + makecell[1] + 'Avg. Perf.' + makecell[2]] = f'{round(np.mean(math_bound_data[target]), 2)}'
    output_line_math[makecell[0] + baseline_nick + makecell[1] + 'Avg. Perf.' + makecell[2]] = f'{round(np.mean(math_bound_data[baseline]), 2)} ' 
    output_line_math[makecell[0] + 'Speedup' + makecell[1] + 'Ratio' + makecell[2]] = round(ratio, 2)
    output_line_math[makecell[0] + 'Projected' + makecell[1] + 'Ratio' + makecell[2]] = f'{round(projected_precision_gemm_dict[target][precision]  / projected_precision_gemm_dict[baseline][precision] , 2)} ({projected_precision_gemm_dict[target][precision]} / {projected_precision_gemm_dict[baseline][precision]})'
    output_line_math[makecell[0] + 'Ideal' + makecell[1] + 'Ratio' + makecell[2]] = f'{round(spec_precision_gemm_dict[target][precision] / spec_precision_gemm_dict[baseline][precision], 2) } ({spec_precision_gemm_dict[target][precision]} / {spec_precision_gemm_dict[baseline][precision]})'
    print(np.mean(math_bound_data[target]))
    print(np.mean(math_bound_data[baseline]))
    print(f"Math bound ratio: {ratio}")
    
    
    # output
    processed_data = []
    processed_data.append(output_line_math)
    processed_data.append(output_line_mem)    
    #print(json.dumps(processed_data ,indent = 4))
    return processed_data, fig