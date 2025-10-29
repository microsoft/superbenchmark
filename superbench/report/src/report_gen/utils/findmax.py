import re
import numpy as np
from .math import calc_ratio

gemm_peak_pattern = r'(TFLOPS|TIOPS)'
cpu_stream_pattern = r'Bandwidth All NUMA'


def match_pattern(p, string):
    if re.search(p, string):
        return True
    else:
        return False
    
def extract_precision(s):
    match = re.search(r'GPU (Computation|Comp.|Comp)/(.*?) (TensorCore|XDLOPs|TC/MC)', s)
    if match:
        return match.group(2)
    else:
        return None
    
def count_element_types(lst):
    type_dict = {}
    for i in lst:
        if type(i) in type_dict:
            type_dict[type(i)] += 1
        else:
            type_dict[type(i)] = 1
    len_type_dict = len(type_dict)
    return len_type_dict

def change_string_to_zero(lst):
    return [0 if isinstance(i, str) else i for i in lst]
    
def find_max_num_index(lst):
    
    num_types = count_element_types(lst)
    if num_types > 1: # there are mixed types, both float, int or string exists, then change string elements to float 0
        lst = change_string_to_zero(lst)
        
    # then find the max value and index
    max_num = max(lst)
    max_index = lst.index(max_num)        
    return max_num, max_index
    
def find_max(sku, precision, default_peak_value, data):
    precision = precision.lower()
    gemm_lt_pattern_ = f'(hipblaslt|cublaslt)-gemm:*.*/{precision}_(\d+)_(\d+)_(\d+)_(\d+)_flops'
    if precision == 'fp32':
        gemm_lt_pattern_ = f'(hipblaslt)-gemm:*.*/{precision}_(\d+)_(\d+)_(\d+)_(\d+)_flops' # hipblaslt fp32 is testing fp32
    if precision == 'tf32/fp32':
        gemm_lt_pattern_ = f'(hipblaslt|cublaslt)-gemm:*.*/{precision.replace("tf32/fp32", "fp32")}_(\d+)_(\d+)_(\d+)_(\d+)_flops' # cublaslt fp32 is testing tf32
    if precision == 'fp8':
        gemm_lt_pattern_ = f'(hipblaslt|cublaslt)-gemm:*.*/(fp8|fp8e4m3|fp8e5m2)_(\d+)_(\d+)_(\d+)_(\d+)_flops'
    if precision == 'fp4':
        gemm_lt_pattern_ = f'(hipblaslt|cublaslt)-gemm:*.*/(fp4|fp4e2m1)_(\d+)_(\d+)_(\d+)_(\d+)_flops'

    
    value_list = [default_peak_value]
    m_list = [0]
    n_list = [0]
    k_list = [0]
    for metric, value in data.items():
        match = re.match(gemm_lt_pattern_, metric)
        if match:
            group_values = match.groups()
            value_list.append(value)
            if precision == 'fp8':
                m_list.append(int(group_values[3]))
                n_list.append(int(group_values[4]))
                k_list.append(int(group_values[5]))
            else:
                m_list.append(int(group_values[2]))
                n_list.append(int(group_values[3]))
                k_list.append(int(group_values[4]))
            
    # find max
    [max_tflops, max_index] = find_max_num_index(value_list)
    
    # print out the shapes
    if max_index ==0:
        print(f"SKU is {sku}, precision is {precision}, peak value achieved using default shape value")
    else:
        print(f"SKU is {sku}, precision is {precision}, peak value achieved using shape of m={m_list[max_index]}, n={n_list[max_index]}, k={k_list[max_index]}")
    
    if isinstance(max_tflops, int) or isinstance(max_tflops, float):
        max_tflops = round(max_tflops, 1)
    return max_tflops

def find_max_cpu_stream(sku, pattern, default_peak_value, data):
    pattern = pattern.lower()
    stream_pattern_ = f'cpu-stream:spread.*'

    value_list = [default_peak_value]
    for metric, value in data.items():
        match = re.match(stream_pattern_, metric)
        if match:
            value_list.append(value)

    # find max
    [max_throughput, max_index] = find_max_num_index(value_list)

    if isinstance(max_throughput, int) or isinstance(max_throughput, float):
        max_throughput = round(max_throughput, 1)
    return max_throughput

def analyze_find_max(content, data, target, baseline, projected_table_object: str='target'):
    
    #content: default table from a generate table generaion process
    #data: raw json
    
    target = target.lower()
    baseline = baseline.lower()
    
    content_out = []
    
    for line in content:
        new_line = {}
        metric_name = line["Metric"]
        # do this only for GEMM peak
        if match_pattern(gemm_peak_pattern, metric_name):
            precision = extract_precision(metric_name)
            for key, value in line.items():
                key_lower = key.lower()
                if (target not in key_lower) and (baseline not in key_lower) and ('ratio' not in key_lower) and ('projected perf.' not in key_lower) and ('spec. perf.' not in key_lower): # 1st column: metric name
                    new_line[key] = value
                elif (target in key_lower): #2nd column: target SKU value
                    max_value_target = find_max(target, precision, value, data[target])
                    new_line[key] = max_value_target
                elif (baseline in key_lower): #3rd column: baseline SKU value
                    max_value_baseline = find_max(baseline, precision, value, data[baseline])
                    new_line[key] = max_value_baseline
                elif ('projected perf.' in key_lower):
                    new_line[key] = value
                    value_projected = value
                elif ('spec. perf.' in key_lower):
                    new_line[key] = value
                    value_spec = value
                elif ('perf. ratio' in key_lower): #column: ratio
                    new_line[key] = calc_ratio(key, max_value_target, max_value_baseline)
                elif ('ratio to projected' in key_lower): # column: ratio
                    if projected_table_object == 'target':
                        new_line[key] = calc_ratio(key, max_value_target, value_projected)
                    elif projected_table_object == 'baseline':
                        new_line[key] = calc_ratio(key, max_value_baseline, value_projected)
                elif ('ratio to spec.' in key_lower): #column: ratio
                    if projected_table_object == 'target':
                        new_line[key] = calc_ratio(key, max_value_target, value_spec)
                    elif projected_table_object == 'baseline':
                        new_line[key] = calc_ratio(key, max_value_baseline, value_spec)
        # do this for cpu-stream, all numa peak
        elif match_pattern(cpu_stream_pattern, metric_name):
            pattern = 'stream'
            for key, value in line.items():
                key_lower = key.lower()
                if (target not in key_lower) and (baseline not in key_lower) and ('ratio' not in key_lower) and ('projected perf.' not in key_lower) and ('spec. perf.' not in key_lower): # 1st column: metric name
                    new_line[key] = value
                elif (target in key_lower): #2nd column: target SKU value
                    max_value_target = find_max_cpu_stream(target, pattern, value, data[target])
                    new_line[key] = max_value_target
                elif (baseline in key_lower): #3rd column: baseline SKU value
                    max_value_baseline = find_max_cpu_stream(baseline, pattern, value, data[baseline])
                    new_line[key] = max_value_baseline
                elif ('projected perf.' in key_lower):
                    new_line[key] = value
                    value_projected = value
                elif ('spec. perf.' in key_lower):
                    new_line[key] = value
                    value_spec = value
                elif ('perf. ratio' in key_lower): #column: ratio
                    new_line[key] = calc_ratio(key, max_value_target, max_value_baseline)
                elif ('ratio to projected' in key_lower): # column: ratio
                    if projected_table_object == 'target':
                        new_line[key] = calc_ratio(key, max_value_target, value_projected)
                    elif projected_table_object == 'baseline':
                        new_line[key] = calc_ratio(key, max_value_baseline, value_projected)
                elif ('ratio to spec.' in key_lower): #column: ratio
                    if projected_table_object == 'target':
                        new_line[key] = calc_ratio(key, max_value_target, value_spec)
                    elif projected_table_object == 'baseline':
                        new_line[key] = calc_ratio(key, max_value_baseline, value_spec)
        # keep the value for metrics not GEMM peak
        else:
            new_line = line
        content_out.append(new_line)
    
    return content_out

