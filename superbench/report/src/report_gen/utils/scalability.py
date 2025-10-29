import re
import numpy as np
from .math import calc_ratio, smart_round, calc_scalability
from .table import s2c_gpu_name, get_raw_data

def analyze_calc_scalability(table, multi_data, single_data, target, baseline, header, ratio_reverse):
    
    #multi_data: raw json, multi node
    #single_data: raw json, single node
    
    target = target.lower()
    baseline = baseline.lower()
    
    # prepare the output
    table_out = table.copy()
    
    # get the metrics
    metrics = table_out["column_group"][0]["metric"]
    
    content = []
    # calculate the scalability table
    for metric in metrics:
        line_new ={}
        # extract metric
        metric_friendly_name = list(metric.keys())[0]
        metric_benchmark_name = metric[metric_friendly_name]

        # assemble line
        # table column 1
        line_new[header] = metric_friendly_name
        # target, multi-node, single node, scalability
        t_multi  = get_raw_data(target, metric_benchmark_name, multi_data)
        t_single = get_raw_data(target, metric_benchmark_name, single_data) 
        t_scalability_str, t_scalability = calc_scalability(t_multi, t_single)
        if ratio_reverse == "True":
            t_scalability_str, t_scalability = calc_scalability(t_single, t_multi)
        line_new[s2c_gpu_name(target)] = t_scalability_str
        # baseline, multi-node, single node, scalability
        b_multi  = get_raw_data(baseline, metric_benchmark_name, multi_data)
        b_single = get_raw_data(baseline, metric_benchmark_name, single_data) 
        b_scalability_str, b_scalability = calc_scalability(b_multi, b_single)
        if ratio_reverse == "True":
            b_scalability_str, b_scalability = calc_scalability(b_single, b_multi)
        line_new[s2c_gpu_name(baseline)] = b_scalability_str
        # data column, c, ratio
        line_new["Ratio"] = calc_ratio(table["title_prefix"] + metric_friendly_name, t_scalability, b_scalability)
        
        # append to table output
        #print(line_new)
        content.append(line_new)

    #print(content)
    table_out["content"] = content
    #print(table_out)
    return content

