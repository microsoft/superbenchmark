import re

def calc_ratio(metric_friendly_name: str, target_value: float, baseline_value: float):
    
    ratio = 'NA'
    # reciprocal, dicipline: larger ratio denotes better performance, thus for time, latency, GPU memory usage metrics, need to calculate the reciprocal
    #time_latency_metric_unit_list = ['(s)', '(ms)', '(us)', '(ns)', 'GPU Memory Usage (GB)']
    #search for keywords in either metric friendly name, or table title prefix part
    cal_reciprocal = False
    pattern = r"(.*\(s\).*)|(.*\(ms\).*)|(.*\(us\).*)|(.*\(ns\).*)|(.*GPU Mem Usage \(GB\).*)"
    if re.match(pattern, metric_friendly_name):
        cal_reciprocal = True
        
    #print(metric_friendly_name)
    #print(cal_reciprocal)
    
    if (isinstance(target_value, int) or isinstance(target_value, float)) and (isinstance(baseline_value, int) or isinstance(baseline_value, float)):
        if target_value != 0 and baseline_value != 0:
            ratio_float = float(target_value)/float(baseline_value)
            if cal_reciprocal:
                ratio_float = 1/ratio_float
            ratio = "{:.2f}".format(ratio_float) + "X"
    return ratio

def gen_size_range(filter_min, filter_max):
    values = []
    current_value = filter_min

    while current_value <= filter_max:
        values.append(current_value)
        current_value *= 2
    return values

def find_value_in_json(regex, json_obj):
    pattern = re.compile(regex)
    for key in json_obj.keys():
        if pattern.match(key):
            return json_obj[key]
    return 'NA'

def analyze_comm_average(analysis, table_in, raw_table, target, baseline, header_first):

    if 'comm_busbw' in analysis:
        filter_min = 1048576        #1MB
        filter_max = 17179869184    #16GB
        metric_type = 'bw'
        size_range = gen_size_range(filter_min, filter_max)
    elif 'comm_lat' in analysis:
        filter_min = 1024           #1KB
        filter_max = 33554432       #32MB
        metric_type = 'lat'
        size_range = gen_size_range(filter_min, filter_max)
    elif 'multinode_ibwrite_busbw_average' in analysis:
        filter_min = 1024           #1KB
        filter_max = 1073741824     #1GB
        metric_type = 'bw'
        size_range = gen_size_range(filter_min, filter_max)
    elif 'multinode_ibwrite_lat_average' in analysis:
        filter_min = 2              #2B
        filter_max = 8388608        #8MB
        metric_type = 'lat'
        size_range = gen_size_range(filter_min, filter_max)
    elif 'inf_lat' in analysis:
        filter_min = 1024           #1KB
        filter_max = 33554432       #32MB
        metric_type = 'lat'
        size_range = ['2208-5608',
                        '11216-320',
                        '1536-4608',
                        '2208-4608',
                        '9216-320',
                        '9216-768',
                        '9216-430',
                        '32-4608', # copilot
                        '1104-5608',
                        '5608-320',
                        '768-4608',
                        '1104-4608',
                        '4608-320',
                        '46084608-768',
                        '9216-430',
                        '16-4608', # dist-copilot
                        '128-128', # llama-2-70b
                        '128-2048',
                        '2048-128',
                        '2048-2048',
                        '1', # llama3-80b
                        '2',
                        '4',
                        '8',
                        '16',
                        '32',
                        '64',
                        '128',
                        '256',
                        '512',
                        '1024',
                        '2048',
                        '4096'
                        ] 
    else:
        filter_min = 0
        filter_max = 34359738368    #32GB
        metric_type = 'bw'
        size_range = gen_size_range(filter_min, filter_max)
        
    if target in raw_table:
        raw_target = raw_table[target]
    else:
        raw_target = {}
    
    if baseline in raw_table:
        raw_baseline = raw_table[baseline]
    else:
        raw_baseline = {}
        
    table_out = []        
    for line in table_in:
        new_line = {}
        
        for line_key, pattern_tmp in line.items(): # for each key of allreduce, alltoall, allgather, prepare lists of values and ratios  
            new_line[header_first] = line_key
            ratio_lst = []
            for size in size_range:
                # assemble the regex for match
                if pattern_tmp.count('\\d+') == 2:
                    pattern = pattern_tmp.replace('\\d+-\\d+', str(size))
                else:
                    pattern = pattern_tmp.replace('\\d+', str(size))
                    
                # match and extract the value for target and baseline
                target_value = find_value_in_json(pattern, raw_target)
                baseline_value = find_value_in_json(pattern, raw_baseline)
                
                # calculate the ratio
                if (not isinstance(target_value, str)) and (not isinstance(baseline_value, str)):
                    if not metric_type == 'lat':
                        ratio = target_value / baseline_value
                    else:
                        ratio = baseline_value / target_value
                    ratio_lst.append(ratio)
            
            # calculate average ratio
            if len(ratio_lst) == 0:
                average_ratio = 'NA'
                average_ratio_str = average_ratio
            else:
                average_ratio = round((sum(ratio_lst) / len(ratio_lst)), 2)
                average_ratio_str = str(average_ratio) + 'X'
        new_line['Average Ratio'] = average_ratio_str
        table_out.append(new_line)

            
    return table_out

def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

def smart_round(value):
    rounded_value = round(value, 2)
    if (rounded_value == 0.0 or int(value) == 0):
        rounded_value = round(value, 3)
    return rounded_value

def calc_scalability(target_value: float, baseline_value: float):
    
    ratio_str = 'NA'
    ratio_float = 'NA'
    
    if (isinstance(target_value, int) or isinstance(target_value, float)) and (isinstance(baseline_value, int) or isinstance(baseline_value, float)):
        if target_value != 0 and baseline_value != 0:
            ratio_float = float(target_value)/float(baseline_value)
            ratio_str = "{:.2f}\%".format(ratio_float * 100)
    return ratio_str, ratio_float