import copy
import re
import json
from .roofline import *
from .findmax import *
from .scalability import *
from .math import *
from .types import DCW
from .table import *
from datetime import date

def convert_table():
    print("hello convert")

def table_replace_key(target: str, baseline: str, input_table_json: dict):
    output_table_json = {}
    if len(input_table_json) == 2:
        [key1, key2] = input_table_json.keys()            
        output_table_json[target] = input_table_json[key1]
        output_table_json[baseline] = input_table_json[key2]
        
    elif len(input_table_json) == 1:
        [key] = input_table_json.keys()
        output_table_json[target] = input_table_json[key]
    
    return output_table_json

def analyze_roof_line_fake(table_in: list, target, baseline):
    raw = table_in.copy()

    processed_data = []
    line={}
    line["Kernels"] = "Math-Limited Kernels"
    line[target] = 23
    line[baseline] = 34
    line["Average Speedup"] = '0.5X'
    processed_data.append(line)
    line={}
    line["Kernels"] = "Mmemory-Limited Kernels"
    line[target] = 5588
    line[baseline] = 7766
    line["Average Speedup"] = '1.8X'
    processed_data.append(line)
    
    print(json.dumps(processed_data ,indent = 4))
    return processed_data, fake_fig

def make_header_narrow(table, makecell):
    output_table = []
    for line in table:
        new_line = {}
        for key, value in line.items():
            if ' ' in key:
                key_splited = key.split(' ')
                new_key = makecell[0] + key.replace(' ', makecell[1]) + makecell[2]
                new_line[new_key] = value
            else:
                new_line[key] = value
        output_table.append(new_line)
        
    return output_table

def get_testbed_name_source_a(target, baseline, source):
    if source is None:
        testbed = target
    elif 'baseline' in source:
        testbed = baseline
    else:
        testbed = target
    return testbed

def get_testbed_name_source_b(target, baseline, source):
    if source is None:
        testbed = baseline
    elif 'target' in source:
        testbed = target
    else:
        testbed = baseline
    return testbed

def get_testbed_name_source_c(target, baseline, source):
    return get_testbed_name_source_b(target, baseline, source)

def get_table_dict(source, raw_table, projected_table, spec_table):
    if source is None:
        table = raw_table
    elif 'projected' in source:
        table = projected_table
    elif 'spec' in source:
        table = spec_table
    else:
        table = raw_table
    return table

def get_table_header(default, target, baseline, source):
    if source is None:
        header = default   
    elif 'benchmark' in source:
        if 'target' in source:
            header = target
        elif 'baseline' in source:
            header = baseline
    elif 'projected' in source:
        header = "Projected Perf."
    elif 'spec' in source:
        header = "Spec. Perf."
    else:
        header = default        

    if source is None:
        header_ratio = "Perf. Ratio"
    elif 'projected' in source:
        header_ratio = "Ratio to Projected"
    elif 'spec' in source:
        header_ratio = "Ratio to Spec."
    else:
        header_ratio = "Perf. Ratio"
        
    return [header, header_ratio]

def get_collective_alg_name(testbed):
    hit_nvidia = False
    for temp in ['v100', 'a100', 'h100', 'h200', 'b200', 'gb200']:
        if temp in testbed.lower():
            hit_nvidia = True
            return 'NCCL'
    hit_amd = False
    for temp in ['mi100', 'mi200', 'mi250x', 'mi300x', 'mi375x', 'mi375']:
        if temp in testbed.lower():
            hit_amd = True
            return 'RCCL'
    if (not hit_nvidia) and (not hit_amd):
        return 'CCL'

def set_scale_raw_table(raw_table, scale_value):
    selected_raw_table = {}

    # get two skus
    skus = list(raw_table.keys())
    target_table = raw_table[skus[0]]
    baseline_table = raw_table[skus[1]]
    #print(skus)
    # set scale key
    scale_key = f'scale {scale_value}'
    # set output sub table according to the scale value
    output_target_table = target_table[scale_key]
    output_baseline_table = baseline_table[scale_key]
    # assemble output table
    selected_raw_table[skus[0]] = output_target_table
    selected_raw_table[skus[1]] = output_baseline_table
    
    return selected_raw_table

def gen_comparison_table(dcw: DCW, SKUNickName: dict, meta: dict, input_table: dict, projected_table: dict={}, spec_table: dict={}, save_to: str='case'):

    target = dcw.Design.Target.lower()
    baseline = dcw.Design.Baseline.lower()
    
    target_nick = SKUNickName["Target"]
    baseline_nick = SKUNickName["Baseline"]
    
    makecell = [f"\\makecell[l]{{", f" \\\\ ", f"}}"] # make a two line cell
    
    table_output = copy.deepcopy(meta)
    tables = table_output["tables"]
    section = table_output["section"]
    fig_to_save = ''
    for table in tables:
        # get the analysis method
        analyze_method = table["analyze_method"] if "analyze_method" in table else None
        # get the source definition
        source_a = table["source_a"] if "source_a" in table else None
        source_b = table["source_b"] if "source_b" in table else None
        source_c = table["source_c"] if "source_c" in table else None  
        # get table index
        index = table["index"]
        # get the table groups
        column_group = table["column_group"]
        # get the table header first cell
        header_first = table["header_1"] if 'header_1' in table else "Metric"
        ratio_reverse = table["ratio_reverse"] if 'ratio_reverse' in table else "False"
        # get shrink header flag
        shrink_header = table["shrink_header"] if 'shrink_header' in table else None

        # get scale for multi node
        if "scale" in table:
            scale_value = table["scale"]
        else: scale_value = 1

        # set raw table to specific raw table
        #print(index)
        raw_table = set_scale_raw_table(input_table, scale_value)
        # scale 1 as the baseline when calculate scalability
        baseline_table_for_scalability = set_scale_raw_table(input_table, 1)

        # set the table title
        if source_a is not None and source_b is not None:
            if 'spec' in source_a and 'spec' in source_b:
                pass
            else:
                title_testbed = target if "target" in source_a else baseline
                table["title_prefix"] = f"{table['title_prefix']} of {s2c_gpu_name(title_testbed)}".replace("_", r"\_")
        else:
            table["title_prefix"] = f"{table['title_prefix']} of {s2c_gpu_name(target)} over {s2c_gpu_name(baseline)}".replace("_", r"\_")

        # set the table lable
        table["label"] = f"{section}-{index}" 

        # set the table content, empty
        table["content"] = []

        # prepare the attributes for getting the value
        testbed_a = get_testbed_name_source_a(target, baseline, source_a)
        raw_table_a = get_table_dict(source_a, raw_table, projected_table, spec_table)
        table_header_name_a = get_table_header(target, target, baseline, source_a)
        
        testbed_b = get_testbed_name_source_b(target, baseline, source_b)
        raw_table_b = get_table_dict(source_b, raw_table, projected_table, spec_table)
        table_header_name_b = get_table_header(baseline, target, baseline, source_b)
        
        testbed_c = get_testbed_name_source_c(target, baseline, source_c)
        raw_table_c = get_table_dict(source_c, raw_table, projected_table, spec_table)
        table_header_name_c = get_table_header(baseline, target, baseline, source_c)
        
        # if both header a and header b are the same, change hearder a to target, change hearder b to baseline
        if table_header_name_a[0] == table_header_name_b[0]:
            table_header_name_a[0] = target
            table_header_name_b[0] = baseline
            table_header_name_a[1] = 'Ratio'
            table_header_name_b[1] = 'Ratio'
        
        if len(column_group) == 1: # basic table, has only one column for target, one column for baseline
            print(f"length of column_group: {len(column_group)}")
            column = column_group[0]                
            for metric in column["metric"]:
                #print(metric.keys())
                line_new ={}
                # extract metric
                metric_friendly_name = list(metric.keys())[0]
                metric_benchmark_name = metric[metric_friendly_name]

                # assemble line
                # table column 1
                line_new["Metric"] = metric_friendly_name
                # data column, a
                a_value = get_raw_data(testbed_a, metric_benchmark_name, raw_table_a) 
                line_new[s2c_gpu_name(table_header_name_a[0])] = a_value
                # data column, b
                b_value = get_raw_data(testbed_b, metric_benchmark_name, raw_table_b) 
                line_new[s2c_gpu_name(table_header_name_b[0])] = b_value
                # data column, c, optional
                if source_c is not None:
                    c_value = get_raw_data(testbed_c, metric_benchmark_name, raw_table_c)
                    line_new[s2c_gpu_name(table_header_name_c[0])] = c_value
                # ratio column, a to b
                line_new[table_header_name_b[1]] = calc_ratio(table["title_prefix"] + metric_friendly_name, a_value, b_value)
                # ratio column, a to c, optional
                if source_c is not None:
                    line_new[table_header_name_c[1]] = calc_ratio(table["title_prefix"] + metric_friendly_name, a_value, c_value)

                # append to table output
                table["content"].append(line_new)

        # apply analysis if needed, this value needs to be set in the table template files.        
        if analyze_method == 'roofline':
            print("performing roofline analysis")
            [content_from_roofline, fig_to_save] = analyze_roof_line(raw_table, s2c_gpu_name(target), s2c_gpu_name(baseline), target_nick, baseline_nick)
            table["content"] = content_from_roofline
        if analyze_method == 'roofline_fp8':
            print("performing roofline fp8 analysis")
            [content_from_roofline, fig_to_save] = analyze_roof_line(raw_table, s2c_gpu_name(target), s2c_gpu_name(baseline), target_nick, baseline_nick, precision='fp8')
            table["content"] = content_from_roofline
        if analyze_method == 'findmax':
            print("performing findmax analysis")
            content_from_find_max = analyze_find_max(table["content"], raw_table, s2c_gpu_name(target), s2c_gpu_name(baseline))
            table["content"] = content_from_find_max
        if analyze_method == 'findmax_target_spec':
            print("performing findmax analysis")
            content_from_find_max = analyze_find_max(table["content"], raw_table, s2c_gpu_name(target), s2c_gpu_name(baseline), 'target')
            table["content"] = content_from_find_max
        if analyze_method == 'findmax_baseline_spec':
            print("performing findmax analysis")
            content_from_find_max = analyze_find_max(table["content"], raw_table, s2c_gpu_name(target), s2c_gpu_name(baseline), 'baseline')
            table["content"] = content_from_find_max
        if analyze_method in ['comm_busbw_average', 'comm_lat_average', 'inf_lat_average', 'multinode_ibwrite_busbw_average', 'multinode_ibwrite_lat_average']:
            print("performing comm average analysis")
            content_from_average = analyze_comm_average(analyze_method, table["column_group"][0]["metric"], raw_table, target, baseline, header_first)
            table["content"] = content_from_average
        if analyze_method == 'calc_scalability':
            print("performing scalability analysis")
            content_from_scalability = analyze_calc_scalability(table, raw_table, baseline_table_for_scalability, target, baseline, header_first, ratio_reverse)
            table["content"] = content_from_scalability
        if analyze_method in ['msccl', 'inference']:
            table_temp = {}
            base_column = None
            target_column = None
            for column in column_group:
                column_name = column["column_name"]
                table_temp[column_name] = []
                for idx, metric in enumerate(column["metric"]):
                    #print(metric.keys())
                    line = {}
                    #
                    metric_friendly_name = list(metric.keys())[0]
                    metric_benchmark_name = metric[metric_friendly_name]
                    target_value = get_raw_data(target, metric_benchmark_name, raw_table)
                    baseline_value = get_raw_data(baseline, metric_benchmark_name, raw_table)
                    
                    line[header_first] = metric_friendly_name
                    #line[makecell[0] + s2c_gpu_name(target)   + makecell[1] + column_name + makecell[2]] = target_value
                    #line[makecell[0] + s2c_gpu_name(baseline) + makecell[1] + column_name + makecell[2]] = baseline_value
                    
                    line[makecell[0] + s2c_gpu_name(target)   + makecell[1] + (column_name.replace('RCCL/NCCL', get_collective_alg_name(target)) if 'RCCL/NCCL' in column_name else column_name) + makecell[2]] = target_value
                    
                    line[makecell[0] + s2c_gpu_name(baseline) + makecell[1] + (column_name.replace('RCCL/NCCL', get_collective_alg_name(baseline)) if 'RCCL/NCCL' in column_name else column_name) + makecell[2]] = baseline_value
                    
                    line[makecell[0] + "Perf. Ratio" + makecell[1] + s2c_gpu_name(target) + '(' + (column_name.replace('RCCL/NCCL', get_collective_alg_name(target)) if 'RCCL/NCCL' in column_name else column_name) + ')\\\\/' + s2c_gpu_name(baseline) + '(' + (column_name.replace('RCCL/NCCL', get_collective_alg_name(baseline)) if 'RCCL/NCCL' in column_name else column_name)+ ')' + makecell[2]] = calc_ratio(table["title_prefix"] + metric_friendly_name, target_value, baseline_value)
                    table_temp[column_name].append(line)
            #print(json.dumps(table_temp, indent = 4))

            # combine table_temp into table["content"]
            group_keys = list(table_temp.keys())
            for column_1, column_2 in zip(table_temp[group_keys[0]], table_temp[group_keys[1]]):
                new_line = {}
                column1_keys = list(column_1.keys())
                column2_keys = list(column_2.keys())
                #print(column1_keys)
                #print(column2_keys)
                new_line[column1_keys[0]] = column_1[column1_keys[0]] # metric name column

                new_line[column1_keys[1]] = column_1[column1_keys[1]] # group 1: target
                new_line[column1_keys[2]] = column_1[column1_keys[2]] # group 1: baseline

                new_line[column2_keys[1]] = column_2[column2_keys[1]] # group 2: target
                new_line[column2_keys[2]] = column_2[column2_keys[2]] # group 2: baseline

                new_line[column1_keys[3]] = column_1[column1_keys[3]] # group 1: ratio
                new_line[column2_keys[3]] = column_2[column2_keys[3]] # group 2: ratio
                table["content"].append(new_line)

        elif len(column_group) == 2: # advanced table, has two columns for target, two columns for baseline
            table_temp = {}
            for column in column_group:
                column_name = column["column_name"]
                table_temp[column_name] = []            
                for metric in column["metric"]:
                    #print(metric.keys())
                    line = {}
                    #
                    metric_friendly_name = list(metric.keys())[0]
                    metric_benchmark_name = metric[metric_friendly_name]
                    target_value = get_raw_data(target, metric_benchmark_name, raw_table)
                    baseline_value = get_raw_data(baseline, metric_benchmark_name, raw_table)
                    line[header_first] = metric_friendly_name
                    #line[makecell[0] + s2c_gpu_name(target)   + makecell[1] + column_name + makecell[2]] = target_value
                    #line[makecell[0] + s2c_gpu_name(baseline) + makecell[1] + column_name + makecell[2]] = baseline_value
                    line[makecell[0] + s2c_gpu_name(target)   + makecell[1] + (column_name.replace('RCCL/NCCL', 'RCCL') if 'RCCL/NCCL' in column_name else column_name) + makecell[2]] = target_value
                    line[makecell[0] + s2c_gpu_name(baseline) + makecell[1] + (column_name.replace('RCCL/NCCL', 'NCCL') if 'RCCL/NCCL' in column_name else column_name) + makecell[2]] = baseline_value
                    line[makecell[0] + "Perf. Ratio"          + makecell[1] + column_name + makecell[2]] = calc_ratio(table["title_prefix"] + metric_friendly_name, target_value, baseline_value)
                    table_temp[column_name].append(line)
            #print(json.dumps(table_temp, indent = 4))

            # combine table_temp into table["content"]
            group_keys = list(table_temp.keys())
            for column_1, column_2 in zip(table_temp[group_keys[0]], table_temp[group_keys[1]]):
                new_line = {}
                column1_keys = list(column_1.keys())
                column2_keys = list(column_2.keys())
                #print(column1_keys)
                #print(column2_keys)
                new_line[column1_keys[0]] = column_1[column1_keys[0]] # metric name column
                
                new_line[column1_keys[1]] = column_1[column1_keys[1]] # group 1: target
                new_line[column1_keys[2]] = column_1[column1_keys[2]] # group 1: baseline
                
                new_line[column2_keys[1]] = column_2[column2_keys[1]] # group 2: target
                new_line[column2_keys[2]] = column_2[column2_keys[2]] # group 2: baseline
                
                new_line[column1_keys[3]] = column_1[column1_keys[3]] # group 1: ratio
                new_line[column2_keys[3]] = column_2[column2_keys[3]] # group 2: ratio
                table["content"].append(new_line)

        if save_to == 'latex': # revise table to split only headers in to sub cells
            if shrink_header == 'True':
                table['content'] = make_header_narrow(table['content'], makecell)
            
    return table_output, fig_to_save

def reorder_json(data):
    # Get the key order of the first item
    key_order = list(data[0].keys())

    # Reorder the items based on the key order of the first item
    reordered_data = []
    for item in data:
        reordered_item = {key: item[key] for key in key_order}
        reordered_data.append(reordered_item)
    
    return reordered_data

def convert_json_to_latex(table_json):
    
    table_json = reorder_json(table_json)
    
    # Get the headers
    headers = list(table_json[0].keys())

    # Start the table
    latex_table = "  \\begin{tabular}{" + " ".join(["l"] * 1 + ["c"] * (len(headers) - 1)) + "}\n"
    latex_table += "\\toprule\n" + " & ".join(headers) + " \\\\\n\\midrule\n"

    # Add the data
    for item in table_json:
        # check if the first key, value represents a breaker
        first_key = list(item.keys())[0]
        first_value = item[first_key]
        if first_value.lower() == 'breaker': # add a midrule line breaker
            latex_table += "\\midrule\n"
        else: # for normal content
            latex_table += " & ".join([str(value) for value in item.values()]) + " \\\\\n"

    # End the table
    latex_table += "\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"


    return latex_table

def gen_write_tables_latex(tables_json: dict):
    
    tables = tables_json["tables"]
    table_str_to_section = ''
    table_str_to_appendix = ''
    for table in tables:
        if 'minipage' in table:
            [minipage_c, minipage_t] = table["minipage"].split("/") # minipage control word: c=current, t=total
            minipage_c = int(minipage_c)
            minipage_t = int(minipage_t)
            #calculate width and distance for minipages
            width = (1 - 0.1) / minipage_t
            dis = 0.1 / (minipage_t - 1)
        else:
            [minipage_c, minipage_t] = [0, 0]
            dis = 0
            width = 0
        

        
        out_str = ''
        caption = table["title_prefix"]
        label = table["label"]
        #print(label)
        save_to_appendix = table["save to appendix"]
        table_lines = convert_json_to_latex(table["content"]).replace("_", r"\_")
        
        if minipage_t == 0: # save tables normally
            if save_to_appendix == "True":
                table_str_to_appendix += f"\n\n\\begin{{table}}\n\\center\n  \\caption{{{caption}}}\n  \\label{{tab:{label}}}"            
                table_str_to_appendix += f"\n{table_lines}"
            else:
                table_str_to_section += f"\n\n\\begin{{table}}\n\\center\n  \\caption{{{caption}}}\n  \\label{{tab:{label}}}"
                table_str_to_section += f"\n{table_lines}"
        else: # save tables to minipages
            table_lines = table_lines.replace("\\end{table}\n\n", "")
            if minipage_c == 1: # only print \n before first minipage
                minipage_begin = f"\n\n"
            else:
                minipage_begin = f""
            if minipage_c == minipage_t: # only print hspace in between
                minipage_middle = f""
            else:
                minipage_middle = f"\\hspace{{{dis}\\textwidth}}\n"
            
            if save_to_appendix == "True":
                table_str_to_appendix += f"{minipage_begin}\\begin{{minipage}}{{{width}\\textwidth}}\n\\center\n  \\captionof{{table}}{{{caption}}}\n  \\label{{tab:{label}}}"
                table_str_to_appendix += f"\n{table_lines}\\end{{minipage}}\n{minipage_middle}"
            else:
                table_str_to_section += f"{minipage_begin}\\begin{{minipage}}{{{width}\\textwidth}}\n\\center\n  \\captionof{{table}}{{{caption}}}\n  \\label{{tab:{label}}}"
                table_str_to_section += f"\n{table_lines}\\end{{minipage}}\n{minipage_middle}"
    
    return table_str_to_section, table_str_to_appendix   

def amend_aggregated_metrics(meta, processed_table):
    # load the meta for all tables in each section
    # tables_meta = meta["tables"]
    # load processed raw for all tables in each section
    tables = processed_table['tables']

    for table in tables:
        column_group = table['column_group']
        content = table['content']
        if len(column_group) == 1: # only one column group
            aggregated_content = amend_content(content, column_group[0])
            table['content'] = aggregated_content
    return processed_table

def amend_content(content, column_groups):
    # get the line template from existing comparison table
    line_template = list(content[0].keys())
    num_column = len(line_template)
    
    if 'aggregated_metric' in column_groups:
        
        # gen line breaker
        breaker = {}
        for i in range(num_column):
            if i == 0:
                breaker[line_template[i]] = 'breaker'
            else:
                breaker[line_template[i]] = ''
        content.append(breaker)        
        
        # gen aggregated line
        aggregated_lines = column_groups['aggregated_metric']
        for aggregated_line in aggregated_lines:
            for ag_key, ag_value in aggregated_line.items():
                line = {}
                for i in range(num_column):
                    if i == 0:
                        line[line_template[i]] = ag_key
                    elif i == (num_column - 1):
                        line[line_template[i]] = calc_avg_on_comparison_table(content, ag_value)
                    else:
                        line[line_template[i]] = ''
                content.append(line)
        return content
    else:
        return content
    
def calc_avg_on_comparison_table(content: dict, keys: list):
    sum_value = []
    # traverse all lines in the comparison table
    for line in content:
        # check whether the metric is in the keys
        metric_name = list(line.values())[0].lower()
        metric_match = False
        for key in keys:
            pattern = re.compile(key.lower())
            if pattern.match(metric_name):
                metric_match = True
                break

        # check the value
        metric_value = list(line.values())[-1].lower().replace('x', '')
        
        # put the value into sum if not NA, and metric name matches
        if (not metric_value.lower() == 'na') and (not metric_value.lower() == '') and (metric_match):
            sum_value.append(float(metric_value))

    # calculate the average value
    if len(sum_value) == 0:
        avg_value = 'NA'
    else:
        avg_value = round(sum(sum_value)/len(sum_value), 2)
        avg_value = str(avg_value) + 'X'
    return avg_value

def process_latex_file_single(input_latex: str) -> str:

    # Find the table header
    header_pattern = re.compile(r'\\toprule\n(.+?)\\midrule', re.DOTALL)
    match = header_pattern.search(input_latex)
    if match:
        header = match.group(1)

        # Split the header into multiple lines
        new_header = ' & '.join(['\\makecell[l]{{{}}}'.format(cell.replace(" ", " \\\\")) for cell in header.split(' & ')])

        # Replace the old header with the new one
        output_latex = input_latex.replace(header, new_header)
    else:
        output_latex = input_latex

    return output_latex

def process_latex_file(input_latex: str) -> str:

    # Define a callback function to process each match
    def process_match(match):
        header = match.group(1)
        new_header = ' & '.join(['\\makecell[l]{{{}}}'.format(cell.replace(" ", " \\\\")) for cell in header.split(' & ')])
        print(new_header)
        return '\\toprule\n' + new_header + '\\'

    # Use re.sub() with the callback function
    output_latex = re.sub(r'\\toprule\n(.+?) \\', process_match, input_latex, flags=re.DOTALL)

    return output_latex

def change_build_date(input_latex: str) -> str:
    
    # Replace the target string
    input_latex = input_latex.replace('\\today', date.today().strftime("%B %d, %Y"))
    return input_latex

def remove_header_tail(text, begin_str, end_str):
    if text.startswith(begin_str):
        text = text[len(begin_str):]
    if text.endswith(end_str):
        text = text[:-len(end_str)]
    return text

def split_tex_content(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # define the possible dividers
    dividers = ['\\begin{table}', '\\begin{minipage}']
    
    # initialize the split parts
    before_table = content
    after_table = ''
    
    # find the first occurrence of any divider
    for divider in dividers:
        if divider in content:
            parts = content.split(divider, 1)
            before_table = parts[0]
            after_table = divider + parts[1]
            break
    
    return before_table, after_table

def append_after_table_content(tex_file, after_table):
    # open the file in append mode
    with open(tex_file, 'a') as file:
        # append the after_table string to the end of the file
        file.write(f'\n\n')
        file.write(after_table)