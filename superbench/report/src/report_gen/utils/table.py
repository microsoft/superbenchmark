import copy
import re
import json
from .roofline import *
from .findmax import *
from .math import *
from .types import DCW

def gen_comparison_table_new(dcw: DCW, SKUNickName: dict, meta: dict, raw_table: dict, spec_table: dict={}, save_to: str='case'):
    
    target = dcw.Design.Target.lower()
    baseline = dcw.Design.Baseline.lower()
    
    target_nick = SKUNickName["Target"]
    baseline_nick = SKUNickName["Baseline"]
    
    
    table_output = copy.deepcopy(meta)
    tables = table_output["tables"]
    section = table_output["section"]
    makecell = [f"\\makecell[l]{{", f" \\\\ ", f"}}"] # make a two line cell
    fig_to_save = ''
    for table in tables:
        table_ins = ComparisonTable(section, table)
        table["label"] = table_ins.write_lable()

        
    

class ComparisonTable:
    def __init__(self, section, table):
        self.analysis_method = table["analyze_method"]
        self.lable = f"{section}-{table['index']}"        
        self.column_group = table["column_group"]
        self.table_to_save = ''
        self.fig_to_save = ''   
        
    def write_lable(self):
        return self.lable

def s2c_gpu_name(input_string):
    # convert small font before digits to capital font
    output_string = ""
    for char in input_string:
        if char.isdigit():
            output_string += input_string[input_string.index(char):]
            break
        else:
            output_string += char.upper()
    return output_string

def get_raw_data(sku: str, table_key: str, raw_table: dict):
    pattern = re.compile(table_key)
    value = 'NA'
    if sku in raw_table:
        table = raw_table[sku]
        for key in table.keys():
            #print(key)
            # If the key matches the pattern
            if pattern.search(key):
                value = table[key]
                if any(tflops in table_key for tflops in ['gemm-flops', 'cublaslt-gemm', 'cublaslt', 'hipblaslt']): # gemm benchmark
                    #print('True')
                    if isinstance(value, float) or isinstance(value, int):
                        value = round(value / 1000, 2)
                    else:
                        value = "NA"
                else: # non gemm benchmark
                    if isinstance(value, float) or isinstance(value, int):
                        #value = round(value, 2)
                        value = smart_round(value)
                    elif 'na' not in value:
                        pass
                    else:
                        value = "NA"
                #print(value)
                break
    else:        
        print(f"wrong table! please perform manual check on the raw table")
    return value

def change_json_key(file, target, baseline):
    # Load the JSON file into a Python dictionary
    with open(file, 'r') as f:
        data = json.load(f)

    # Replace the key in the dictionary
    data[target] = data.pop('target')
    data[baseline] = data.pop('baseline')

    # Write the dictionary back into the JSON file
    with open(file, 'w') as f:
        json.dump(data, f, indent = 4)