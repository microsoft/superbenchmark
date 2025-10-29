import json
import re
import os

def get_value_from_jsonl(file_path, search_key):
    """
    Reads a JSONL file and returns the value for a specified key.

    :param file_path: Path to the JSONL file.
    :param search_key: The key to search for in the JSON objects.
    :return: The value associated with the search_key, or None if the key is not found.
    """
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Parse the JSON object from the line
                json_obj = json.loads(line)
                
                # Check if the key exists in the JSON object
                if search_key in json_obj:
                    return json_obj[search_key]
        
        # If the key is not found in any JSON object
        return None

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print("Error decoding JSON from file.")
        return None

def open_json(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def modify_nccl_keys(data):
    new_data = {}
    for sku_name, sku_data in data.items():
        new_sku_data = {}
        for key, value in sku_data.items():
            new_key = re.sub(r'(nccl-bw:)(nvlink-)(allgather|allreduce|alltoall|broadcast|reduce|reducescatter)(/)(allgather|allreduce|alltoall|broadcast|reduce|reducescatter)(_\d+)(_busbw)', r'\1bw:\3:nvlink/\5\6\7', key)
            new_key = re.sub(r'(nccl-bw:)(nvlink-)(allgather|allreduce|alltoall|broadcast|reduce|reducescatter)(/)(allgather|allreduce|alltoall|broadcast|reduce|reducescatter)(_\d+)(_time)', r'\1lat:\3:nvlink/\5\6\7', new_key)
            new_sku_data[new_key] = value
        new_data[sku_name] = new_sku_data

    return new_data

def remove_gemm_flops_key(data):
    key = "gemm-flops/fp16_flops"
    for sku_name, sku_data in data.items():
        if key in sku_data:
            del sku_data[key]
    return data

def add_new_keys(data):
    for sku_name, sku_data in data.items():
        keys = list(sku_data.keys())
        index = keys.index("mem-bw/h2d_bw") + 1 if "mem-bw/h2d_bw" in keys else len(keys)
        sku_data = {k: sku_data[k] for k in keys[:index]} 
        sku_data["mem-bw/d2h_bw_remote"] = 0
        sku_data["mem-bw/h2d_bw_remote"] = 0
        sku_data.update({k: data[sku_name][k] for k in keys[index:]})
        data[sku_name] = sku_data
    #print('look for mem-bw/(d2h|h2d)_bw:(0|2)')
    return data

def modify_dict_with_jsonl(dict_data, jsonl_file, target_key, jsonl_key):
    """
    Modifies a key's value in the dictionary with a value from a JSONL file.

    :param dict_data: The dictionary to modify.
    :param jsonl_file: Path to the JSONL file.
    :param target_key: The key in the dictionary whose value needs to be changed.
    :param jsonl_key: The key in the JSONL file to get the new value from.
    :return: The modified dictionary.
    """
    # Get the new value from the JSONL file
    new_value = get_value_from_jsonl(jsonl_file, jsonl_key)

    if new_value is not None:
        # Traverse the dictionary and update the target key's value
        for sku_name, sku_data in dict_data.items():
            if target_key in sku_data:
                sku_data[target_key] = new_value

    return dict_data

def modify_d2hh2d(data, raw_result_file):
    data = modify_dict_with_jsonl(data, raw_result_file, "mem-bw/d2h_bw", "gpu-copy-bw:perf/gpu0_to_cpu_by_sm_under_numa1_bw")
    data = modify_dict_with_jsonl(data, raw_result_file, "mem-bw/h2d_bw", "gpu-copy-bw:perf/cpu_to_gpu0_by_sm_under_numa1_bw")
    data = modify_dict_with_jsonl(data, raw_result_file, "mem-bw/d2h_bw_remote", "mem-bw/d2h_bw:0")
    data = modify_dict_with_jsonl(data, raw_result_file, "mem-bw/h2d_bw_remote", "mem-bw/h2d_bw:0")
    return data

def save_json(new_data, output_file):
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=2)

def main(DEBUG=True):
    if DEBUG:
        # test
        res_dir = '/home/lequ/tmp/tmp6_gb200_nccl/'
        input_file = os.path.join(res_dir, 'input.json')
        jsonl_file = os.path.join(res_dir, 'jsonl.jsonl')
        output_file = os.path.join(res_dir, 'output.json')
    else:
        # real
        res_dir = '/home/lequ/_git/Lucia-Agents/agents/lucia-report-gen-agent/src/report_gen/data/gb200v09vm_h100/raw/backup/20250319a/'
        input_file = os.path.join(res_dir, 'extract_summary_metrics_result-vm-0.9.json')
        jsonl_file =  os.path.join(res_dir, 'results-summary-vm-0.9.jsonl')
        output_file =  os.path.join(res_dir, 'output.json')
    
    # load
    dict_1 = open_json(input_file)
    # modify
    dict_2 = modify_nccl_keys(dict_1)
    dict_3 = remove_gemm_flops_key(dict_2)
    dict_4 = add_new_keys(dict_3)
    dict_5 = modify_d2hh2d(dict_4, jsonl_file)
    # save
    dict_out = dict_5
    save_json(dict_out, output_file)

if __name__ == "__main__":
    main(False)