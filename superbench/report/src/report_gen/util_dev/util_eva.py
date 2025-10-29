import re
import json

def match_must_include(truth, res):
    # match if all key word in truth exists in res
    truth_list = truth.lower().split()
    truth_set = set(truth_list)
    
    res_list = res.lower().split()
    res_set = set(res_list)
    match = False
    if truth_set <= res_set:
        match = True
    return match

def match_must_include_result(truth, res):
    
    # Load json1 and json2 if they are not already python dictionaries
    if isinstance(truth, str):
        truth = json.loads(truth)
    if isinstance(res, str):
        res = json.loads(res)
    
    # extract the hardware, sku and workload of the benchmark_resutl into a list of items
    def extract_hsw(json_obj):
        extracted_data = []
        for hardware in json_obj:
            if isinstance(json_obj[hardware], list):
                # Iterate over each item in the hardware's list
                for item in json_obj[hardware]:
                    # Extract the workload key and values
                    hardware_key = item['hardware']
                    sku_key = item['sku']
                    workload_key = item['workload']
                    # Append the extracted data to the list
                    extracted_data.append((hardware_key, sku_key, workload_key))
        return extracted_data
    
    key_truth = list(truth.keys())
    key_res = list(res.keys())
    key_truth = [key.lower() for key in key_truth]
    key_res = [key.lower() for key in key_res]
    key_truth = [key.replace(" gpu", "") for key in key_truth]
    key_res = [key.replace(" gpu", "") for key in key_res]
    match_num_key = (key_truth == key_res)

    # all workload in truth exists in res, or both truth and res are empty, true positive
    extracted_hsw_truth = extract_hsw(truth)
    extracted_hsw_res = extract_hsw(res)
    match_recall = False
    if len(extracted_hsw_truth) == 0:
        if len(extracted_hsw_res) == 0:
            match_recall = True
    elif (all(item in extracted_hsw_res for item in extracted_hsw_truth) ):
        match_recall = True

    # lens not match, false positive or true negative
    match_accuracy = len(extracted_hsw_truth) == len(extracted_hsw_res)
    
    match_recall = match_num_key and match_recall
    match_accuracy = match_accuracy and match_recall
    return match_recall, match_accuracy

def is_correct_markdown_or_no_table(s):  
    # Check if any markdown table exists  
    pattern_table = r"\|(.+\|)+\r?\n\|(:?[-]+:?\|)+\r?\n(\|.+)+"  
    if re.search(pattern_table, s, re.MULTILINE):  
        # If table exists, check if it is correctly formatted  
        pattern_correct = r"(^|\n\n)\|(.+\|)+\r?\n\|(:?[-]+:?\|)+\r?\n((\|.+)+)(\n\n|$)"  
        return bool(re.search(pattern_correct, s, re.MULTILINE))  
    else:  
        # If no table exists, return true  
        return True

def test_match():
    truth = 'fp8 gemm'
    res = ['test gemm fp8', 'test GEMM fp8', 'GEMM', 'test GEMM']
    match = []
    for r in res:
        m = match_must_include(truth, r)
        match.append(m)
        print(f' res is: \t{r}\n truth is: \t{truth}\n match is: \t{m}\n\n')
        
def count_keys(json_obj, key):
    count = 0
    if type(json_obj) == str:
        return 0
    if key in json_obj:
        count += 1
    for k, v in json_obj.items():
        if isinstance(v, dict):
            count += count_keys(v, key)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    count += count_keys(item, key)
    return count

def evaluateion_dcw_output(gt_list, logs_list):
    num_correct = {
        'qtype': 0,
        'target': 0,
        'baseline': 0,
        'criterion': 0,
        'workload': 0,
        'dcw': 0,
        'benchmark_result_recall': 0,
        'benchmark_result_accuracy': 0,
        'summary_table': 0,
    }

    incorrected_ids = {
        'qtype': [],
        'target': [],
        'baseline': [],
        'criterion': [],
        'workload': [],
        'dcw': [],
        'benchmark_result_recall': [],
        'benchmark_result_accuracy': [],
        'summary_table': [],
    }
    
    def perform_match_on_one_case():
        index = gt["question_id"]
        if gt["question_id"] == log["question_id"]:            
            # do match
            #print(f"processing index {index}")
            match_type = match_must_include(gt["ground_truth_0"]["type"], log["output_0"]["type"])
            match_target = match_must_include(gt["ground_truth_1"]["dcw"]["Design"]["Target"], log["output_1"]["dcw"]["Design"]["Target"])
            match_baseline = match_must_include(gt["ground_truth_1"]["dcw"]["Design"]["Baseline"], log["output_1"]["dcw"]["Design"]["Baseline"])
            match_criterion = match_must_include(gt["ground_truth_1"]["dcw"]["Criterion"], log["output_1"]["dcw"]["Criterion"].replace('10','ten').replace('5','five').replace('1','one'))
            match_workload = match_must_include(gt["ground_truth_1"]["dcw"]["Workload"], log["output_1"]["dcw"]["Workload"].replace('all-to-all','alltoall'))            
            match_benchmark_res_recall, match_benchmark_res_accuracy = match_must_include_result(gt["ground_truth_2"]["benchmark_res"], log["output_2"]["benchmark_res"])


            is_summary_table_correct = is_correct_markdown_or_no_table(log["output_3"]["summary"])
            
            # evaluate question classification
            if match_type:
                num_correct['qtype'] += 1
            else:
                incorrected_ids['qtype'].append(log["question_id"])
            
            # evaluate dcw match result against ground truth
            if match_target:
                num_correct['target'] += 1
            else:
                incorrected_ids['target'].append(log["question_id"])
            if match_baseline:
                num_correct['baseline'] += 1
            else:
                incorrected_ids['baseline'].append(log["question_id"])
            if match_criterion:
                num_correct['criterion'] += 1
            else:
                incorrected_ids['criterion'].append(log["question_id"])
            if match_workload:
                num_correct['workload'] += 1
            else:
                incorrected_ids['workload'].append(log["question_id"])
                
            if (match_target & match_baseline & match_criterion & match_workload):
                num_correct['dcw'] += 1
            
            # evaluate benchmark result match result against ground truth
            if match_benchmark_res_recall:
                num_correct['benchmark_result_recall'] += 1
            else:
                incorrected_ids['benchmark_result_recall'].append(log["question_id"])
                
            # check markdown table format
            if is_summary_table_correct:
                num_correct['summary_table'] += 1
            else:
                incorrected_ids['summary_table'].append(log["question_id"])
                
            # check benchmark retrieval accuracy
            if match_benchmark_res_accuracy:
                num_correct['benchmark_result_accuracy'] += 1
            else:
                incorrected_ids['benchmark_result_accuracy'].append(log["question_id"])
            
                
        return index

    total_cases = len(gt_list)
    if (len(logs_list) == total_cases):
        print("Performing evaluation on all questions")
        for gt, log in zip(gt_list, logs_list):
            perform_match_on_one_case()        
    else:        
        for log in logs_list:
            for gt in gt_list:
                perform_match_on_one_case() 
                if gt["question_id"] == log["question_id"]:
                    index = log["question_id"]
                    print(f"Performing evaluation on selected questions {index}")
    
    return num_correct, incorrected_ids