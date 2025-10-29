from datetime import datetime
import time
import os
import json

def is_json(value):
    try:
        json_object = json.loads(value)
    except ValueError as e:
        return False
    return True

def load_benchmark_res(value):
    json_objects = value
    parsed_objects = []
    if json_objects:
        for obj in json_objects:
            if is_json(obj):
                parsed_objects.append(json.loads(obj))
        return parsed_objects
    else:
        return json_objects

def generate_debug_log(agent, index, user_prompt, stop_func=3):

    output_dict = {
        "question_id": str(index),
        "input_question": user_prompt,
        "output_0": {"function_name": "question_classification"},
        "output_1": {"function_name": "gen_dcw"},
        "output_2": {"function_name": "retrieve_benchmark_result","benchmark_res": {}},
        "output_3": {"function_name": "summarize_benchmark_result"},
        "output_4": {"function_name": "analyze_benchmark_result"},
    }
    
    print(f"\n\n[DEBUG] question_id is {index}")
    print(f">>>> Debug: question is: \n {user_prompt}")
    #(summary, benchmark_res, full_dcw, x1, guid, x2) = agent.benchmarking_session(user_prompt)
    
    start_time0 = time.time()
    question_type = agent.question_classification(user_prompt)
    duration0 = time.time() - start_time0
    output_dict["output_0"]["type"] = question_type
    output_dict["output_0"]["duration"] = duration0
    print(f">>>> Debug: question type is: \n {question_type}")
    
    if question_type.count("1") > 0: # evaluation
    
        start_time1 = time.time()
        full_dcw = agent.gen_dcw(user_prompt)
        #full_dcw = agent.dcw_parser.parse(full_dcw)
        dcw = full_dcw.dict()
        duration1 = time.time() - start_time1
        output_dict["output_1"]["dcw"] = dcw
        output_dict["output_1"]["duration"] = duration1
        print(f">>>> Debug: dcw is: \n {full_dcw}")
        
        if stop_func > 1:    
            start_time2 = time.time()
            benchmark_res = agent.retrieve_benchmark_result(full_dcw)
            duration2 = time.time() - start_time2    
            for key, value in benchmark_res.items():
                json_objects = load_benchmark_res(value)           
                output_dict["output_2"]["benchmark_res"][key] = json_objects
            output_dict["output_2"]["duration"] = duration2
            print(f">>>> Debug: benchmark_res is: \n {benchmark_res}")

        if stop_func > 2:
            start_time3 = time.time()
            summary = agent.summarize_benchmark_result(user_prompt, full_dcw, benchmark_res)
            duration3 = time.time() - start_time3
            output_dict["output_3"]["summary"] = summary
            output_dict["output_3"]["duration"] = duration3
            print(f">>>> Debug: summary is: \n {summary}")
    
    elif question_type.count("2") > 0: # analysis
        start_time4 = time.time()
        analysis_message = agent.analyze_session(summary)
        duration4 = time.time() - start_time4
        output_dict["output_4"]["duration"] = duration4
        print(f">>>> Debug: analysis response is: \n {analysis_message}")
         
        
    return output_dict

def save_debug_log(output_dict, folder_path, index):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
            
    # Save the JSON output to a separate file for each question    
    with open(f"{folder_path}/samples_output_{index:03d}.json", "w") as output_file:
        json.dump(output_dict, output_file, indent=4)
        
def generate_log_path(user_prompt_index, user_prompt_lines):
    # Get the current date and time and set the output folder nmae
    now = datetime.now()
    folder_name = now.strftime("try_%Y_%m_%d_%H_%M_%S")

    # change folder name to debug if user_prompt_index is none
    if user_prompt_index is not None:
        user_prompt_lines_debug = [user_prompt_lines[i] for i in user_prompt_index]
        folder_name = "debug"
    else:
        user_prompt_lines_debug = user_prompt_lines

    # Create the folder if it does not exist
    folder_path = f"{folder_name}"
    
    return folder_path, user_prompt_lines_debug

def load_debug_log(file_path = ""):
    files = sorted(os.listdir(file_path))
    json_files = [file for file in files if file.endswith(".json")]
    logs_list = []
    for json_file in json_files:
        file_name = os.path.join(file_path, json_file)
        with open(file_name, "r") as file:
            log_json = json.load(file)
            logs_list.append(log_json)            
    return logs_list