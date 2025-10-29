# ground truth

import json
import os
from util_question import get_user_question

def create_initial_gt(infradio, question_md_file, save_path):

    user_prompt_lines = get_user_question("/home/lequ/InfraWise/sample_questions/samples_infrawise.md")

    # Create the folder if it does not exist
    folder_path = f"/home/lequ/InfraWise/sample_questions/ground_truth_xxx"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for (i, user_prompt) in enumerate(user_prompt_lines):
        
        index = i
        
        output_dict = {
            "question_id": str(index),
            "input_question": user_prompt,
            "ground_truth_1": {"function_name": "gen_dcw"},
            "ground_truth_2": {"function_name": "retrieve_benchmark_result","benchmark_res": {}},
            "ground_truth_3": {"function_name": "summarize_benchmark_result"},
        }
        
        print(f"\n\n[DEBUG] question_id is {index}")

        full_dcw = infradio.gen_dcw(user_prompt)
        full_dcw = infradio.dcw_parser.parse(full_dcw)
        dcw = full_dcw.dict()
        output_dict["ground_truth_1"]["dcw"] = dcw  
        
        # Convert the output list to JSON format
        json_output = json.dumps(output_dict, indent=4)
        
        # Save the JSON output to a separate file for each question
        
        with open(f"{folder_path}/question_{index:03d}.json", "w") as output_file:
            json.dump(output_dict, output_file, indent=4)


def append_benchmark_res_gt(init_gt_path, benchmark_res_path, save_path):
    
    def read_json_files(path):
        files = sorted(os.listdir(path))
        json_files = [file for file in files if file.endswith(".json")]
        json_list = []
        for json_file in json_files:
            file_name = os.path.join(path, json_file)
            with open(file_name, "r") as file:
                json_obj = json.load(file)
                json_list.append(json_obj)
        return json_list
        
    
    # Read init ground truth, which consists the input question, the dcw ground truth
    init_gt_list = read_json_files(init_gt_path)
    print(f"number of init gt files: {len(init_gt_list)}")            
            
    # Read generated benchmark_result files
    benchmark_res_list = read_json_files(benchmark_res_path)
    print(f"number of benchmark result files: {len(init_gt_list)}")
    
    # append
    modified_gt_list = init_gt_list.copy()
    for modified_gt, benchmark_res in zip(modified_gt_list, benchmark_res_list):
        if (modified_gt["question_id"] == benchmark_res["question_id"]) & (modified_gt["input_question"] == benchmark_res["input_question"]):
            id = modified_gt["question_id"]
            print(f"question id {id} match, processing")
            modified_gt["ground_truth_2"]["benchmark_res"] = benchmark_res["output_2"]["benchmark_res"]
    
    # Save    
    if save_path == init_gt_path:
        print("ending!!!")
        return 'save no'
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for save_dict in modified_gt_list:
            save_id = int(save_dict["question_id"])
            with open(f"{save_path}/ground_truth_{save_id:03d}.json", "w") as output_file:
                json.dump(save_dict, output_file, indent=4)
        return 'save yes'
    