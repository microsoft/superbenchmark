import os
import json

def get_user_question(file_name = ""):
    user_prompt_lines = []

    with open(file_name, "r") as file:
        for line in file:
            # Remove leading and trailing whitespaces (including newlines)
            stripped_line = line.strip()

            # Check if the line is not empty and does not start with '#'
            if stripped_line and not stripped_line.startswith("#"):
                user_prompt_lines.append(stripped_line)
    print(len(user_prompt_lines))
    return user_prompt_lines

def get_user_question_json(file_path = ""):
    user_prompt_lines = []
    user_prompt_jsons = []
    files = sorted(os.listdir(file_path))
    json_files = [file for file in files if file.endswith(".json")]
    for json_file in json_files:
        file_name = os.path.join(file_path, json_file)
        with open(file_name, "r") as file:
            user_prompt_json = json.load(file)
            user_prompt = user_prompt_json["input_question"]
            user_prompt_lines.append(user_prompt)
            user_prompt_jsons.append(user_prompt_json)
            
    print(len(user_prompt_lines))
    print(len(user_prompt_jsons))
    return user_prompt_lines, user_prompt_jsons