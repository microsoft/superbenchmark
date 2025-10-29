[task]
Your task is to learn from user's input, compare it to available arguments, and output the argument and its desired value based on user's intension, using below format:
```json
{
    "name": "argument name",
    "value": "2"
}
```
Example 1:
user input = build pdf
output should be
```json
{
    "name": "pdf_run",
    "value": "True"
}
```

Example 2:
user input = build id 16
output should be
```json
{
    "name": "case_id",
    "value": "11"
}
```

[available arguments]
argument name: case_id, type=int, help='Index of existing builds, integer'
argument name: case_folder, type=str, help='Please specify the new report case folder path'
argument name: bypass_gpt', action='store_true', help='Bool value, set to True will bypass the GPT sessions'
argument name: build_run', action='store_true', help='Bool value, set to True will trigger a full report generation'
argument name: pdf_run', action='store_true', help='Bool value, set to True will only rebuild the pdf from prompts'
argument name: modify_content, type=str, choices={workload}, help='Please specify which section content you would like to modify'


[knowledge]
'es' in argument modify_content possible value, means executive summary