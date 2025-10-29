import os
import re
import json

def get_prompt_from(path:str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    with open(path, "r") as f:
        return f.read()


def extract_all_json(benchmark, nested=False):
    # two level json object
    if nested:
        pattern = r'(\{.*?\{?.*?\}?.*?\})'
        if '```json' in benchmark:
            pattern = r'```json\s*\n*(\{.*?\{?.*?\}?.*?\})\n*```'
        elif '```jsonl' in benchmark:     
            pattern = r'```jsonl\s*\n*(\{.*?\{?.*?\}?.*?\})\n*```'
    # one level json object
    else:
        pattern = r'(\{.*?\})'
        if '```json' in benchmark:
            pattern = r'```json\s*\n*(\{.*?\})\n*```'
        elif '```jsonl' in benchmark:     
            pattern = r'```jsonl\s*\n*(\{.*?\})\n*```'    
    

    json_match = re.search(pattern, benchmark, re.DOTALL)

    if json_match:
        json_config = re.findall(pattern, benchmark, re.DOTALL)
        return json_config
    else:
        return None
    
def extract_all_json_fake(benchmark, nested=False):
    # two level json object
    if nested:
        pattern = r'(\{.*?\{.*?\}.*?\})'
        if '```json' in benchmark:
            pattern = r'```json\s*\n*(\{.*?\{.*?\}.*?\})\n*```'
        elif '```jsonl' in benchmark:     
            pattern = r'```jsonl\s*\n*(\{.*?\{.*?\}.*?\})\n*```'
    # one level json object
    else:
        pattern = r'(\{.*?\})'
        if '```json' in benchmark:
            pattern = r'```json\s*\n*(\{.*?\})\n*```'
        elif '```jsonl' in benchmark:     
            pattern = r'```jsonl\s*\n*(\{.*?\})\n*```'    
    

    json_match = re.search(pattern, benchmark, re.DOTALL)

    if json_match:
        json_config = re.findall(pattern, benchmark, re.DOTALL)
        return json_config
    else:
        return None
    
def extract_yaml(benchmark):          
    if '```yaml' in benchmark:     
        pattern = r'```(?:yaml)\s*\n*(.*?)\n*```'
        json_match = re.search(pattern, benchmark, re.DOTALL)

        if json_match:
            json_config = re.findall(pattern, benchmark, re.DOTALL)
            return json_config[-1]
        else:
            return None
    else:
        return benchmark

def extract_json(benchmark, nested=True):               
    jsons = extract_all_json(benchmark, nested)
    if jsons:
        return jsons[-1]
    return None

def is_valid_json(json_str):
    try:
        json.loads(json_str)
    except ValueError:
        return False
    return True

retry_function = lambda func, max_retries, *args, **kwargs: (  
    func(*args, **kwargs) if max_retries == 0 else (  
        func(*args, **kwargs) or retry_function(func, max_retries - 1, *args, **kwargs)  
    )  
) if max_retries > 0 else None  

async def retry_async(func, max_retries, *args, **kwargs):
    if max_retries == 0:
        return await func(*args, **kwargs)

    result = await func(*args, **kwargs)
    if result is not None:
        return result

    return await retry_async(func, max_retries - 1, *args, **kwargs)

def extract_metrics(raw_data):
    metric_data = {}
    for line in raw_data.split("\n"):
        match = re.search(r"\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\d+\.\d+)\s*\|", line)
        if match:
            metric, stat, value = match.groups()
            if 'mem_max_bandwidth' not in metric:
                metric_data[metric] = float(value)
    return metric_data
