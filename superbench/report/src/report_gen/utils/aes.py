from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
import os
import re
import pandas as pd
import numpy as np
import time

pd.set_option('future.no_silent_downcasting', True)

# temp, not used
def extract(sentence = 'I love to play football'):
    n = 2
    sixgrams = ngrams(sentence.split(), n)

    for grams in sixgrams:
        print(grams)

# temp, not used
def extract_n_gram_ml(sentence = 'I love to play football'):
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(list(sentence))
    print(vectorizer.get_feature_names())
    print(X.toarray())

# temp, todo
# score-specific word embeddings SSWEs

# temp, read latex file
def read_tex_files(folder_path: str):
    ss = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".tex"):
            with open(os.path.join(folder_path, filename), 'r') as file:
                s = file.read()
                ss[filename] = s  
    return ss

def save_tex_file(folder, filename, content):
    # Check if the file exists in the folder
    if os.path.isfile(os.path.join(folder, filename)):
        print(f'file {filename} in folder {folder}')
        # If it exists, open the file in write mode
        with open(os.path.join(folder, filename), 'w') as file:
            # Write the content to the file
            file.write(content)
    else:
        print(f'file {filename} not in folder {folder}')

def extract_tables_latex(content: str):
    pattern = r"\\begin\{table\}(.*?)\\end\{table\}"
    tables = re.findall(pattern, content, re.DOTALL)
    return tables

def extract_table_detail(table):
    caption = re.search(r'\\caption\{(.*?)\}', table)
    label = re.search(r'\\label\{(.*?)\}', table)
    header = re.search(r'\\toprule(.*?)\\midrule', table, re.DOTALL)
    content = re.search(r'\\midrule(.*?)\\bottomrule', table, re.DOTALL)

    if caption:
        caption = caption.group(1)
    if label:
        label = label.group(1)
    if header:
        header = header.group(1).strip().split('\n')
    if content:
        content = content.group(1).strip().split('\n')

    return [caption, label, header, content]

def extract_number(s):
    match = re.search(r'\s(\d+(\.\d+)?)[\sXx]', s)
    return float(match.group(1)) if match else None

# validate table
def check_table_aggregation_number(report_path):
    output_text = ''
    print(f"checking tables...")
    section_dict = read_tex_files(report_path)
    
    # iterate sections
    for section, sec_content in section_dict.items():
        output_text += f">>>> checking tables for [{section}]\n"
        
        # extract all tables in each section
        tables = extract_tables_latex(sec_content)
        
        # parse each table 
        for table in tables:
            [caption, label, header, table_content] = extract_table_detail(table)
            output_text += f"[info]\ntable\ncaption = {caption}\nlabel = {label}\nheader = {header}\ncontent = {table_content}\n"
            rows = []
            #print(label)
            for row in table_content:
                cells = row.replace("\\", "").split('&')
                new_cells= []
                for cell in cells:
                    new_cell = extract_number(cell)
                    new_cells.append(new_cell)
                rows.append(new_cells)
            # for each table, calculate the numbers
            # Create a DataFrame
            df = pd.DataFrame(rows)

            # Replace None with NaN for calculations
            df = df.replace({None: np.nan})
            # Ensure all columns are numeric for aggregation
            df = df.apply(pd.to_numeric, errors='coerce')

            # Calculate mean, min, max
            mean_val = round(df.mean(), 2)
            min_val = round(df.min(), 2)
            max_val = round(df.max(), 2)

            mean_str = "\t\t".join(map(str, mean_val))
            min_str = "\t\t".join(map(str, min_val))
            max_str = "\t\t".join(map(str, max_val))
            output_text += f"[debug] mean values are:\t {mean_str}\n"
            output_text += f"[debug] min  values are:\t {min_str}\n"
            output_text += f"[debug] max  values are:\t {max_str}\n"
    
    return output_text

# count the number of incorrect latex package usage
def count_occurrences(input_string):
    patterns = [r'\\textbf \\textcolor blue\{',
                r'\\paragraph\{\}',
                r'\{\{.*\}\}',
                r'\[(O|o)utput\]']
    out_string = ''
    
    section_dict = read_tex_files(input_string)
    
    print(f"checking latex patterns...")
    
    # iterate sections
    for section, sec_content in section_dict.items():
        out_string += f">>>> checking incorrect latex patterns for [{section}]\n"
        out_string += f"[info]\n"
    
        for pattern in patterns:
            matches = re.findall(pattern, sec_content)
            if len(matches) == 0: # no incorrect match
                error_msg = ''
            elif len(matches) > 0: # incorrect match(es) found
                error_msg = '[error]'
            out_string += f"[debug] match pattern {pattern} count: {error_msg} {len(matches)} \n"
  
    return out_string

# count the number of incorrect references
def count_invalid_refs(input_string):
    pattern = r'\\ref\{(tab:|fig:|sec:|eq:)?[^}]*\}'
    out_string = ''
    
    section_dict = read_tex_files(input_string)
    
    print(f"checking incorrect references...")
    
    # iterate sections
    for section, sec_content in section_dict.items():
        out_string += f">>>> checking incorrect references for [{section}]\n"
        out_string += f"[info]\n"
    
        matches = re.findall(pattern, sec_content)
        invalid_refs = [match for match in matches if not match.startswith(('tab:', 'fig:', 'sec:', 'eq:'))]
        if len(invalid_refs) == 0: # no incorrect match
            error_msg = ''
        elif len(invalid_refs) > 0: # incorrect match(es) found
            error_msg = '[error]'
        out_string += f"[debug] incorrect reference count: {error_msg} {len(invalid_refs)} \n"
   
    return out_string

# check for grammer error or typo
def check_grammer_error_typo(report_class, input_string):
    sys_prompt = f"Your task is to check for any grammer error or typo. If no grammer error or typo exists, output 'you are fine', otherwise output the grammer error or the typo with an indicator header [error]: \n\n"
    
    section_dict = read_tex_files(input_string)
    
    print(f"checking grammer and typo...")
    
    out_string = ''
    # iterate sections
    for section, sec_content in section_dict.items():
        if section != 'table.tex':
            out_string += f">>>> checking grammer and typo for [{section}]\n"
            out_string += f"[info]\n"
            # trigger a GPT session to perform the check
            print(f"check section {section}")
            error_msg = report_class.call_openai_api(sys_prompt, sec_content)
            out_string += f"[debug] grammer and typo checking result: {error_msg} \n"
   
    return out_string

# remove head and tail if any
def remove_header_tail_folder(tex_folder):

    section_dict = read_tex_files(tex_folder)
    
    print(f"removing unecessary headers and tails from .tex files...")
    out_string = ''
    
    # iterate sections
    for section, sec_content in section_dict.items():
        out_string += f">>>> removing unecessary headers and tails for [{section}]\n"
        out_string += f"[info]\n"
        if '```latex' in sec_content and '```' in sec_content:
            new_sec_content = sec_content.replace('```latex', '').replace('```', '')
            save_tex_file(tex_folder, section, new_sec_content)
            out_string += f"[debug] removed headers and tails for {section}\n"
            print(f"{section} header/tails are removed")
            #if 'summary' in section:
                #print(f"{section} old content is\n{sec_content}")
                #print(f"{section} new content is\n{new_sec_content}")
        else:
            out_string = out_string
    
    time.sleep(10)
    return out_string

def remove_lines_starting_with(text, prefixes):
    lines = text.splitlines()
    filtered_lines = [line for line in lines if not any(line.startswith(prefix) for prefix in prefixes)]
    return "\n".join(filtered_lines)