import pandas as pd
import json
import os

def append_tables(json_objects):
    # Create a dictionary to store the combined tables
    combined = {}

    # Iterate over each json object
    for obj in json_objects:
        # Get the section name
        section = obj['section']

        # If this section is not in the combined dictionary, add it
        if section not in combined:
            combined[section] = obj
        else:
            # If this section is already in the combined dictionary, append the tables
            combined[section]['tables'].append(obj['tables'][0])

    # Convert the combined dictionary back into a list of json objects
    combined_objects = list(combined.values())

    return combined_objects

def excel_to_json(path, file_name):
    # Load spreadsheet
    excel_file = os.path.join(path, file_name)
    xl = pd.ExcelFile(excel_file)

    # Load a sheet into a DataFrame by its name
    sheet_names = xl.sheet_names
    
    final_json = []
    
    for sheet in sheet_names:
        df = xl.parse(sheet)
        
        # Replace nan or NaN with "NA"
        df = df.fillna("NA")

        # Get title and label
        save_to_appendix = df.columns[0]
        section = df.iloc[0, 0]
        index = df.iloc[1, 0]
        title = df.iloc[2, 0]

        # Drop title and label rows
        df = df.iloc[3:]

        # Set new header
        df.columns = df.iloc[0]
        df = df[1:]

        # Convert DataFrame to JSON
        json_data = df.to_dict('records')

        # Create final markdown object
        final_markdown = {
            "title": title,
            "label": section + '-' + str(index),
            "data": json_data
        }
        
        # Create table JSON object
        table = {
            "index": str(index),
            "save to appendix": str(save_to_appendix),
            "label": f'{str(section)}-{str(index)}',
            "title_prefix": title,
            "content": json_data
        }
        
        # for latex
        final_json.append({
            "section": section,
            "tables": [table]
        })

        # Write JSON data into a file
        if index == 1:            
            with open(os.path.join(path, f'md', f'{section}.md'), 'w') as markdown_file:
                markdown_file.write(f"title: {final_markdown['title']}\n")
                markdown_file.write(f"label: {final_markdown['label']}\n")
                json.dump(final_markdown['data'], markdown_file, indent=4)
        else:
            with open(os.path.join(path, f'md', f'{section}.md'), 'a') as markdown_file:
                markdown_file.write(f"\n\n")
                markdown_file.write(f"title: {final_markdown['title']}\n")
                markdown_file.write(f"label: {final_markdown['label']}\n")
                json.dump(final_markdown['data'], markdown_file, indent=4)
                
    # for latex
    combined_objects = append_tables(final_json)
    for combined_object in combined_objects:
        section = combined_object['section']
        with open(os.path.join(path, f'json', f'{section}.json'), 'w') as json_file:
            json.dump(combined_object, json_file, indent=4)
    