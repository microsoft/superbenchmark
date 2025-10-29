import os
import glob

def translate(report_class, content, reserved, language):
    if language == 'english':
        return content
    translated = translate_with_llm(report_class, content, reserved, language)
    return translated

def translate_tex_file(report_class, file_path, reserved, language):
    print(f"Translating file: {file_path} to {language}")
    with open(file_path, 'r') as file:
        filedata = file.read()
    translated = translate(report_class, filedata, reserved, language)
    with open(file_path, 'w') as file:
        file.write(translated)

def translate_build_ins(report_class, report_file_path, reserved, language):
    # list latex files
    tex_files = glob.glob(os.path.join(report_file_path, '*.tex'))
    tex_files.sort()  # Sort files by name
    for tex_file in tex_files:
        translate_tex_file(report_class, tex_file, reserved, language)

def translate_with_llm(report_class, content, reserved, language):
    sys_prompt = f"Your task is to translate the following text to {language.capitalize()}.\n\nYou must keep the reserved words {reserved} unchanged.\n\nYou must use the latex syntax format to output, and maintain the original latex structure.\n\n"
    user_prompt = f"Here is the text:\n{content}"
    print(f"translating to {language}...")
    translated = report_class.call_openai_api(sys_prompt, user_prompt)
    return translated
