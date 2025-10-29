import sys
import json
import os
import json
import copy
import shutil
import glob
import time
import threading
import multiprocessing
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import SingleQuotedScalarString as sq
from datetime import datetime
from colorama import Fore, Style
import ast

from .utils import *
from .config import PROMPT_DIR, BLOB_LOCAL_DIR, CASE_DIR, BUILD_DIR, BUILD_TEMP_DIR

# ENV
# REPORTGEN_TIMEOUT, int, default 180
# REPORTGEN_SKIP_AZURE_SAVE, bool, default True

class Report :

    def __init__(self,
                model: LLMSession = LLMSession(),
                verbose: bool=False,
                debug: bool=False) -> None:
        self.model = model
        self.verbose = verbose
        self.build_retry_cnt = 0
        self._debug = debug
        translate_language = os.environ.get('TRANSLATE_LANGUAGE')
        if translate_language and translate_language.lower() == 'chinese':
            self.translate_language = 'chinese'
        else:
            self.translate_language = 'english'

        # help message
        help_doc_path = os.path.join(PROMPT_DIR, 'infrawise_help.json')
        with open(help_doc_path) as help_file:
            help_msg = json.load(help_file)
        self.help_msg = help_msg
        
    
    def call_openai_api(self, system_prompt, user_prompt):
        if self.model is not None:
            return self.model.chat(system_prompt, user_prompt)
        else:
            raise ValueError("No model provided to Report instance.")
    
    # Generate report 
    def report_gen_section(self, dcw: DCW, section_id: int, section: str, subsection: str = '', benchmark_result: dict = {}, file_format = 'markdown', bypass=False) -> str:
        
        def get_task_prompt(section, subsection): 
            prompt = get_prompt_from(os.path.join(PROMPT_DIR, f"report", f'report_task.md'))
            if subsection == '':
                prompt = prompt.format(title=section,format=file_format)
            else:
                prompt = prompt.format(title=subsection,format=file_format)
            return prompt
        
        def get_knowledge_prompt(dcw, section, subsection):            
            if subsection == '':
                prompt_file_name = f'report_gen_{section}.md'
            else:
                prompt_file_name = f'report_gen_{section}_{subsection}.md'     
            prompt = get_prompt_from(os.path.join(PROMPT_DIR, f"report", prompt_file_name))
            prompt = prompt.replace("{target}", dcw.Design.Target).replace("{baseline}", dcw.Design.Baseline)
            return prompt
        
        # [task]
        task_prompt = get_task_prompt(section, subsection)
        # [tool]
        latex_prompt = get_prompt_from(os.path.join(PROMPT_DIR, f"report", f'report_tool_{file_format}.md'))
        tool_prompt = latex_prompt
        # [knowledge, content of output]
        knowledge_prompt = get_knowledge_prompt(dcw, section, subsection)
        # [case]
        case_prompt = get_prompt_from(os.path.join(CASE_DIR, f'case_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}', f'{section_id}_{section}.md'))
        case_prompt = case_prompt.replace("{target}", dcw.Design.Target).replace("{baseline}", dcw.Design.Baseline)

        # system prompt
        sys_prompt = f'{task_prompt}\n\n{tool_prompt}\n\n{knowledge_prompt}'

        # user prompt
        user_prompt = f'{case_prompt}'
        
        if bypass:
            resp = 'test abc'
        else:
            resp = self.call_openai_api(sys_prompt, user_prompt)
        return resp
    
    def report_gen_summary_conclusion(self, dcw: DCW, section_id: int, section_title: str, section_list: list, file_format = 'markdown', bypass=False) -> str:
        prompt_file_name = f'{section_id}_{section_title}.md'
        # [case]
        case_prompt = get_prompt_from(os.path.join(CASE_DIR, f'case_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}', prompt_file_name))
        # [tool]
        latex_prompt = get_prompt_from(os.path.join(PROMPT_DIR, f"report", f'report_tool_{file_format}.md'))
        tool_prompt = latex_prompt
        # [system prompt]
        sys_prompt = case_prompt + tool_prompt
        sys_prompt = sys_prompt.replace("{target}", dcw.Design.Target).replace("{baseline}", dcw.Design.Baseline).replace("{format}", file_format)
        user_prompt = f"[case] is\n{str(section_list)}\n\n"
        
        if bypass:
            resp = 'test fff'
        else:
            resp = self.call_openai_api(sys_prompt, user_prompt)
        return resp

    def save_tex_file(self, title, content, report_file_path, file_format='markdown'):
        if file_format == 'latex':
            file_ext = 'tex'
        else:
            file_ext = 'md'
        with open(f"{report_file_path}/{title}.{file_ext}", "w") as file:
            file.write(content)

    def fill_chapter_content(self, sec_category):
        chapter = f'\\chapter{{}}\n\\clearpage'
        if isinstance(sec_category, str):
            splited = sec_category.split(':')
            if len(splited) == 2:
                title = splited[1]
                chapter = f'\\chapter{{{title}}}\n\\clearpage'
        return chapter

    def section_worker(self, build, dcw, section, report_file_path, file_format, bypass):
        section_id = section['section_id']
        section_shortname = section['sec_shortname']
        section['comment'] = f"generating {section_shortname}"  
        if section['sec_category'] == 'benchmark':
            logger.warning(f">> [internal info]: Build {build}, generating report: section content: {section_shortname}\n")
            section['content'] = self.report_gen_section(dcw, section_id, section_shortname, '', {}, file_format, bypass)
            self.save_tex_file(f'{section_id}_{section_shortname}', section['content'], report_file_path, file_format)
            logger.info(f">> [internal info]: Build: {build}, report_gen_section [cat:benchmark]: {section_id}_{section_shortname} finished\n")
        elif (section['sec_category'] == 'appendix') or (section['sec_category'] == 'manual'):
            logger.warning(f">> [internal info]: Build {build}, generating report: appendix content: {section_shortname}\n")
            section['content'] = self.report_gen_summary_conclusion(dcw, section_id, section_shortname, [], file_format, bypass)
            self.save_tex_file(f'{section_id}_{section_shortname}', section['content'], report_file_path, file_format)
            logger.info(f">> [internal info]: Build: {build}, report_gen_summary_conclusion [cat:appendix, manual]: {section_id}_{section_shortname} finished\n")
        elif (section['sec_category'] == 'chapter'):
            logger.warning(f">> [internal info]: Build {build}, generating report: chapter content: {section_shortname}\n")
            section['content'] = self.fill_chapter_content(section['sec_category'])
            self.save_tex_file(f'{section_id}_{section_shortname}', section['content'], report_file_path, file_format)
            logger.info(f">> [internal info]: Build: {build}, report_gen_summary_conclusion [cat:chapter]: {section_id}_{section_shortname} finished\n")

    def fill_content_benchmark_appendix(self, build, dcw, report, report_file_path, file_format, bypass=False):
        threads = []
        for section in report:
            t = threading.Thread(target = self.section_worker, args=(build, dcw, section, report_file_path, file_format, bypass))
            threads.append(t)
            t.start()

        # Wait for all threads to finish
        for t in threads:
            t.join()

    def summary_worker(self, build, dcw, section, sections_list, report_file_path, file_format, bypass):
        section_id = section['section_id']
        section_shortname = section['sec_shortname']
        section['comment'] = f"generating {section_shortname}"

        # for section es and conclusion sections
        if ((section['sec_category'] == 'es') | (section['sec_category'] == 'conclusion')):
            logger.warning(f">> [internal info]: Build: {build}, generating report: report summary: {section_shortname}\n")
            section['content'] = self.report_gen_summary_conclusion(dcw, section_id, section_shortname, sections_list, file_format, bypass)
            self.save_tex_file(f'{section_id}_{section_shortname}', section['content'], report_file_path, file_format)
            logger.info(f">> [internal info]: Build: {build}, report_gen_summary_conclusion [cat:es, conclusion]: {section_id}_{section_shortname} finished\n")

    def fill_content_summary_conclusion(self, build, dcw, report, report_file_path, file_format, bypass=False, use_multithreading=True):
        # assemble the contents
        sections_list=[]
        for section in report:
            if section['sec_category'] == 'benchmark' or section['sec_category'] == 'manual':
                sections_list.append(section['content'])
        logger.info(f"Performance section count: {len(sections_list)}\n")
        # generating contents
        if use_multithreading: # use multi-threading
            threads = []
            for section in report:
                t = threading.Thread(target = self.summary_worker, args=(build, dcw, section, sections_list, report_file_path, file_format, bypass))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()
        else: # disable multi-threading for sections with unbalanced finish time
            for section in report:
                self.summary_worker(build, dcw, section, sections_list, report_file_path, file_format, bypass)
            
    def combine_main_latex(self, dcw, final_report_file_path, report_class):
        if report_class == 'benchmark_report' or report_class == 'simulation_report':
            report_file_name = 'report'
        elif report_class == 'business_impact_report':
            report_file_name = 'business'
        else:
            logger.info('wrong report class')
        
        # load report template
        main_latex_str = get_prompt_from(os.path.join(PROMPT_DIR, f"report", f"report_template/{self.translate_language}/report.tex"))

        # load structure
        section_str = ''
        appendix_str = ''
        after_spacer = False
        latex_class = 'article'
        with open(f"{final_report_file_path}/zzz_log.json", 'r') as structure_file:
            latex_structure = json.load(structure_file)
        for i, part in enumerate(latex_structure):
            if part["sec_category"] in ["spacer"]:
                after_spacer = True                
            if part["sec_category"] in ["manual", "es", "benchmark", "conclusion", "appendix", "chapter"]:
                if after_spacer == False:
                    section_str += f"\\input{{{part['section_id']}_{part['sec_shortname']}.tex}} \n"
                else:
                    if i == len(latex_structure) - 1 or 'chapter' in part['sec_shortname']:
                        appendix_str += f"\\input{{{part['section_id']}_{part['sec_shortname']}.tex}} \n"
                    else:
                        appendix_str += f"\\input{{{part['section_id']}_{part['sec_shortname']}.tex}} \n\\clearpage \n"
            if part["sec_category"] in ["table"]:
                if after_spacer == False:
                    section_str += f"\\input{{{part['sec_shortname']}.tex}} \n"
                else:
                    if i == len(latex_structure) - 1:
                        appendix_str += f"\\input{{{part['sec_shortname']}.tex}} \n"
                    else:
                        appendix_str += f"\\input{{{part['sec_shortname']}.tex}} \n\\clearpage \n"
                # remove empty table section
                table_file_path = f"{final_report_file_path}/{part['sec_shortname']}.tex"
                # only append the table section if the length of content of table.tex is larger than 50
                if os.path.exists(table_file_path) and os.path.getsize(table_file_path) < 50:
                    appendix_str = appendix_str.replace(f"\\clearpage \n\\input{{{part['sec_shortname']}.tex}}", "")
            if part["sec_category"] in ["chapter"]:
                latex_class = 'report'

        # change latex class
        logger.info(f'latex class: {latex_class}')
        main_latex_str = main_latex_str.replace(f'<<<latexclass>>>', latex_class)
            
            
        # load title
        with open(os.path.join(CASE_DIR, f"case_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}", f"title.json"), 'r') as title_file:
            title_json = json.load(title_file)
        if isinstance(title_json, list):
            if len(title_json) == 2:
                title_str = title_json[0]["title"]
                subtitle_str = title_json[1]["subtitle"]
            elif len(title_json) == 1:
                title_str = title_json[0]["title"]
                subtitle_str = ''
            else:
                title_str = ''
                subtitle_str = ''

        # load author
        author_str = ''
        with open(os.path.join(CASE_DIR, f"case_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}", f"author.json"), 'r') as author_file:
            author_json = json.load(author_file)
        for j, author_group in enumerate(author_json):
            name_len = 0
            for i, author in enumerate(author_group["authors"]):
                name_len += len(author['name'])
                if author["footnote"] != '':
                    if (i == len(author_group["authors"]) - 1) or (name_len > 70):
                        author_str += f"{author['name']} $^{author['footnote']}$ \\vspace{{0.2cm}} \\\\ \n"
                    else:
                        author_str += f"{author['name']} $^{author['footnote']}$ \\hspace{{0.12cm}}\n"
                if author["footnote"] == '':
                    if (i == len(author_group["authors"]) - 1) or (name_len > 70):
                        author_str += f"{author['name']} \\vspace{{0.2cm}} \\\\ \n"
                        name_len = 0
                    else:
                        author_str += f"{author['name']} \\hspace{{0.12cm}} \n"
            if j == len(author_json) - 1:
                author_str += f"\\textbf{{\\normalsize {author_group['affiliation']}}} \\vspace{{0.4cm}} \\\\"
            else:
                author_str += f"\\textbf{{\\normalsize {author_group['affiliation']}}} \\vspace{{0.4cm}} \\\\ \n"

        with open(f"{final_report_file_path}/{dcw.Design.Target.lower()}_{report_file_name}.tex", 'w') as main_tex_file:
            main_latex_str = main_latex_str.replace("<<<title>>>", title_str)
            main_latex_str = main_latex_str.replace("<<<subtitle>>>", subtitle_str)
            main_latex_str = main_latex_str.replace("<<<author>>>", author_str)
            main_latex_str = main_latex_str.replace("<<<section>>>", section_str)
            main_latex_str = main_latex_str.replace("<<<appendix>>>", appendix_str)
            main_latex_str = change_build_date(main_latex_str)
            main_tex_file.write(main_latex_str)
            
        # images
        img_dir_source = os.path.join(CASE_DIR, f"case_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}", f"img")
        img_dir_dest   = os.path.join(final_report_file_path, 'img')
        logger.info(img_dir_source)
        logger.info(img_dir_dest)
        if not os.path.exists(img_dir_dest):
            shutil.copytree(img_dir_source, img_dir_dest)
        else:
            for file_name in os.listdir(img_dir_source):
                shutil.copy2(os.path.join(img_dir_source, file_name), img_dir_dest)
            
            
        # style
        sty_source = os.path.join(PROMPT_DIR, f"report", f"report_template/{self.translate_language}/techreport.sty")
        sty_dest   = final_report_file_path
        dst_file = os.path.join(sty_dest, os.path.basename(sty_source)) 
        logger.info(sty_source)
        logger.info(sty_dest)
        logger.info(copy)
        shutil.copy2(sty_source, sty_dest)
        # change the date
        with open(dst_file, 'r') as sty_file_read:
            sty_str = sty_file_read.read()
        sty_str = change_build_date(sty_str)
        with open(dst_file, 'w') as sty_file_write:
            sty_file_write.write(sty_str)
        
        # makefile
        make_source = os.path.join(PROMPT_DIR, f"report", f"report_template/{self.translate_language}/Makefile")
        make_dest   = final_report_file_path
        make_file = os.path.join(sty_dest, os.path.basename(make_source)) 
        logger.info(make_source)
        logger.info(make_dest)
        with open(make_source, 'r') as file_src:
            make_file_content = file_src.read()
        make_file_content = make_file_content.replace("{target}", dcw.Design.Target)
        with open(os.path.join(make_dest, 'Makefile'), 'w') as file_dest:
            file_dest.write(make_file_content)            
            
    def fill_content_table(self, dcw:DCW, SKUNickName:dict, section_id: int, section: str, report_file_path: str, save_to: str='latex') -> list:
        
        # switch raw json file
        #if 'multinode' in section:
        #    raw_json_file_name = f"result_summary_raw_multinode.json"
        #else:
        #    raw_json_file_name = f"result_summary_raw.json"
        raw_json_file_name = f"result_summary_raw_multinode.json"
        
        raw_json_path = os.path.join(CASE_DIR, f'case_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}/raw', raw_json_file_name)
        projected_value_path = os.path.join(CASE_DIR, f'case_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}/raw', f"projected.json")
        spec_value_path = os.path.join(CASE_DIR, f'case_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}/raw', f"spec.json")
        table_meta_file = os.path.join(PROMPT_DIR, f'report/table', f"{section}_table_meta.json")
        manual_table_file = os.path.join(CASE_DIR, f'case_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}', f"table_manual.md")
        if os.path.exists(table_meta_file):
            logger.info(f"processing tables for [{section}]")
            raw_table_str = get_prompt_from(raw_json_path)
            # load raw json table
            raw_table_json = json.loads(raw_table_str.replace("'", '"'))
            # replace key with DCW value
            raw_table = table_replace_key(dcw.Design.Target.lower(), dcw.Design.Baseline.lower(), raw_table_json)
            #logger.info(raw_table)
            
            # load projected_table
            if os.path.exists(projected_value_path):
                projected_table_str = get_prompt_from(projected_value_path)
                projected_table_json = json.loads(projected_table_str.replace("'", '"'))
            else:
                projected_table_json = None
            
            # load spec
            if os.path.exists(spec_value_path):
                spec_table_str = get_prompt_from(spec_value_path)
                spec_table_json = json.loads(spec_table_str.replace("'", '"'))
            else:
                spec_table_json = None
            
            # load table meta
            table_meta_str = get_prompt_from(table_meta_file)
            table_meta = json.loads(table_meta_str)
            #logger.info(table_meta)
            
            # generate a sub table in json format
            [comparison_table, fig_to_save] = gen_comparison_table(dcw, SKUNickName, table_meta, raw_table, projected_table_json, spec_table_json, save_to)
            #logger.info(comparison_table)
            
            # amend the aggregated metrics to the tables
            comparison_table_amended = amend_aggregated_metrics(table_meta, comparison_table)
            
            # save fiture
            if not fig_to_save == '':
                fig_folder = os.path.join(report_file_path, f"img")
                fig_file_name = f"{section}.png"
                if not os.path.exists(fig_folder):
                    os.makedirs(fig_folder)
                fig_to_save.savefig(os.path.join(fig_folder, fig_file_name))
                logger.info(f"figure for [{section_id}_{section}] saved to {fig_folder}")
            
            if save_to == 'case':
                # save json table to case field
                with open(os.path.join(CASE_DIR, f'case_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}', f"{section_id}_{section}.md"), 'a+') as file:
                    file.seek(0)  # move the cursor to the start of the file
                    file_content_existing = file.read()
                    if '[case]' in file_content_existing:
                        # delete '[case]' and all following text
                        file_content_existing = file_content_existing.split('[case]')[0]
                        file.seek(0)
                        file.truncate()
                        file.write(file_content_existing)

                    # write new save secgtion
                    logger.info(f">> save [{section_id}_{section}] table to prompt case part")
                    file.write("[case]")
                    file.write("\n\n")
                    tables = comparison_table_amended["tables"]
                    for table in tables:
                        if ("save to prompt" not in table) or (("save to prompt" in table) and (table["save to prompt"] != "False")):
                            # in case some tables are not aimed to be put into prompt, but need to appear in the latex, then set 'save to prompt' to False
                            title = table["title_prefix"]
                            label = table["label"]
                            content = table["content"]
                            # write tables
                            file.write(f"title: {title}\n")
                            file.write(f"label: {label}\n")
                            file.write(json.dumps(content, indent=4))
                            file.write("\n\n")
                    
            elif save_to == 'latex':   
                logger.info(f">> save [{section}] table to .tex")
                # conver the tables
                [table_str_to_section, table_str_to_appendix] = gen_write_tables_latex(comparison_table_amended)
                
                # append tables to non-appendix sections
                with open(os.path.join(f"{report_file_path}", f"{section_id}_{section}.tex"), 'a+') as file:
                    table_str_to_section = table_str_to_section.replace("\\begin{table}", "\\begin{table}[H]")
                    file.write(table_str_to_section)

                with open(os.path.join(f"{report_file_path}", f"table.tex"), 'a+') as file:
                    file.seek(0)
                    table_title = f"\\section{{Figure and Table}}"
                    if table_title not in file.read():
                        file.write(f"{table_title}\n")
                    if table_str_to_appendix != '':
                        table_str_to_appendix = f"\n\\subsection{{{section.replace('_', ' ').capitalize()}}}"+ '\n\n' + table_str_to_appendix
                    table_str_to_appendix = table_str_to_appendix.replace("\\begin{table}", "\\begin{table}[H]")
                    file.write(table_str_to_appendix)
                    
        elif (section == 'table') and (save_to == 'latex'): #append table_manual to table.tex, this should be done with all previous sections are processed, table.tex is final section to process
            logger.info(f">> append manual tables to [{section}]")  
            if os.path.exists(manual_table_file):
                table_manual_str = get_prompt_from(manual_table_file)
                table_manual_str = '\n\n' + table_manual_str
                with open(os.path.join(f"{report_file_path}", f"table.tex"), 'r') as file:
                    lines = file.readlines()
                with open(os.path.join(f"{report_file_path}", f"table.tex"), 'w') as file:
                    for line in lines:
                        file.write(line)
                        if line.strip() == f"\\section{{Figure and Table}}":
                            file.write(table_manual_str + '\n')
                
        elif (save_to == 'from_json_file'): #directly parse excel generated json file
            raw_json_folder =os.path.dirname(raw_json_path)
            excel_to_json = os.path.join(raw_json_folder, f'excel/json', f'{section}.json')     
            if os.path.exists(excel_to_json):
                logger.info(f">> generate from json files for [{section}]")
                # load the json file generated from excel file
                with open (excel_to_json, 'r') as f:
                    comparison_table_from_excel = json.load(f)
                [table_str_to_section, table_str_to_appendix] = gen_write_tables_latex(comparison_table_from_excel)
                # split cells in first row of the table, to make it prettier
                table_str_to_section = process_latex_file(table_str_to_section)
                # append tables to non-appendix sections
                with open(os.path.join(f"{report_file_path}", f"{section_id}_{section}.tex"), 'a+') as file:
                    table_str_to_section = table_str_to_section.replace("\\begin{table}", "\\begin{table}[H]")
                    file.write(table_str_to_section)
  
        else:
            logger.info(f"bypass [{section}] for table gen")

    def replace_sku_nick_name(self, folder_path, old_string, new_string):
        
        def replace_string_in_single_str(input_str, old_string, new_string):
            # define the pattern for a hyperlink
            hyperlink_pattern = r'(\\href{[^}]*)' + old_string
            # check if old_string is in the specified pattern
            if re.search(hyperlink_pattern, input_str):
                # if it is, do not replace it
                return input_str
            else:
                # if it's not, replace it
                input_str = re.sub(old_string, new_string, input_str, flags=re.IGNORECASE)
                return input_str
                
        def replace_string_in_file(file_path, old_string, new_string):
            with open(file_path, 'r') as file:
                filedata = file.read()
            filedata = replace_string_in_single_str(filedata, old_string, new_string)
            with open(file_path, 'w') as file:
                file.write(filedata)
        
        # list latex files
        tex_files = glob.glob(os.path.join(folder_path, '*.tex'))
        for tex_file in tex_files:
            replace_string_in_file(tex_file, old_string, new_string)
      
    ### Automated Essay Scoring for report quality validation
    def report_aes_latex(self, report_file_path: str) -> str:
        # check table number
        table_check_str = check_table_aggregation_number(report_file_path)        
        with open(os.path.join(report_file_path, f"000_validation_output.log"), 'w') as file:
            file.write(table_check_str)
            
        # check for wrong latex pattern number
        tex_check_str = count_occurrences(report_file_path)        
        with open(os.path.join(report_file_path, f"000_validation_output.log"), 'a') as file:
            file.write(tex_check_str)
            
        # check for incorrect references
        ref_check_str = count_invalid_refs(report_file_path)        
        with open(os.path.join(report_file_path, f"000_validation_output.log"), 'a') as file:
            file.write(ref_check_str)
                
        # check for incorrect references
        grammer_typo_check_str = check_grammer_error_typo(self, report_file_path)        
        with open(os.path.join(report_file_path, f"000_validation_output.log"), 'a') as file:
            file.write(grammer_typo_check_str)
        
        # remove header and tail
        remove_header_tail_folder(report_file_path) 
    
    #### exposed to external call, (e.g., infrawise or lucia)
    # create a new case
    def create_new_case(self, case_folder):
                    
        def copy_undefined_section_prompt(case_prompt_path, workload_str):
            workload_str = workload_str.replace(' ', '').split(',')
            workload_list = [workload for workload in workload_str]
            for workload in workload_list:
                empty_file = os.path.join(case_prompt_path, f'empty.md')
                prompt_file = os.path.join(case_prompt_path, f'{workload}.md')
                if not os.path.isfile(prompt_file):
                    # not exists, copy from the empty.md
                    shutil.copyfile(empty_file, prompt_file)
                    # modify the name
                    with open(prompt_file, 'r') as f:
                        content = f.read()
                    content = content.replace('{name}', workload)
                    with open(prompt_file, 'w') as f:
                        f.write(content)

        def validate_case_input(case):
            # paths
            result_summary_path = os.path.join(case, 'raw', 'result_summary_raw_multinode.json')
            author_path = os.path.join(case, 'author.json')
            entry_path = os.path.join(case, 'entry.json')
            title_path = os.path.join(case, 'title.json')

            logger.info(os.path.exists(result_summary_path))

            # check result_summary file
            if not os.path.exists(result_summary_path):
                return False
            try:
                with open(result_summary_path, 'r') as f:
                    result_summary_json = json.load(f)
                if 'target' not in result_summary_json or 'baseline' not in result_summary_json:
                    return False
            except (json.JSONDecodeError, FileNotFoundError):
                return False

            # check author.json
            if not os.path.exists(author_path):
                return False

            # check entry.json
            if not os.path.exists(entry_path):
                return False
            try:
                with open(entry_path, 'r') as f:
                    entry_json = json.load(f)[0]
                if 'dcw_str' not in entry_json or 'SKUNickName' not in entry_json:
                    return False
            except (json.JSONDecodeError, FileNotFoundError, IndexError):
                return False

            # check title.json
            if not os.path.exists(title_path):
                return False
            try:
                with open(title_path, 'r') as f:
                    title_json = json.load(f)[0]
                if 'title' not in title_json:
                    return False
            except (json.JSONDecodeError, FileNotFoundError, IndexError):
                return False

            # all checks passed
            return True

        def retrieve_blob_folder(case):
            local_blob_case = os.path.join(BLOB_LOCAL_DIR, case)
            logger.info(f'blob case folder: {local_blob_case}')
            return local_blob_case
        
        # check if the input case_folder is local or remote
        logger.info(f'input case folder: {case_folder}')
        if not case_folder.startswith('/usr/src/') and not case_folder.startswith('/home') and not case_folder.startswith('/Users/'): # azure dir
            case_folder = retrieve_blob_folder(case_folder)
        
        if validate_case_input(case_folder):
            logger.info(f'input is valide')
            # load the entry
            with open(os.path.join(case_folder, f'entry.json'), 'r') as f:
                entry_dict = json.load(f)
                
            # instantiate and abstract the dcw
            dcw_str = str(entry_dict[-1]["dcw_str"])
            sku_set_obj = SKUSet(0)
            dcw_structure = sku_set_obj.text_to_dcw(str(dcw_str))
            
            # extract the target, baseline, nickname
            target = entry_dict[-1]['dcw_str']['Design']['Target']
            baseline = entry_dict[-1]['dcw_str']['Design']['Baseline']
            SKUNickName = entry_dict[-1]['SKUNickName']
            target_nickname = entry_dict[-1]['SKUNickName']['Target']
            baseline_nickname = entry_dict[-1]['SKUNickName']['Baseline']
            workload = entry_dict[-1]['dcw_str']['Workload']
            # set SKU_SET
            target_rev = remove_prefix_case_insensitive(target, target_nickname)
            baseline_rev = remove_prefix_case_insensitive(baseline, baseline_nickname)
            sku_set = f"{target_nickname}_{target_rev}_{baseline_nickname}_{baseline_rev}"
            
            # copy, case prompt template, only copy if does not exist
            case_prompt_path = self.create_report_case(target, baseline)
            
            # copy, raw folder, alway copy the new version
            shutil.copytree(os.path.join(case_folder, 'raw'), os.path.join(case_prompt_path, 'raw'), dirs_exist_ok=True)
            # change keys
            change_json_key(os.path.join(case_prompt_path, f'raw/projected.json'), target, baseline)
            change_json_key(os.path.join(case_prompt_path, f'raw/spec.json'), target, baseline)   
            change_json_key(os.path.join(case_prompt_path, f'raw/result_summary_raw_multinode.json'), target, baseline)
            
            # copy un-predefined section prompt
            copy_undefined_section_prompt(case_prompt_path, workload)

            # copy the 'author.json' file
            shutil.copy2(os.path.join(case_folder, 'author.json'), os.path.join(case_prompt_path, 'author.json'))

            # copy the 'title.json' file
            shutil.copy2(os.path.join(case_folder, 'title.json'), os.path.join(case_prompt_path, 'title.json'))
            
            msg = f"config build new case {sku_set}"
            
        else:
            logger.error(f'input is invalide')
            sku_set = None
            dcw_structure = None
            SKUNickName = None
            workload = None
            msg = f"incorrect input files, please check them"
        return sku_set, SKUNickName, dcw_structure, workload, msg
    
    # get an exising case
    def get_sku_set(self, id: int):
        sku_set = SKUSet(id)
        sku_set_str, dcw_str, SKUNickName, dcw_structure, msg = sku_set.get_existing_build_config()
        return sku_set_str, SKUNickName, dcw_structure, msg
    
    # define report structure
    def define_report_structure(self, report_class: str, SKU_SET: str, workload_str: str):
        structure = ReportStructure()
        report_structure_dbg = structure.get_report_structure(report_class, SKU_SET, workload_str)
        return report_structure_dbg
    
    # check if an input is already existed, during register
    def is_existing_entry(self, target: str, baseline: str, entries: dict) -> (bool, str):
        # check if a newly input entry is already existing
        existing = False
        Existing_SKU_SET = ''
        for entry in entries:
            if (entry["dcw_str"]["Design"]["Target"] == target) and (entry["dcw_str"]["Design"]["Baseline"] == baseline):
                existing = True
                Existing_SKU_SET = entry["SKU_SET"]
        return existing, Existing_SKU_SET
    
    # add a new entry, during register
    def add_a_new_entry(self, target: str, baseline: str, target_nickname: str, baseline_nickname: str, entries: dict) -> dict:
        # prepare the values for the entry
        Design = {}
        Design["Target"] = target
        Design["Baseline"] = baseline
        dcw_str = {}
        dcw_str["Design"] = Design
        dcw_str["Criterion"] = "default",
        dcw_str["Workload"] = "basic, gemm, communication, superbench, training, inference"
        SKUNickName = {}
        SKUNickName["Target"] = target_nickname
        SKUNickName["Baseline"] = baseline_nickname
        # assemble an entry
        new_entry = {}
        if isinstance(entries, list):
            new_entry["idx"] = len(entries) + 1
        else:
            new_entry["idx"] = 32768
        new_entry["SKU_SET"] = (target + baseline).upper()
        new_entry["dcw_str"] = dcw_str
        new_entry["SKUNickName"] = SKUNickName
        # append it to the whole entries
        entries.append(new_entry)
        return entries

    # create a new case folder, during create
    def create_report_case(self, target: str, baseline: str):
        
        def count_files_in_directory(directory):
            return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
        
        # copy case prompt folder
        case_template_path = os.path.join(PROMPT_DIR, f'report/case')
        case_prompt_path = os.path.join(CASE_DIR, f'case_{target.lower()}_{baseline.lower()}')
        logger.info(case_template_path)
        logger.info(case_prompt_path)
        
        # only do this if case folder does not exists
        if not os.path.exists(case_prompt_path): # only do the copy if the case folder does not exist
            # elegant copy, keep repeating the copy action until the counts in destination match the source
            while True:
                try:
                    shutil.copytree(case_template_path, case_prompt_path)
                except Exception as e:
                    logger.error(f"Error copying files: {e}")
                time.sleep(30)  # delay for 30 seconds to let copy finish

                # Check if all files were copied
                if count_files_in_directory(case_template_path) == count_files_in_directory(case_prompt_path):
                    logger.info(f"All files copied. Total files: {count_files_in_directory(case_prompt_path)}")
                    break
                else:
                    logger.error(f"Files not copied correctly. Retrying...")
                    shutil.rmtree(case_prompt_path)  # remove the directory to try again
            # change keys
            change_json_key(os.path.join(case_prompt_path, f'raw/projected.json'), target, baseline)
            change_json_key(os.path.join(case_prompt_path, f'raw/spec.json'), target, baseline)   
            change_json_key(os.path.join(case_prompt_path, f'raw/result_summary_raw_multinode.json'), target, baseline)
        else: # case folder already exists
            logger.info(f"case folder {case_prompt_path} already exists, contains {count_files_in_directory(case_prompt_path)} files (ref: 37)")
        return case_prompt_path

    # for AzureDevOps pipeline build
    def modify_build_yaml(self, target: str, baseline: str, target_nickname: str, baseline_nickname: str):
        
        # find position in a list
        def find_last_matching_position(lst, match_condition_key, match_condition_value):
            position = -1
            for i, item in enumerate(lst):
                if isinstance(item, dict):
                    if match_condition_key in item:
                        if match_condition_value in item.get(match_condition_key):
                            position = i
            return position
                            
        
        # retieve build blocks from existing yaml file
        yaml = YAML()
        three_upper_levels_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        azure_pipeline_build_yaml = os.path.join(three_upper_levels_path, f'.azure-pipelines/report_dev_mi300x_pv2.yml')
        with open(azure_pipeline_build_yaml, 'r') as f:
            pipeline_build_dict = yaml.load(f)
        pipeline_steps = pipeline_build_dict["steps"]
        
        # find the position to insert the new build script and new copy task
        pos_last_build = find_last_matching_position(pipeline_steps, "script", "sed -i")
        pos_last_copy = find_last_matching_position(pipeline_steps, "task", "CopyFiles")
        pos_last_post = find_last_matching_position(pipeline_steps, "task", "PublishBuildArtifacts")
        
        # make sure the quoting state is not changed
        pipeline_build_dict['pool']['vmImage'] = sq('ubuntu-latest')
        pipeline_build_dict['steps'][pos_last_post]['inputs']['pathToPublish'] = sq('$(Build.ArtifactStagingDirectory)')
        
        
        # assemble the new build script item
        new_build_external_dict = {}
        new_build_internal_dict = {}
        new_build_external_dict["script"] = f'make -C InfraWise/report/InfraWise_reports_{target}_{baseline}/dev && cp InfraWise/report/InfraWise_reports_{target}_{baseline}/dev/{target}_report.pdf InfraWise/report/InfraWise_reports_{target}_{baseline}/dev/Azure_{target_nickname}_AI_Platform_Quality_{target_nickname}_vs_{baseline_nickname}_external.pdf'
        new_build_external_dict["displayName"] = f'Build {target.upper()} {baseline.upper()}, external'
        new_build_internal_dict["script"] = f"sed -i 's/\\\\newcommand{{\\\\msrvshort}}\\[1\\]{{MSRV}}/\\\\newcommand{{\\\\msrvshort}}\\[1\\]{{MSRA (Vancouver)}}/g' InfraWise/report/InfraWise_reports_{target}_{baseline}/dev/{target}_report.tex && sed -i 's/\\\\newcommand{{\\\\msrvlong}}\\[1\\]{{Microsoft Research Vancouver}}/\\\\newcommand{{\\\\msrvlong}}\\[1\\]{{Microsoft Research Asia, Vancouver}}/g' InfraWise/report/InfraWise_reports_{target}_{baseline}/dev/{target}_report.tex && make -C InfraWise/report/InfraWise_reports_{target}_{baseline}/dev && cp InfraWise/report/InfraWise_reports_{target}_{baseline}/dev/{target}_report.pdf InfraWise/report/InfraWise_reports_{target}_{baseline}/dev/Azure_{target_nickname}_AI_Platform_Quality_{target_nickname}_vs_{baseline_nickname}_internal.pdf"
        new_build_internal_dict["displayName"] = f'Build {target.upper()} {baseline.upper()}, internel'
        # assemble the new copy task item
        new_copy_dict = {}
        new_copy_dict["task"] = 'CopyFiles@2'
        new_copy_dict["inputs"] = {}
        new_copy_dict["inputs"]["contents"] = f'InfraWise/report/InfraWise_reports_{target}_{baseline}/dev/Azure_{target_nickname}_AI_Platform_Quality_{target_nickname}_vs_{baseline_nickname}_external.pdf\nInfraWise/report/InfraWise_reports_{target}_{baseline}/dev/Azure_{target_nickname}_AI_Platform_Quality_{target_nickname}_vs_{baseline_nickname}_internal.pdf'
        new_copy_dict["inputs"]["targetFolder"] = f'$(Build.ArtifactStagingDirectory)'
        
        # insert in revers order
        pipeline_steps_new = pipeline_steps.copy()
        pipeline_steps_new.insert(pos_last_copy + 1, new_copy_dict)
        pipeline_steps_new.insert(pos_last_build + 1, new_build_internal_dict)
        pipeline_steps_new.insert(pos_last_build + 1, new_build_external_dict)
        
        # write back the yaml file
        pipeline_build_dict_new = pipeline_build_dict.copy()
        pipeline_build_dict_new["steps"] = pipeline_steps_new
        
        with open(azure_pipeline_build_yaml, 'w') as f:
            yaml.dump(pipeline_build_dict_new, f)

    def extract_section_to_modify(self, user_input, report_structure):
        #logger.info(report_structure)
        valid_sections = []
        for valid_sec in report_structure:
            if valid_sec["sec_category"] in ["es", "benchmark", "conclusion", "appendix", "manual"]:
                valid_file = f"{valid_sec['section_id']}_{valid_sec['sec_shortname']}"
                valid_sections.append(valid_file)
        
        logger.info(f'valid_sections is {valid_sections}')
                
        # extract the section name which user intent to modify
        system_prompt = get_prompt_from(os.path.join(PROMPT_DIR, f'extract_section_name.md'))
        user_prompt = f'user input =\n{user_input}\n\navailable section names =\n{str(valid_sections)}\n\n'
        # extract
        resp = self.call_openai_api(system_prompt, user_prompt)
        extracted_section_name = remove_header_tail(resp, f'```text\n', f'\n```')
        logger.info(f'user_input is {user_input}')
        logger.info(f'extracted_section_name is {extracted_section_name}')
        return extracted_section_name, valid_sections

    # rebuild one section
    def regenerate_one_section(self, dcw, user_input, report_structure):
        id_section, valid_sections = self.extract_section_to_modify(user_input, report_structure)
        logger.info(valid_sections)
        
        # locate the latex folder
        report_file_root_first = os.path.join(BUILD_TEMP_DIR, f"InfraWise_reports_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}")
        sub_dir_first = "dev"
        report_file_path = os.path.join(report_file_root_first, sub_dir_first)
        
        # do the modification
        if id_section in valid_sections:            
            for section in report_structure:
                if str(section['section_id']) == id_section.split("_")[0]:
                    logger.info(f'rebuilding section_id = {id_section.split("_")[0]}, section_name = {id_section.split("_")[1]}')
                    self.rebuild_one_section(dcw, section, report_file_path, report_structure)
            
            skip_pdf_latex = False
            output = f'rebuild section: {id_section} finished, '
        else:    
            skip_pdf_latex = True
            output = f'rebuild section skipped, do not exist: {id_section}'
        return skip_pdf_latex, output

    # change prompt
    def modify_prompt(self, dcw, user_input, report_structure):
        id_section, valid_sections = self.extract_section_to_modify(user_input, report_structure)
        logger.info(valid_sections)
        
        # locate the latex folder
        report_file_root_first = os.path.join(BUILD_TEMP_DIR, f"InfraWise_reports_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}")
        sub_dir_first = "dev"
        report_file_path = os.path.join(report_file_root_first, sub_dir_first)
        
        # do the modification
        if id_section in valid_sections:
            # get old section template
            case_template_file = os.path.join(CASE_DIR, f'case_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}', f'{id_section}.md')
            case_template = get_prompt_from(case_template_file)
            # split the content into two parts
            parts = case_template.split('[case]', 1)
            logger.info(f'old content:\n{parts[0]}')
            
            # get prompts for modifying the template
            system_prompt = get_prompt_from(os.path.join(PROMPT_DIR, f'modify_template.md'))
            user_prompt = f'user input =\n{user_input}\n\nprompt template =\n{parts[0]}\n\n'
            # modify
            resp = self.call_openai_api(system_prompt, user_prompt)
            modified_part = remove_header_tail(resp, f'```text', f'```')
            
            # combine the modified first part and the original second part
            if len(parts) > 1:
                new_content = '[case]'.join([modified_part, parts[1]])
            else:
                new_content = modified_part
            # write the new content back to the same file
            with open(case_template_file, 'w') as file:
                file.write(new_content)
            logger.info(f'new content:\n{new_content}')
            
            for section in report_structure:
                if str(section['section_id']) == id_section.split("_")[0]:
                    logger.info(f'rebuilding section_id = {id_section.split("_")[0]}, section_name = {id_section.split("_")[1]}')
                    self.rebuild_one_section(dcw, section, report_file_path, report_structure)
            
            output = f'finished editing: {id_section}.md, only {id_section}.tex is regenerated'
            skip_pdf_latex = False
        else:    
            output = f'do not exist: {id_section}.md'
            skip_pdf_latex = True
        return skip_pdf_latex, output
    
    # change latex
    def modify_latex(self, dcw, user_input, report_structure):
        section, valid_sections = self.extract_section_to_modify(user_input, report_structure)
        logger.info(section)
        logger.info(valid_sections)
        
        # locate the latex folder
        report_file_root_first = os.path.join(BUILD_TEMP_DIR, f"InfraWise_reports_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}")
        sub_dir_first = "dev"
        report_file_path = os.path.join(report_file_root_first, sub_dir_first)
        tex_file_to_modify = os.path.join(report_file_path, f'{section}.tex')
        
        skip_pdf_latex = True
        
        # do the modification
        if section in valid_sections and os.path.isfile(tex_file_to_modify):
            # get existing section latex
            with open(tex_file_to_modify, 'r') as f:
                existing_tex = f.read()

            # split the content into parts
            parts = existing_tex.split(f'\\begin{{table}}', 1)
            logger.info(f'splited into {len(parts)} parts')
            logger.info(f'old content:\n{parts[0]}')
            
            # get prompts for modifying the template
            system_prompt = get_prompt_from(os.path.join(PROMPT_DIR, f'modify_template.md'))
            user_prompt = f'user input =\n{user_input}\n\nprompt template =\n{parts[0]}\n\n'
            # modify
            resp = self.call_openai_api(system_prompt, user_prompt)
            modified_part = remove_header_tail(resp, f'```text', f'```')
            prefixes_to_remove = ["\\end{document}", "\\documentclass", "\\usepackage", "\\begin{document}"]
            modified_part = remove_lines_starting_with(modified_part, prefixes_to_remove)
            
            # combine the modified first part and the original second part
            if len(parts) > 1:
                new_content = f'\n\n\\begin{{table}}'.join([modified_part, parts[1]])
            else:
                new_content = modified_part
            # write the new content back to the same file
            with open(tex_file_to_modify, 'w') as file:
                file.write(new_content)
            logger.info(f'new content:\n{new_content}')
            
            output = f'finished editing: {section}'
            skip_pdf_latex = False
        else:    
            output = f'do not exist: {section}.tex'
            skip_pdf_latex = True
        return skip_pdf_latex, output

    def rebuild_one_section(self, dcw, section, report_file_path, report_structure):
        # get exisintg tex:
        sec_id = section["section_id"]
        sec_shortname = section["sec_shortname"]
        tex_file = os.path.join(report_file_path, f"{sec_id}_{sec_shortname}.tex")
        
        # prepare for regenerate summary
        sections_list=[]
        for element in report_structure:
            #logger.info(element['section_id'])
            #logger.info(element['sec_category'])
            #logger.info(element['content'])
            if element['sec_category'] == 'benchmark' or element['sec_category'] == 'manual':
                sections_list.append(section['content'])
        
        if os.path.exists(tex_file):
            before_table, after_table = split_tex_content(tex_file)
            if section['sec_category'] == 'benchmark' or section['sec_category'] == 'manual':
                self.section_worker(1, dcw, section, report_file_path, 'latex', False)
            elif section['sec_category'] == 'es' or section['sec_category'] == 'conclusion' or section['sec_category'] == 'appendix':
                self.summary_worker(1, dcw, section, sections_list, report_file_path, 'latex', False)
            append_after_table_content(tex_file, after_table)
            logger.info(f"{tex_file}, re_build finished.")
        else:
            logger.info(f"no file found {tex_file}, regenerate skipped.")

    #### agent buid pipeline functions
    # parse table
    def parse_raw_tables(self, report_content_dict, dcw, SKUNickName, report_file_path):
        for item in report_content_dict:
            self.fill_content_table(dcw, SKUNickName, item["section_id"], item["sec_shortname"], report_file_path, save_to='case')

    # remove existing build
    def remove_existing_build(self, folder_path):
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            files = glob.glob(f'{folder_path}/*')
            for f in files:
                if os.path.isfile(f):
                    os.remove(f)
                elif os.path.isdir(f):
                    shutil.rmtree(f)
            logger.info(f"Content of the folder {folder_path} has been removed.")
        else:
            logger.info(f"Folder {folder_path} does not exist.")
    
    # generate section
    def generate_sections(self, report_file_path_list, report_content_dict_list, dcw, OUTPUT_FORMAT, BYPASS_GPT):
        threads = []
        start = datetime.now()
        for i, (report_file_path, report_content_dict) in enumerate(zip(report_file_path_list, report_content_dict_list)):
            self.fill_content_benchmark_appendix(i, dcw, report_content_dict, report_file_path, OUTPUT_FORMAT, BYPASS_GPT)
            logger.info(datetime.now() - start)
            t = threading.Thread(target = self.fill_content_benchmark_appendix, args=(i, dcw, report_content_dict, report_file_path, OUTPUT_FORMAT, BYPASS_GPT))
        for t in threads:
            t.join()

    # generate summary
    def generate_summary_and_conclusion(self, report_file_path_list, report_content_dict_list, dcw, OUTPUT_FORMAT, BYPASS_GPT, SKU_SET):
        import time
        if SKU_SET in ["MACHFPGAA100H100"]:
            USE_MULTITHREADING = False
            time.sleep(30)
        else:
            USE_MULTITHREADING = True
        threads = []
        start = datetime.now()
        for i, (report_file_path, report_content_dict) in enumerate(zip(report_file_path_list, report_content_dict_list)):
            self.fill_content_summary_conclusion(i, dcw, report_content_dict, report_file_path, OUTPUT_FORMAT, BYPASS_GPT, USE_MULTITHREADING)
            logger.info(datetime.now() - start)
            t = threading.Thread(target = self.fill_content_summary_conclusion, args=(i, dcw, report_content_dict, report_file_path, OUTPUT_FORMAT, BYPASS_GPT, USE_MULTITHREADING))
        for t in threads:
            t.join()

    # generate table section
    def generate_table_sections(self, report_file_path_list, report_content_dict, dcw, SKUNickName):
        for i, report_file_path in enumerate(report_file_path_list):
            logger.info(f"Build Attempt: {i}")
            for item in report_content_dict:
                self.fill_content_table(dcw, SKUNickName, item["section_id"], item["sec_shortname"], report_file_path, save_to='latex')

    # combine report
    def combine_report(self, report_file_path_list, dcw, REPORT_CLASS, report_content_dict):
        for i, report_file_path in enumerate(report_file_path_list):
            logger.info(f"Build Attempt: {i}")
            self.combine_main_latex(dcw, report_file_path, REPORT_CLASS)
            with open(f"{report_file_path}/zfinal_log.json", "w") as file:
                json.dump(report_content_dict, file, indent=4)

    # validate report
    def validate_report(self, report_file_path_list, dcw, SKU_SET, SKUNickName):
        for i, report_file_path in enumerate(report_file_path_list):
            logger.info(f"Build Attempt: {i}")
            if SKU_SET in ["MACHFPGAA100H100"]:
                self.replace_sku_nick_name(report_file_path, 'machfpga', 'MACH FPGA')
                self.replace_sku_nick_name(report_file_path, 'a100pcie', 'A100 PCIe')
                self.replace_sku_nick_name(report_file_path, 'h100sxm', 'H100 SXM')
                self.replace_sku_nick_name(report_file_path, 'h100nvl', 'H100 NVL')
                self.replace_sku_nick_name(report_file_path, dcw.Design.Target, 'MACH FPGA')
                self.replace_sku_nick_name(report_file_path, dcw.Design.Baseline, 'A100 H100')
            else:
                self.replace_sku_nick_name(report_file_path, dcw.Design.Target, SKUNickName["Target"])
                self.replace_sku_nick_name(report_file_path, dcw.Design.Baseline, SKUNickName["Baseline"])
            self.report_aes_latex(report_file_path)

    # generate report content
    def generate_report_content(self, report_structure_dbg, BUILD_TIME, dcw):        
        SAVE_TO_DEV = False
        report_content_dict = report_structure_dbg.copy()
        report_file_root_first = os.path.join(BUILD_TEMP_DIR, f"InfraWise_reports_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}")
        report_file_root_local = os.path.join(BUILD_TEMP_DIR, f"InfraWise_reports_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}")
        sub_dir_first = "dev"
        sub_dir_local = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        report_file_path_list = []
        report_content_dict_list = []
        for build in range(BUILD_TIME):
            if build == 0:
                report_file_root = report_file_root_first
                sub_dir = sub_dir_first
            else:
                report_file_root = report_file_root_local
                sub_dir = sub_dir_local + '_' + str(build)
            report_file_path = os.path.join(report_file_root, sub_dir)
            report_file_path_list.append(report_file_path)
            report_content_dict_list.append(report_content_dict)
        return report_file_path_list[0], report_content_dict_list[0], report_file_path_list, report_content_dict_list

    # generate report structure
    def generate_report_structure(self, report_file_path_list, report_content_dict_list):
        for i, (report_file_path, report_content_dict) in enumerate(zip(report_file_path_list, report_content_dict_list)):
            logger.info(f"Build Attempt: {i}")
            if not os.path.exists(report_file_path):
                os.makedirs(report_file_path)
            logger.info(f'contents of sections are saved to {report_file_path}')
            with open(f"{report_file_path}/zzz_log.json", "w") as file:
                json.dump(report_content_dict, file, indent=4)

    # make pdf
    def build_pdf(self, report_file_path, dcw, SKUNickName):
        import subprocess
        
        target = dcw.Design.Target.lower()
        baseline = dcw.Design.Baseline.lower()
        target_nick = SKUNickName["Target"].upper()
        baseline_nick = SKUNickName["Baseline"].upper()
        logger.info(f'{target} vs. {baseline}')
        logger.info(f'{target_nick} vs. {baseline_nick}')
        
        # get timestamp
        timestamp = str(datetime.now().strftime('%Y%m%d-%H%M%S'))

        # build
        logger.info(f'building report output, format = pdf, external')
        # make command
        make_command = f"make -C {report_file_path}"
        # check if Makefile exists
        if os.path.isfile(os.path.join(report_file_path, 'Makefile')):
            logger.info(f'Makefile exists')
        else:
            logger.info(f'Makefile does not exist')
        process = subprocess.Popen(make_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        #if output:
            #logger.info(output.strip())
        if error:
            logger.error(f"Error in make command: {error.strip()}")
        # cp command
        logger.info(f'start copy pdf')
        build_directory = os.path.join(BUILD_DIR, f'{target}_{baseline}')
        # Check if build_directory exists, if not create it
        release_file_name = f"PerfGate_{target_nick}vs{baseline_nick}_{target}_{timestamp}"
        if not os.path.exists(build_directory):
            os.makedirs(build_directory)
        external_pdf_file = f"{build_directory}/{release_file_name}.pdf"
        cp_pdf_command = f"cp {report_file_path}/{target}_report.pdf {external_pdf_file}"
        process = subprocess.Popen(cp_pdf_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        if error:
            logger.error(f"Error in cp command: {error.strip()}")
        logger.info(f'local saved')
        # convert to docx
        if False:
            logger.info(f'start copy docx')
            convert_pdf_to_word(f"{report_file_path}/{target}_report.pdf", f"{report_file_path}/{target}_report.docx")
            external_docx_file = f"{build_directory}/{release_file_name}.docx"
            cp_docx_command = f"cp {report_file_path}/{target}_report.docx {external_docx_file}"
            process = subprocess.Popen(cp_docx_command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            if error:
                logger.error(f"Error in cp command: {error.strip()}")
            logger.info(f'local saved')
        # upload to azure blob
        # download a directory from Azure Blob Storage
        # OP2
        # sub = Shuguang Liu's Team Dev/Test
        # account = lequ

        # read the environment variable and determine whether to upload to Azure
        SKIP_AZURE_SAVE = os.environ.get('REPORTGEN_SKIP_AZURE_SAVE', 'True') == 'True'
        logger.info(f'SKIP_AZURE_SAVE value is {SKIP_AZURE_SAVE}')
        if SKIP_AZURE_SAVE:
            azure_path = 'skipped'
        else:
            # local blob dir
            local_blob_build_file = os.path.join(BLOB_LOCAL_DIR, f'build/{target_nick}_{baseline_nick}', os.path.basename(external_pdf_file))
            logger.info(f'local blob build file: {local_blob_build_file}')

            # ensure the directory exists before copying the file
            local_blob_dir = os.path.dirname(local_blob_build_file)
            if not os.path.exists(local_blob_dir):
                os.makedirs(local_blob_dir)
            
            # copy the external PDF file to the local blob build directory
            try:
                shutil.copy(external_pdf_file, local_blob_build_file)
                logger.info(f'Copied {external_pdf_file} to {local_blob_build_file}')
            except FileNotFoundError:
                logger.error(f"File not found: {external_pdf_file}")
            except Exception as e:
                logger.error(f"Error copying file: {e}")

            # azure blob dir
            bloburl = "https://luciastore.blob.core.windows.net/"
            container = "mount"
            dest_file = os.path.join(f"agents/infrawise-report-gen/build/{target_nick}_{baseline_nick}", os.path.basename(external_pdf_file))
            azure_path = os.path.join(bloburl, container, dest_file)
            logger.info(f'azure saved')

        return external_pdf_file, azure_path
    
    # build run
    def build_run(self, report_structure_dbg, BUILD_TIME, dcw, SKUNickName, OUTPUT_FORMAT, BYPASS_GPT, SKU_SET, REPORT_CLASS):
    
        report_file_path, report_content_dict, report_file_path_list, report_content_dict_list = self.generate_report_content(report_structure_dbg, BUILD_TIME, dcw)
        self.remove_existing_build(report_file_path)
        self.generate_report_structure(report_file_path_list, report_content_dict_list)
        self.change_markdown_index(dcw, report_content_dict)
        self.parse_raw_tables(report_content_dict, dcw, SKUNickName, report_file_path)
        self.generate_sections(report_file_path_list, report_content_dict_list, dcw, OUTPUT_FORMAT, BYPASS_GPT)
        self.generate_summary_and_conclusion(report_file_path_list, report_content_dict_list, dcw, OUTPUT_FORMAT, BYPASS_GPT, SKU_SET)
        # this is where report_content_dict_list is updated
        if not BYPASS_GPT:
            self.do_translate(report_file_path_list, SKUNickName, self.translate_language)
        self.generate_table_sections(report_file_path_list, report_content_dict, dcw, SKUNickName)
        self.combine_report(report_file_path_list, dcw, REPORT_CLASS, report_content_dict)
        if not BYPASS_GPT:
            self.validate_report(report_file_path_list, dcw, SKU_SET, SKUNickName)
        external_pdf_file, azure_path = self.build_pdf(report_file_path, dcw, SKUNickName)
        # generate a small paragraph for summary
        self.gen_quick_message(report_file_path_list, dcw)
        return external_pdf_file, azure_path, report_content_dict_list[0] # multi build, only return the first build
    
    # generate prompt
    def generate_prompt(self, report_structure_dbg, BUILD_TIME, dcw, SKUNickName, OUTPUT_FORMAT, BYPASS_GPT, SKU_SET, REPORT_CLASS):
    
        report_file_path, report_content_dict, report_file_path_list, report_content_dict_list = self.generate_report_content(report_structure_dbg, BUILD_TIME, dcw)
        self.remove_existing_build(report_file_path)
        self.generate_report_structure(report_file_path_list, report_content_dict_list)
        self.change_markdown_index(dcw, report_content_dict)
        self.parse_raw_tables(report_content_dict, dcw, SKUNickName, report_file_path)

    
    def build_run_with_timeout(self, report_structure_dbg, BUILD_TIME, dcw, SKUNickName, OUTPUT_FORMAT, BYPASS_GPT, SKU_SET, REPORT_CLASS):
        try:
            external_pdf_file, azure_path, report_content_dict_list = self.build_run_timeout(report_structure_dbg, BUILD_TIME, dcw, SKUNickName, OUTPUT_FORMAT, BYPASS_GPT, SKU_SET, REPORT_CLASS)
        except Exception as e:
            # Handle the error here
            logger.error(f"An error occurred: {e}")
            # You can also log the error or take other actions as needed
            external_pdf_file, azure_path, report_content_dict_list = None, None, []
        return external_pdf_file, azure_path, report_content_dict_list
    
    # build run, rerun in case of error or timeout
    @staticmethod
    def _build_run_target_static(result, exception, report_structure_dbg, BUILD_TIME, dcw, SKUNickName, OUTPUT_FORMAT, BYPASS_GPT, SKU_SET, REPORT_CLASS):
        """Static target function for multiprocessing - avoids pickle issues with self"""
        try:
            # Create a new Report instance inside the process to avoid pickling issues
            report_instance = Report()
            result.append(report_instance.build_run(report_structure_dbg, BUILD_TIME, dcw, SKUNickName, OUTPUT_FORMAT, BYPASS_GPT, SKU_SET, REPORT_CLASS))
        except Exception as e:
            exception.append(e)

    def build_run_timeout(self, report_structure_dbg, BUILD_TIME, dcw, SKUNickName, OUTPUT_FORMAT, BYPASS_GPT, SKU_SET, REPORT_CLASS):
        max_retry_cnt = 5

        result = multiprocessing.Manager().list()
        exception = multiprocessing.Manager().list()


        process = multiprocessing.Process(target=Report._build_run_target_static, args=(result, exception, report_structure_dbg, BUILD_TIME, dcw, SKUNickName, OUTPUT_FORMAT, BYPASS_GPT, SKU_SET, REPORT_CLASS))
        process.start()
        # read timeout threshold
        BUILD_TIMEOUT = int(os.environ.get('REPORTGEN_TIMEOUT', 180)) # in seconds
        logger.info(f'BUILD_TIMEOUT value is {BUILD_TIMEOUT}')
        timeout = BUILD_TIMEOUT
        process.join(timeout = timeout)  # timeout value in seconds

        if process.is_alive():
            logger.error(f"build_run() is taking longer than {timeout} seconds. Restarting...")
            process.terminate()
            process.join()
            self.build_retry_cnt += 1  # increment retry_count
            logger.info(f'self.build_retry_cnt is {self.build_retry_cnt}')
            if self.build_retry_cnt <= max_retry_cnt:  # check if retry_count is less than or equal to 5
                return self.build_run_timeout(report_structure_dbg, BUILD_TIME, dcw, SKUNickName, OUTPUT_FORMAT, BYPASS_GPT, SKU_SET, REPORT_CLASS)
            else:
                self.build_retry_cnt = 0  # reset retry count
                raise Exception("Maximum retry attempts exceeded.")
        elif exception:
            logger.error(f"build_run() raised an exception: {exception[0]}. Restarting...")
            self.build_retry_cnt += 1  # increment retry_count
            logger.info(f'self.build_retry_cnt is {self.build_retry_cnt}')
            if self.build_retry_cnt <= max_retry_cnt:  # check if retry_count is less than or equal to 5
                return self.build_run_timeout(report_structure_dbg, BUILD_TIME, dcw, SKUNickName, OUTPUT_FORMAT, BYPASS_GPT, SKU_SET, REPORT_CLASS)
            else:
                self.build_retry_cnt = 0  # reset retry count
                raise Exception("Maximum retry attempts exceeded.")
        else:
            self.build_retry_cnt = 0  # reset retry count
            return result[0]
        
    # only pdflatex
    def pdf_run(self, report_structure_dbg, BUILD_TIME, dcw, SKUNickName, SKU_SET, BYPASS_GPT):
    
        report_file_path, report_content_dict, report_file_path_list, report_content_dict_list = self.generate_report_content(report_structure_dbg, BUILD_TIME, dcw)
        if not BYPASS_GPT:
            self.validate_report(report_file_path_list, dcw, SKU_SET, SKUNickName)
        external_pdf_file, azure_path = self.build_pdf(report_file_path, dcw, SKUNickName)
        return external_pdf_file, azure_path, report_content_dict

    def change_markdown_index(self, dcw, report_content_dict):
        for sec in report_content_dict:
            sec_id = sec["section_id"]
            sec_name = sec["sec_shortname"]
            sec_file = os.path.join(CASE_DIR, f'case_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}', f'{sec_name}.md')
            new_sec_file = os.path.join(CASE_DIR, f'case_{dcw.Design.Target.lower()}_{dcw.Design.Baseline.lower()}', f'{sec_id}_{sec_name}.md')
            
            if os.path.exists(sec_file) and not os.path.exists(new_sec_file):
                shutil.copy(sec_file, new_sec_file)

        logger.info(report_content_dict)

    # register a new case
    def register_new_report(self):
        output_msg = ''
        
        def get_sku(sku_type: str) -> str:
            output_msg = ''
            sku_type = sku_type.upper()
            prompt = f"Enter the name of {sku_type} SKU, include a specific FW version if applicable (e.g.: mi300xbkc2406): "
            sku_str = input(prompt).lower()
            logger.info(f"The {sku_type} is: {sku_str}")
            output_msg = '/n'
            return sku_str, output_msg

        def get_nick_name(sku_type: str) -> str:
            output_msg = ''
            sku_type = sku_type.upper()
            prompt = f"Enter the nick name of {sku_type} SKU, this will be exactly what SLT sees in the report, it should be simple (e.g.: MI300): "
            sku_nickname = input(prompt)
            logger.info(f"The {sku_type} SKU is: {sku_nickname}")
            output_msg = '/n'
            return sku_nickname, output_msg
        
        def valid_input(choice: str, count: int, key_word: str = 'new') -> bool:
            if choice == 'new' or choice == 'end':
                return True
            elif is_integer(choice):
                if (int(choice) > 0) and (int(choice) < (count + 1)):
                    return True
                else:
                    return False

        # read the entry files
        entry_file = os.path.join(PROMPT_DIR, f'report/report_entry/entry.json')
        if os.path.isfile(entry_file):
            with open(entry_file) as f:
                entries = json.load(f)
            logger.info('entry file read success')
        else:
            logger.error('no entry file')

        # load entries
        # prepare registries
        output_msg += f"Please choose from below existing report entries or create a new entry:\n"
        for entry in entries:
            registry_idx = entry["idx"]
            registered_target = entry["dcw_str"]["Design"]["Target"]
            registered_baseline = entry["dcw_str"]["Design"]["Baseline"]
            registry_message = f"Input {registry_idx} to build for target = {registered_target} and baseline = {registered_baseline}\n"
            output_msg += registry_message
        output_msg += f"Input 'new' to create a new report entry\n"
        
        # check and wait until receives a valid input
        while True:
            choice = input("Please select which report you want to build by input the index: ")
            if valid_input(choice, len(entries)):
                break
            logger.error("Invalid input. Please try again.")

        if choice == 'new':
            if self._debug:
                target = 'aaa'
                baseline = 'ccc'
                target_nickname = 'AAA'
                baseline_nickname = 'CCC'
                existing = False
                Existing_SKU_SET = 'NA'
            else:
                # 0. Ask for: target, baselin, target nick name, baseline nick name
                target, msg_target = get_sku('target')
                baseline, msg_baseline = get_sku('baseline')
                target_nickname, msg_target_name = get_nick_name('target')
                baseline_nickname, msg_baseline_name = get_nick_name('baseline')
                output_msg = msg_target + msg_baseline + msg_target_name + msg_baseline_name
                # 1. check if the new sku pair already existes
                existing, Existing_SKU_SET = self.is_existing_entry(target, baseline, entries)
            if existing:
                logger.info(f"The SKUs you want to benchmark are already benchmarked")
                SKU_SET = Existing_SKU_SET
            else:
                # 2. create a new SKU_SET object and register the entry into entry file
                if self._debug:
                    pass
                else:                
                    entries = self.add_a_new_entry(target, baseline, target_nickname, baseline_nickname, entries)
                with open(entry_file, 'w') as f:
                    json.dump(entries, f, indent=4)
                # 3. prepare folders to hold the data and non-general prompt
                case_prompt_path = self.create_report_case(target, baseline)
                self.modify_build_yaml(target, baseline, target_nickname, baseline_nickname)
                logger.info(f"The new SKU_SET build entry has been registered")
                # 4. prompt a message to users to copy the data
                logger.info(f"\nplease do the follow steps manually:\ncopy the content of below json files into files in\n{case_prompt_path}\n'projected.json -> projected performance'\n'spec.json -> theoretical performance'\n'result_summary_raw.json -> single node SuperBench output'\n'result_summary_raw_multinode.json -> multi node SuperBench'\n\nmodify the report title and author list as you with in\n'title.son'\n'author.json'\n\n")
                SKU_SET = (target + baseline).upper()
        elif (choice == 'end'): # build existing
            SKU_SET = 'NA'
        else: # build existing
            SKU_SET = entries[int(choice) - 1]['SKU_SET']
        
        return SKU_SET, output_msg

    # change parameter value
    def set_parameter_value(self, user_message, workload):
        prompt_file_name = f'modify_parameter.md'
        system_prompt = get_prompt_from(os.path.join(PROMPT_DIR, prompt_file_name))
        if workload is None:
            workload = "['abstract','background','es','basic','basicmach','efficiency_sim','gemm','communication','superbench','training','inference','inferencemach','inference_sim','llama2_sim','multinode_ibwrite','multinode_communication','multinode_superbench','multinode_inference','multinode_sb_scale','multinode_inf_scale','accuracy','functionality','conclusion','projected','peak','spec_comparison','method']"
        system_prompt = system_prompt.replace("{workload}", workload)
        logger.info(system_prompt)
        system_prompt = f'{system_prompt}'
        user_prompt = f'[user input] \n {user_message}'
        resp = self.call_openai_api(system_prompt, user_prompt)
        #logger.info(f'system_prompt is {system_prompt}\nuser_prompt is {user_prompt}\nresponse is {resp}')
        r_str = extract_json(resp, nested=False)
        r_dict = json.loads(r_str)
        logger.info(r_dict)
        logger.info(type(r_dict))
        if r_dict is not None and 'name' in r_dict and 'value' in r_dict:
            parameter_to_change = r_dict['name']
            new_value = r_dict['value']
        else:
            parameter_to_change = 'none'
            new_value = 'none'
        return parameter_to_change, new_value

    # improve expression in the latex contents
    def improve_expression(self, original_content: str, improvement_reason: str) -> str:
        # improvement prompt
        sys_prompt = f"Your task is to improve the original expression based on the improvement reason, and output the expression after improvement. You must preserve the latex format.\n"
        # user prompt
        user_prompt = f"[original expression] is \n{original_content}\n\n [improvement reason] is \n{improvement_reason}\n\n"
        
        output =self.call_openai_api(sys_prompt, user_prompt)
        
        return output

    # clear the appendix table
    def clear_appendix_table(self, report_file_path: str):
        table_tex_file = os.path.join(f"{report_file_path}", f"table.tex")
        
        # Check if the file exists
        if os.path.exists(table_tex_file):
            # If it exists, delete it
            os.remove(table_tex_file)

        # Create a new file
        with open(table_tex_file, 'w') as file:
            pass  # Do nothing, just create the file

    def gen_quick_message(self, report_file_path_list, dcw):
        logger.info(f'Generating quick summary')
        # check if report_file_path_list is a list and not empty
        if isinstance(report_file_path_list, list):
            if report_file_path_list:
                folder = report_file_path_list[0]

                # check if the folder exists, flag to check if 'summary.tex' file is found
                if os.path.isdir(folder):
                    summary_found = False

                    # iterate over files in the folder
                    for file_name in os.listdir(folder):
                        # check if the file name contains 'summary.tex'
                        if 'summary.tex' in file_name:
                            summary_found = True
                            summary_file_name = file_name
                            break  # xxit the loop if 'summary.tex' is found

                    # create a new file 'quick_summary.md' in the folder
                    quick_summary_path = os.path.join(folder, 'quick_summary.md')
                    with open(quick_summary_path, 'w') as quick_summary_file:
                        # write 'haha' if 'summary.tex' is found, otherwise write 'hehe'
                        if summary_found:
                            with open(os.path.join(folder, summary_file_name), 'r') as f:
                                user_prompt = f.read()
                            sys_prompt = get_prompt_from(os.path.join(PROMPT_DIR, f"report", f'gen_quick_summary.md'))
                            resp = self.call_openai_api(sys_prompt, user_prompt)
                            quick_summary_file.write(resp)
                        else:
                            quick_summary_file.write('hehe')

    # translate
    def do_translate(self, report_file_path_list, SKUNickName, language):
        for i, report_file_path in enumerate(report_file_path_list):
            logger.info(f"Build Attempt: {i}")
            translate_build_ins(self, report_file_path, [SKUNickName["Target"], SKUNickName["Baseline"]], language)

class SKUSet:
    def __init__(self, id: int):
        self.id = id
        self.entry_file = os.path.join(PROMPT_DIR, 'report/report_entry/entry.json')
        self.entries = self.load_entries()

    def load_entries(self):
        if os.path.isfile(self.entry_file):
            with open(self.entry_file) as f:
                return json.load(f)
        else:
            raise FileNotFoundError('No entry file found')

    def extract_dcw_entries(self, SKU_SET):
        for entry in self.entries:
            if entry["SKU_SET"] == SKU_SET:
                return entry["dcw_str"], entry["SKUNickName"]
        return '', ''

    def text_to_dcw(self, text: str) -> DCW:
        text_dict = ast.literal_eval(text)
        design = Design(Target=text_dict['Design']['Target'], Baseline=text_dict['Design'].get('Baseline', None))
        return DCW(Design=design, Criterion=text_dict['Criterion'][0], Workload=text_dict['Workload'])

    def get_existing_build_config(self):
        output_msg = "Please choose from below existing report entries or create a new entry:\n"
        for entry in self.entries:
            registry_message = f"Input {entry['idx']} to build for target = {entry['dcw_str']['Design']['Target']} and baseline = {entry['dcw_str']['Design']['Baseline']}\n"
            output_msg += registry_message
        output_msg += f"Input 'new' to create a new report entry\n"
        output_msg += f"Input {self.id} received!!!\n"

        SKU_SET = self.entries[int(self.id) - 1]['SKU_SET']
        dcw_str, SKUNickName = self.extract_dcw_entries(SKU_SET)
        
        dcw_structure = self.text_to_dcw(str(dcw_str))
        
        output_msg += f"{dcw_str}\n"
        output_msg += f"{SKUNickName}\n"

        return SKU_SET, dcw_str, SKUNickName, dcw_structure, output_msg