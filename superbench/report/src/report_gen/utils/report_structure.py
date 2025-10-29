
import json
from enum import Enum


class SectionCategory(Enum):
    # takes 1 input from prompt.case folder, use python to convert:
    ## title, author, date, appendix
    
    # takes 2 inputs from prompt.case_target_baseline folder and prompt.report folder, use LLM to convert:
    ## benchmark
    
    # takes 2 inputs from prompt.report and output of benchmark, use LLM to convert:
    ## es, conclusion
    
    # spacer is only used to combine into the final main tex
    
    title = "title"
    author = "author"
    date = "date"
    es = "es"
    benchmark = "benchmark"
    conclusion = "conclusion"
    spacer = "spacer"
    appendix = "appendix"
    table = "table"
    manual = "manual"
    chapter = "chapter"
  
class ReportStructure:
    def __init__(self):
        self.sections = {
            'title': self.create_section('title', 'title', SectionCategory.title),
            'author': self.create_section('author', 'author', SectionCategory.author),
            'date': self.create_section('date', 'date', SectionCategory.date),
            'abstract': self.create_section('abstract', 'abstract', SectionCategory.manual),
            'background': self.create_section('background', 'background', SectionCategory.manual),
            'es': self.create_section('es', 'es', SectionCategory.manual),
            'summary': self.create_section('summary', 'summary', SectionCategory.es),
            'basic': self.create_section('basic', 'basic', SectionCategory.benchmark),
            'basicmach': self.create_section('basicmach', 'basicmach', SectionCategory.benchmark),
            'efficiency_sim': self.create_section('efficiency_sim', 'efficiency_sim', SectionCategory.manual),
            'gemm': self.create_section('gemm', 'gemm', SectionCategory.benchmark),
            'gemmfp8': self.create_section('gemmfp8', 'gemmfp8', SectionCategory.benchmark),
            'gemmfp4': self.create_section('gemmfp4', 'gemmfp4', SectionCategory.benchmark),
            'communication': self.create_section('communication', 'communication', SectionCategory.benchmark),
            'communication_nomsccl': self.create_section('communication_nomsccl', 'communication_nomsccl', SectionCategory.benchmark),
            'streambw': self.create_section('streambw', 'streambw', SectionCategory.benchmark),
            'superbench': self.create_section('superbench', 'superbench', SectionCategory.benchmark),
            'training': self.create_section('training', 'training', SectionCategory.benchmark),
            'inference': self.create_section('inference', 'inference', SectionCategory.benchmark),
            'inferencemach': self.create_section('inferencemach', 'inferencemach', SectionCategory.benchmark),
            'inference_sim': self.create_section('inference_sim', 'inference_sim', SectionCategory.benchmark),
            'llama2_sim': self.create_section('llama2_sim', 'llama2_sim', SectionCategory.benchmark),
            'multinode_ibwrite': self.create_section('multinode_ibwrite', 'multinode_ibwrite', SectionCategory.benchmark),
            'multinode_communication': self.create_section('multinode_communication', 'multinode_communication', SectionCategory.benchmark),
            'multinode_superbench': self.create_section('multinode_superbench', 'multinode_superbench', SectionCategory.benchmark),
            'multinode_inference': self.create_section('multinode_inference', 'multinode_inference', SectionCategory.benchmark),
            'multinode_sb_scale': self.create_section('multinode_sb_scale', 'multinode_sb_scale', SectionCategory.benchmark),
            'multinode_inf_scale': self.create_section('multinode_inf_scale', 'multinode_inf_scale', SectionCategory.benchmark),
            'accuracy': self.create_section('accuracy', 'accuracy', SectionCategory.manual),
            'functionality': self.create_section('functionality', 'functionality', SectionCategory.manual),
            'conclusion': self.create_section('conclusion', 'conclusion', SectionCategory.conclusion),
            'spacer': self.create_section('spacer', 'spacer', SectionCategory.spacer),
            'projected': self.create_section('projected', 'projected', SectionCategory.benchmark),
            'peak': self.create_section('peak', 'peak', SectionCategory.benchmark),
            'spec_comparison': self.create_section('spec_comparison', 'spec_comparison', SectionCategory.benchmark),
            'method': self.create_section('method', 'method', SectionCategory.appendix),
            'table': self.create_section('table', 'table', SectionCategory.table),
            'chapter': self.create_section('chapter', 'chapter', SectionCategory.chapter),
        }

    def create_section(self, sec_id, sec_shortname, sec_category = SectionCategory.benchmark):
        section = {
            "section_id": sec_id,                   # the index in integer, in the order of reading the report
            "sec_shortname": sec_shortname,         # short name, user defined
            "sec_category": str(sec_category.value),# category of this section, refer to class SectionCategory for available values
            "comment": "",                          # only for debug
            "content": ""                           # the content of this section, generated by InfraWise
        } 
        return section

    def get_report_structure(self, report_class, sku_set, workload_str):
        if report_class == 'benchmark_report':
            return self.get_benchmark_report_structure(sku_set, workload_str)
        elif report_class == 'simulation_report':
            return self.get_simulation_report_structure(sku_set, workload_str)
        else:
            raise ValueError('Unsupported report class')

    def get_benchmark_report_structure(self, sku_set, workload_str):
        # define a full section set
        sections_pt1 = ['title', 'author', 'date']
        sections_pt2 = ['abstract', 'background', 'summary', 'basic', 'gemm', 'gemmfp8', 'gemmfp4', 'communication', 'communication_nomsccl', 'streambw', 'superbench', 'training', 'inference', 'multinode_ibwrite', 'multinode_communication', 'multinode_superbench', 'multinode_inference', 'conclusion', 'spacer', 'projected', 'peak', 'method', 'chapter']
        sections_pt3 = ['table']
        all_sections = sections_pt1 + sections_pt2 + sections_pt3
        
        # predefine dbg section set
        sku_set_sections = {
            'MI300xH100': ['title', 'author', 'date', 'abstract', 'background', 'summary', 'basic', 'gemm', 'communication', 'superbench', 'training', 'inference', 'multinode_ibwrite', 'multinode_communication', 'multinode_superbench', 'multinode_inference', 'conclusion', 'spacer', 'projected', 'method', 'table'],
            'VMBM': ['basic'],
            'MI300xCanaryH100March': ['title', 'author', 'date', 'summary', 'basic', 'gemm', 'communication', 'superbench', 'training', 'inference', 'multinode_ibwrite', 'multinode_communication', 'multinode_superbench', 'multinode_sb_scale', 'multinode_inference', 'conclusion', 'spacer', 'projected', 'method', 'table'],
            'MI300xCanaryH100': ['title', 'author', 'date', 'multinode_ibwrite', 'multinode_communication', 'multinode_superbench', 'multinode_inference', 'spacer', 'table'],
            'H200H100': ['title', 'author', 'date', 'abstract', 'background', 'summary', 'basic', 'gemm', 'communication', 'superbench', 'training', 'inference', 'conclusion', 'spacer', 'peak', 'method', 'table'],
            'H200MI300xbkc2405': ['title', 'author', 'date', 'abstract', 'background', 'summary', 'basic', 'gemm', 'communication', 'superbench', 'training', 'inference', 'conclusion', 'spacer', 'peak', 'method', 'table'],
            'H200PerfMode': ['title', 'author', 'date', 'abstract', 'background', 'summary', 'basic', 'gemm', 'communication', 'superbench', 'training', 'inference', 'conclusion', 'spacer', 'peak', 'method', 'table'],
            'H200MI300xbkc2406opt': ['title', 'author', 'date', 'abstract', 'background', 'summary', 'basic', 'gemm', 'communication', 'superbench', 'training', 'inference', 'conclusion', 'spacer', 'peak', 'method', 'table'],
            'H200MI300xtop': ['title', 'author', 'date', 'background', 'superbench', 'training', 'inference', 'table'],
            'MACHFPGAA100H100': ['title', 'author', 'date', 'es', 'summary', 'background', 'basicmach', 'inferencemach', 'accuracy', 'functionality', 'conclusion', 'spacer', 'table'],
        }
        print(f'workload_str is {workload_str}')

        if workload_str is not None:
            sections_pt2 = self.convert_workload_list(workload_str, all_sections)
            dbg_sections = sections_pt1 + sections_pt2 + sections_pt3
            print(f"{sections_pt2}")
        elif sku_set in sku_set_sections.keys():
            dbg_sections = sku_set_sections.get(sku_set)
        else:
            dbg_sections = all_sections
        # 1st return the list of all possible sections
        # create a new list of sections to return
        new_sections = []
        for index, section in enumerate(dbg_sections):
            if section in self.sections:
                name = self.sections[section]['sec_shortname']
                cat = SectionCategory(self.sections[section]['sec_category'])
            else:
                name = section
                cat = SectionCategory(SectionCategory('manual'))                         
            new_section = self.create_section(index, name, cat)
            new_sections.append(new_section)

        return new_sections
    def get_simulation_report_structure(self, sku_set, workload_str):
        # define a full section set
        sections_pt1 = ['title', 'author', 'date']
        sections_pt2 = ['abstract', 'summary', 'efficiency_sim', 'spec_comparison', 'inference_sim', 'llama2_sim', 'conclusion', 'spacer']
        sections_pt3 = ['table']
        all_sections = sections_pt1 + sections_pt2 + sections_pt3
        
        # predefine dbg section set
        sku_set_sections = {
            'ROCMSIMCUDASIM': ['title', 'author', 'date', 'abstract', 'summary', 'efficiency_sim', 'spec_comparison', 'inference_sim', 'llama2_sim', 'conclusion', 'spacer', 'table'],
        }
        if workload_str is not None:
            sections_pt2 = self.convert_workload_list(workload_str, all_sections)
            dbg_sections = sections_pt1 + sections_pt2 + sections_pt3
        elif sku_set in sku_set_sections.keys():
            dbg_sections = sku_set_sections.get(sku_set)
        else:
            dbg_sections = all_sections

        # 1st return the list of all possible sections
        # create a new list of sections to return
        new_sections = []
        for index, section in enumerate(dbg_sections):
            if section in self.sections:
                name = self.sections[section]['sec_shortname']
                cat = SectionCategory(self.sections[section]['sec_category'])
            else:
                name = section
                cat = SectionCategory(SectionCategory('manual'))                         
            new_section = self.create_section(index, name, cat)
            new_sections.append(new_section)
        return new_sections
    
    def convert_workload_list(self, workload_str, all_sections):
        workload_list = workload_str.replace(' ', '').split(',')
        for workload in workload_list:
            if workload not in all_sections:
                self.sections[workload] = self.create_section(workload, workload, SectionCategory.manual)
                print(f'adding new section to the structure {workload}')
        workload_list = [workload for workload in workload_list]
        return workload_list
        
    @staticmethod
    def get_valid_sections():
        return ReportStructure().sections.keys()
    

