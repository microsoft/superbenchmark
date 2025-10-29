
import argparse
import os
import time

from .utils import logger
from .report import Report, ReportStructure


class InParameters:
    """Input parameters of the agent."""
    def __init__(self, data) -> None:
        """Constructor."""
        self.case_id = data.get('case_id', None)
        self.case_folder = data.get('case_folder', None)
        self.generate_prompt = data.get('generate_prompt', None)
        self.pdf_run = data.get('pdf_run', None)
        self.build_run = data.get('build_run', None)
        self.modify_prompt = data.get('modify_prompt', None)
        self.modify_pdf = data.get('modify_pdf', None)
        self.regenerate_one_section = data.get('regenerate_one_section', None)


class OutParameters:
    """Out parameters of the agent."""
    def __init__(self, status, build_sku, local_save_path, azure_save_path):
        """Constructor."""
        self.build_status = status
        self.build_sku = build_sku
        self.save_path_local = local_save_path
        self.save_path_azure = azure_save_path


# Define a class inherited from AgentApiServer and implement 2 abstraction functions:
#   - perform_operation
#   - build_in_parameters

class ReportGen:
    """ReportGen business logic class."""
    def __init__(self, args):
        """Constructor."""
        self._args = args
        self.report = Report()  # Initialize Report object here
        # internal used state control word, no need to let user to set them directly
        self.sku_set = None
        self.dcw_str = None
        self.SKUNickName = None
        self.dcw_structure = None
        self.workload = None
        self.report_structure_dbg = None
        # by default, do not disable gpt
        self.disable_gpt = os.environ.get('REPORTGEN_DISABLE_GPT', 'False') == 'True'
        self.report_content_dict_list = []

    
    def perform_operation(self, in_parameters):
        """Call azure openai and infra controller."""

        # Main code:
        self._args = in_parameters
        logger.info(f'[internal control word] infrawise-report-gen started')
        
        # output
        msg_status = "general"
        build_sku = None
        msg_local_save_path = None
        msg_azure_save_path = None
        
        # get operation name
        this_operation = self.get_operation_name()
        start_time = time.time()
        
        # get report config
        if self._args.case_folder is not None:
            # building a new case
            self.sku_set, self.SKUNickName, self.dcw_structure, self.workload, msg = self.report.create_new_case(self._args.case_folder)
            logger.info(f'config build new case {self.sku_set}')
            # define report structure
            self.report_structure_dbg = self.report.define_report_structure('benchmark_report', self.sku_set, self.workload)
            logger.info(f'define report structure')

            # output
            msg_status = msg
            build_sku = self.sku_set
            msg_local_save_path = None
            msg_azure_save_path = None
            
            # report build run
            external_pdf_file, azure_path, self.report_content_dict_list = self.report.build_run_with_timeout(self.report_structure_dbg, 1, self.dcw_structure, self.SKUNickName, 'latex', self.disable_gpt, self.sku_set, 'benchmark_report')
            logger.info('Full report build finished')

            # output
            if external_pdf_file is not None:
                msg_status = msg_status + f" Full report build finished, pdf saved"
            build_sku = self.sku_set
            msg_local_save_path = external_pdf_file
            msg_azure_save_path = azure_path
            
            # Reset the case_folder parameter to None
            self._args.case_folder = None
            logger.info('Parameter case_folder has been reset to None')
            self._args.build_run = False
            logger.info('Parameter build_run has been reset to False')

        elif self._args.case_id is not None:
            # building an existing case
            build_id = self._args.case_id
            self.sku_set, self.SKUNickName, self.dcw_structure, msg = self.report.get_sku_set(build_id)
            logger.info(f'config build existing instance {build_id}')
            # define report structure
            self.report_structure_dbg = self.report.define_report_structure('benchmark_report', self.sku_set, None)
            logger.info(f'define report structure')

            # output
            msg_status = f"config build existing instance {build_id}"
            build_sku = self.sku_set
            msg_local_save_path = None
            msg_azure_save_path = None
            
            # report build run
            external_pdf_file, azure_path, self.report_content_dict_list = self.report.build_run_with_timeout(self.report_structure_dbg, 1, self.dcw_structure, self.SKUNickName, 'latex', self.disable_gpt, self.sku_set, 'benchmark_report')
            logger.info('Full report build finished')

            # output
            if external_pdf_file is not None:
                msg_status = msg_status + f" Full report build finished, pdf saved"
            build_sku = self.sku_set
            msg_local_save_path = external_pdf_file
            msg_azure_save_path = azure_path
            
            # Reset the case_id parameter to None
            self._args.case_id = None
            logger.info('Parameter case_id has been reset to None')
            self._args.build_run = False
            logger.info('Parameter build_run has been reset to False')
        
        elif self._args.pdf_run:
            # pdf run
            external_pdf_file, azure_path, self.report_content_dict_list = self.report.pdf_run(self.report_structure_dbg, 1, self.dcw_structure, self.SKUNickName, self.sku_set, self.disable_gpt)
            logger.info('pdf run finished')

            # Reset the build_run parameter to False
            self._args.pdf_run = False
            logger.info('Parameter pdf_run has been reset to False')

            # output
            if external_pdf_file is not None:
                msg_status = f"Full report build finished, pdf saved"
            else:
                msg_status = f"Build error, pdf save skipped"
            build_sku = self.sku_set
            msg_local_save_path = external_pdf_file
            msg_azure_save_path = azure_path

        elif self._args.build_run:
            # report build run
            external_pdf_file, azure_path, self.report_content_dict_list = self.report.build_run_with_timeout(self.report_structure_dbg, 1, self.dcw_structure, self.SKUNickName, 'latex', self.disable_gpt, self.sku_set, 'benchmark_report')
            logger.info('Full report build finished')

            # Reset the build_run parameter to False
            self._args.build_run = False
            logger.info('Parameter build_run has been reset to False')

            # output
            if external_pdf_file is not None:
                msg_status = f"Full report build finished, pdf saved"
            else:
                msg_status = f"Build error, pdf save skipped"
            build_sku = self.sku_set
            msg_local_save_path = external_pdf_file
            msg_azure_save_path = azure_path
        
        elif self._args.generate_prompt:
            # report prompt generation
            self.report.generate_prompt(self.report_structure_dbg, 1, self.dcw_structure, self.SKUNickName, 'latex', self.disable_gpt, self.sku_set, 'benchmark_report')
            logger.info('Prompt generation finished')

            # Reset the build_run parameter to False
            self._args.generate_prompt = False
            logger.info('Parameter generate_prompt has been reset to False')

            # output

            msg_status = f"Prompt generation"
            build_sku = self.sku_set
            msg_local_save_path = None
            msg_azure_save_path = None
            
        elif self._args.modify_prompt is not None:
            # modify contents (via modifying the prompts)
            logger.info('modify content via chaing the prompt')
            if self.dcw_structure is not None:
                skip_pdf_latex, output = self.report.modify_prompt(self.dcw_structure, self._args.modify_prompt, self.report_content_dict_list)
                if not skip_pdf_latex:
                    external_pdf_file, azure_path, self.report_content_dict_list = self.report.pdf_run(self.report_structure_dbg, 1, self.dcw_structure, self.SKUNickName, self.sku_set, self.disable_gpt)
                    output += f', pdf regenerated.'
                else:
                    output = f'pdf regeneration skipped, the section you ask does not exist.'
                    external_pdf_file = None
                    azure_path = None
            else:
                output = f'there is no valid build input, please provide input'
                external_pdf_file = None
                azure_path = None
                

            # Reset the modify_prompt parameter to None
            self._args.modify_prompt = None
            logger.info('Parameter modify_prompt has been reset to None')

            # output
            msg_status = f"{output}"
            build_sku = self.sku_set
            msg_local_save_path = external_pdf_file
            msg_azure_save_path = azure_path

        elif self._args.regenerate_one_section is not None:
            # rebuild one section
            logger.info('rebuild one section')
            if self.dcw_structure is not None:
                skip_pdf_latex, output = self.report.regenerate_one_section(self.dcw_structure, self._args.regenerate_one_section, self.report_content_dict_list)
                if not skip_pdf_latex:
                    external_pdf_file, azure_path, self.report_content_dict_list = self.report.pdf_run(self.report_structure_dbg, 1, self.dcw_structure, self.SKUNickName, self.sku_set, self.disable_gpt)
                    output += f'pdf regenerated.'
                else:
                    output = f'pdf regeneration skipped, the section you ask does not exist.'
                    external_pdf_file = None
                    azure_path = None
            else:
                output = f'there is no valid build input, please provide input'
                external_pdf_file = None
                azure_path = None

            # Reset the rebuild one section parameter to None
            self._args.regenerate_one_section = None
            logger.info('Parameter regenerate_one_section has been reset to None')

            # output
            msg_status = f"{output}"
            build_sku = self.sku_set
            msg_local_save_path = external_pdf_file
            msg_azure_save_path = azure_path

        elif self._args.modify_pdf is not None:
            # modify contents (via modifying the latex/pdf file)
            logger.info('modify content via chaing the latex/pdf file')
            if self.dcw_structure is not None:
                skip_pdf_latex, output = self.report.modify_latex(self.dcw_structure, self._args.modify_pdf, self.report_content_dict_list)
                if not skip_pdf_latex:
                    external_pdf_file, azure_path, self.report_content_dict_list = self.report.pdf_run(self.report_structure_dbg, 1, self.dcw_structure, self.SKUNickName, self.sku_set, self.disable_gpt)
                    output += f', pdf regenerated.'
                else:
                    external_pdf_file = None
                    output = f'pdf regeneration skipped, the section you ask does not exist.'
                    azure_path = None
            else:
                output = f'there is no valid build input, please provide input'
                external_pdf_file = None
                azure_path = None
            logger.info('pdflatex run finished')
            
            # Reset the modify_pdf parameter to None
            self._args.modify_pdf = None
            logger.info('Parameter modify_pdf has been reset to None')

            # output
            msg_status = f"{output}"
            build_sku = self.sku_set
            msg_local_save_path = external_pdf_file
            msg_azure_save_path = azure_path
                
        logger.info(f'[internal control word] self.sku_set value is {self.sku_set}')
        logger.info(f'[internal control word] self.workload value is {self.workload}')
        logger.info(f'[internal control word] self.report_structure_dbg value is {self.report_structure_dbg}')
        out_parameters = OutParameters(msg_status, build_sku, msg_local_save_path, msg_azure_save_path)
        return out_parameters

    def build_in_parameters(self, data):
        """Build input parameters."""
        in_parameteters = InParameters(data["data"])
        return in_parameteters

    def get_operation_name(self):
        """Get the current operation name, This should exactly follow the branch definitions of function perform_operation"""
        if self._args.case_folder is not None:
            return "report-gen-full-build"
        elif self._args.case_id is not None:
            return "report-gen-full-build"
        elif self._args.pdf_run:
            return "report-gen-pdf-rebuild"
        elif self._args.build_run:
            return "report-gen-full-build"
        elif self._args.generate_prompt:
            return "report-gen-generate-prompt"   
        elif self._args.modify_prompt is not None:
            return "report-gen-modify-prompt"   
        elif self._args.regenerate_one_section is not None:
            return "report-gen-section-build"
        elif self._args.modify_pdf is not None:
            return "report-gen-pdf-rebuild"
        else:
            return "report-gen-other"


def parse_args():
    valid_sections = ReportStructure.get_valid_sections()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-id', type=int, help='Index of existing builds, integer', default=None)
    parser.add_argument('--case-folder', type=str, help='Please specify the new report case folder path', default=None)
    parser.add_argument('--pdf-run', type=bool, help='Bool value, set to True will trigger a pdf latex build', default=False)
    parser.add_argument('--build-run', type=bool, help='Bool value, set to True will trigger a full report generation', default=False)
    parser.add_argument('--generate_prompt', type=bool, help='Bool value, set to True will trigger a prompt generation', default=False)
    parser.add_argument('--modify-prompt', type=str, choices=valid_sections, help='Please specify the section name of whose prompt you would like to modify, and specify how you want to modify it', default=None)
    parser.add_argument('--modify-pdf', type=str, choices=valid_sections, help='Please specify the section name of whose report/pdf/latex you would like to modify, and specify how you want to modify it', default=None)
    parser.add_argument('--regenerate-one-section', type=str, choices=valid_sections, help='Please specify the section name of which you want to rebuild', default=None)
    
    
    args = parser.parse_args()
    return args
