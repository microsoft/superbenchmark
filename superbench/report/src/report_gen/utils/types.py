from langchain_core.output_parsers import PydanticOutputParser  # type: ignore
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Union, Optional, Tuple


class Benchmark(BaseModel):
    name: str = Field(..., description="benchmark name")
    parameters: Union[str, Dict] = Field('default', description="benchmark parameters")
    reason: Optional[str] = Field(..., description="benchmark reason")
    target: Optional[str] = Field(..., description="benchmark target")
    hardware: str = Field(..., description="benchmark hardware")
    
    class Config:
        extra = "allow"  # Allow arbitrary fields

class BList(BaseModel):
    benchmarks: List[Benchmark] = Field(..., description="benchmark list")
    additional_benchmark_requirement: Optional[List[Dict]] = Field(
        description="additional benchmark requirement", default=None)


Default_blist_parser = PydanticOutputParser(pydantic_object=BList)

class HardwareSpec(BaseModel):
    SKU: str = ""
    GPU: str
    GPU_vendor: str 
    Number_of_GPUs: str = ""
    GPU_Memory: str = ""
    GPU_Memory_Bandwidth: str = ""
    Interconnect: Dict[str, str] = {}
    Tensor_Core_Performance: Dict[str, str] = {}
    Infiniband_Interconnect: str = ""
    CPU: str = ""
    Disk: str = ""
    Setting: str = ""
    
    class Config:
        extra = "allow"  # Allow arbitrary fields

class SystemData(BaseModel):
    NewHardwareSpec:  Optional[HardwareSpec] 
    BaselineHardwareSpec: Optional[HardwareSpec] 
    NewFeature: Optional[str] = Field(
        default=None, description="New feature or new optimization of the new hardware")

class AdditionalData(BaseModel):
    NonePerfData: Optional[str] = Field(..., description="Additional none-performance data for analysis")
    SystemData: SystemData # = Field(..., description="Hardware specs")

class Design(BaseModel):
    Target: str = Field(..., description="Target design")
    Baseline: Optional[str] = Field(default=None, description="Base design")

class DCW(BaseModel):
    Design: Design
    Criterion: str = Field(..., description="Evaluation Criterion")
    Workload: str = Field(..., description="Workload")
    #AdditionalData: Optional[AdditionalData] #=Field(..., description="addtional data")


Default_dcw_parser = PydanticOutputParser(pydantic_object=DCW)



# class BenDioInput(BaseModel):

#     NewHardware: str = Field(
#         ...,
#         description="The new GPU hardware or architecture or SKU to evaluate")
#     BaselineHardware: str = Field(
#         ...,
#         description=
#         "The baseline GPU hardware or architecture or SKU to compare with")
#     NewHardwareSpec: str = Field(
#         ...,
#         description=
#         "The specification of the new hardware, including the computation flops, DRAM size, DRAM banwidth, PCIe bandwidth, networking bandwidth, etc."
#     )
#     BaselineHardwareSpec: str = Field(
#         ..., description="The specification of the baseline hardware")
#     NewFeature: str = Field(
#         ..., description="New feature or new optimization of the new hardware")
#     Criteria: str = Field(
#         ..., description="The criteria to evaluate the new hardware")
#     Workload: str = Field(
#         ...,
#         description=
#         "The workload list need to consider when evaluating the new hardware")

#     # def __init__(self, design:str, criteria: str, workload: str):
#     #     self.design = design
#     #     self.criteria = criteria
#     #     self.workload = workload


# Default_input_parser = PydanticOutputParser(pydantic_object=BenDioInput)
