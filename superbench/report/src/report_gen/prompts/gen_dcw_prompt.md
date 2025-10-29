Your task is to interpret the input user's question, and retrive three key components via interpretation of user's question.

these three key components are:
1. Design: 
The design describes what testbed to evaluate. This is either provided by the user, or you can know it by understanding the design difference between the target design and the baseline design.
the design is usually a virtual machine (VM), a virtual machine with a specific feature, or a GPU type.
some commonly used VM SKU names are NDv4, NDmv4, NDv4_MI200, NDv5, NCv3.
some commonly used GPU types are A100, H100, MI200, MI300x.
2. Workload:
The workload describes what types of benchmarks, models, algorithms, performance measurement tools user want to run on the design testbed. Usually user will provide a specific workload to run, please use this. If user does not provide a specific workload, you need to make suggestion. In the case that target design exists, baseline design does not exist, use 'default' value for workload.
In the case that both target design and baseline design exist, you need to understand which sub system of the server is related to the difference of the two designs, and suggest a proper workload to focus benchmarking this sub system's performance.
3. Criterion:
Criterion is short for evaluation criterion, it describes what principle user wants to use to assess the performance, effectiveness or quality between the baseline and target design. 
In the case that target design exists, baseline design does not exist, use 'default' value for criterion.
In the case that both target design and baseline design exist, if user provides a specific criterion, use it, otherwise use 'default'.
some commonly used criterion includes assese both testbed's peak performance, assess both testbed's cost-effectiveness, evaluate within a time duration, etc.

Note: These three compoenents are semantically independent.

Example 1:  
User's question: is there a need or requirement for SR-IOV feature for the new design?
Design: baseline design is without SKU with SR-IOV, target design is SKU with with SR-IOV.
Criterion: default
Workload: virtualization

Example 2:  
User's question: what is the gpu memory bandwidth perforamnce of SKU MM?
Design: baseline design is empty, target design is SKU MM.
Criterion: default
Workload: gpu memory bandwidth

Example 3:
User's question: evaluate H100 gpu gpt-2 training perforamnce in one minute using just one benchmark
Design: baseline design is empty, target design is H100 GPU.
Criterion: one minute, one benchmark
Workload: gpt-2 training


Please output these three key components in the following format:
{  
    "Design": {
        "Target": "",
        "Baseline": ""
    },  
    "Criterion": "",  
    "Workload": ""
}
