Your task is to understand the user's question, and assign a type to it.

Three question types are supported:

1. [evaluation]: it means the user wants to assess or query or ask the performance or features of these server designs based on a certain set of decision criteria. Usually, questions of this type can be the performance comparison of two sku/server, or performance evaluation of one sku/sever.
2. [analysis]: it means the user want to analyze the benchmark data from the session before.
3. [others]: it means user's prompt is neither [evaluation] nor [summary]. Note that the above types have specific definition. If user question is not belong to [evaluation] nor [summary], it should be [others]. 


The followings are some examples:
[Example 1]
User question:
The newly designed server SKU uses the 2 DIMMs per channel memory configuration.   
The reference server SKU uses the 1 DIMM per channel memory configuration.   
We need to evaluate these two server designs based on decision making criteria of maximum-performance.    
1. [evaluation]

[Example 2]
User question: 
Now，please analyze the above results.
Can you perform an analysis on the benchmark data.
2. [analyze]

[Example 3]
User question: 
what is the weather  
3. others

[Example 4]
User question: 
write/create a new benchmark for ...
3. others

Please output  in the following format:
Type: [1-3]. [question type]