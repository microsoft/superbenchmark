[task]
Your task is to userstand which section user wants to modify from the user message, by selecting from available section names:
```text

```
Example 1:
user input = add one more subsection before the subsection evaluation to describe to test tool mlc in the superbench section
available section names = 
```
['title', 'author', 'date', 'abstract', 'background', 'es', 'summary', 'basic', 'basicmach', 'efficiency_sim', 'gemm', 'communication', 'superbench', 'training', 'inference', 'inferencemach', 'inference_sim', 'llama2_sim', 'multinode_ibwrite', 'multinode_communication', 'multinode_superbench', 'multinode_inference', 'multinode_sb_scale', 'multinode_inf_scale', 'accuracy', 'functionality', 'conclusion', 'spacer', 'projected', 'peak', 'spec_comparison', 'method', 'table', 'technology', 'future']
```
output should be
```text
superbench
```


Example 1:
user input = modify the figure to improve the multi-node communication performance expression
available section names = 
```
['title', 'author', 'date', 'abstract', 'background', 'es', 'summary', 'basic', 'basicmach', 'efficiency_sim', 'gemm', 'communication', 'superbench', 'training', 'inference', 'inferencemach', 'inference_sim', 'llama2_sim', 'multinode_ibwrite', 'multinode_communication', 'multinode_superbench', 'multinode_inference', 'multinode_sb_scale', 'multinode_inf_scale', 'accuracy', 'functionality', 'conclusion', 'spacer', 'projected', 'peak', 'spec_comparison', 'method', 'table', 'technology', 'future']
```
output should be
```text
multinode_communication
```