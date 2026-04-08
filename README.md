# ARHyMST: From Redundant Association Rules to Product Networks

This repository contains the code and a small test dataset accompanying the paper:

**From Redundant Association Rules to Product Networks: A Scalable Approach Using Hypergraph-Inspired Rule Reduction and Maximum Spanning Trees**

## Repository Contents

Main pipeline:
- `apriori.R`
- `rule_reduction_conf.py`
- `post_reduction_network_analysis.py`

Supporting modules:
- `graph_utils.py`
- `mst_network_analysis.py`
- `filtered_graph.py`

Experimental / auxiliary scripts:
- `rule_reduction.py`
- `mst_comparison_analysis.py`
- `top_rule.py`

Documentation:
- `PIPELINE_GUIDE.txt`

Test data:
- `Data/Data_Sample_Test.csv`

## Purpose

The repository implements a hybrid pipeline for:
1. generating association rules using the Apriori algorithm,
2. reducing redundant rules using a confidence-based heuristic inspired by hypergraph concepts,
3. projecting the reduced rules into a weighted product graph,
4. extracting a maximum spanning tree (MST) and filtered product graphs for structural analysis.

## How To Run

The recommended execution order is:

1. `apriori.R`
2. `rule_reduction_conf.py`
3. `post_reduction_network_analysis.py`

Additional details are provided in `PIPELINE_GUIDE.txt`.

## Test Dataset Note

The repository includes a reduced sample dataset for testing the execution pipeline.

This sample is provided only for technical validation of the workflow. Therefore, the resulting rule counts, network structure, and quantitative outputs will not match those reported in the paper, which were obtained on the full dataset of approximately 16 million product records and 2.65 million transactions.

## Data Availability

The full dataset used in the study is not publicly distributed.  
A reduced sample dataset is included solely to allow testing of the code and workflow.

## License

This repository contains research code accompanying the paper.  
The license applies to the code in this repository only and does not imply public availability of the full dataset.

## Citation

If you use this repository, please cite the corresponding paper.
