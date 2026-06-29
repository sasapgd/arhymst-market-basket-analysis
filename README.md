# From Redundant Association Rules to Product Networks

Reproducible code for the study **From Redundant Association Rules to Product Networks: A Scalable Approach Using Hypergraph-Inspired Rule Reduction and Maximum Spanning Trees**.

The workflow mines association rules with Apriori, removes rules that do not provide sufficient improvement over simpler antecedents, projects the retained rules into a weighted product graph, and extracts a Maximum Spanning Tree (MaxST). The repository also contains the sensitivity, edge-weighting, segment, statistical-validation, and correlation-based comparison analyses reported with the study.

## Data

The repository includes an anonymized one-million-row sample in `Data/`. Because `Data/` is the default input directory, the baseline pipeline can be started immediately after cloning the repository. The sample is intended for technical validation of the workflow; its rule counts, network structure, and quantitative outputs are not expected to match the manuscript results.

The full anonymized transaction dataset used for the reported results is available from Zenodo:

[https://doi.org/10.5281/zenodo.20788608](https://doi.org/10.5281/zenodo.20788608)

To reproduce the manuscript results, replace the sample file inside `Data/` with the transaction files from the Zenodo archive. Keep the directory name and file structure unchanged:

```text
arhymst-market-basket-analysis/
├── Data/
│   ├── transactions_part_01.csv
│   ├── transactions_part_02.csv
│   └── ...
├── apriori.R
└── ...
```

Keep the directory name `Data/` unchanged. The scripts read files matching `Data/transactions*.csv` by default. Do not keep the sample and full files together when reproducing the manuscript, because every matching file is read as part of the same dataset.

The main analysis requires these columns:

- `PERSON_PUBLIC_KEY`
- `DATE`
- `CHANNEL`
- `PRODUCT_CATEGORY`

A basket is defined by `PERSON_PUBLIC_KEY`, `DATE`, and `CHANNEL`. The segment analysis additionally requires `GENDER` and either `AGE_GROUP` or numeric `AGE`.

## Software requirements

The analysis was verified with:

- Python 3.13.1
- R 4.5.3
- R packages `data.table` 1.18.4 and `arules` 1.7-14

Create a Python environment and install the required packages:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS.

Install the R packages:

```r
install.packages(c("data.table", "arules"))
```

All commands below are run from the repository root. Use `python` and `Rscript` commands that point to the prepared environments.

## Baseline pipeline

The manuscript baseline uses support `0.001`, confidence `0.30`, minimum lift `1`, `maxlen=3`, and confidence-improvement threshold `delta=0.05`. Graph edges are weighted by `Lift × Confidence`.

### Default execution without arguments

The baseline pipeline can be run without command-line arguments:

```bash
Rscript apriori.R
python rule_reduction_conf.py
python graph_utils.py
python mst_network_analysis.py
python filtered_graph.py
python post_reduction_network_analysis.py
```

In this mode, the scripts read `Data/transactions*.csv` and use the manuscript baseline: support `0.001`, confidence `0.30`, minimum lift `1`, `maxlen=3`, no Apriori mining-time limit (`maxtime=0`), the standard `arules` redundancy filter, confidence-improvement threshold `delta=0.05`, and `Lift × Confidence` graph weights. Intermediate and final files are written beside the scripts. The included sample verifies execution; after replacing it with the published full dataset, the same commands reproduce the baseline rule set and MaxST.

`apriori.R` accepts `--maxtime=SECONDS`; `0` disables the internal `arules` time limit and is the default. Keeping this limit disabled is important for longer-rule experiments because the package otherwise defaults to five seconds and can stop before the requested `maxlen` is fully evaluated.

By default, `filtered_graph.py` and `post_reduction_network_analysis.py` process only the `100%` graph. To reproduce all filtered graph variants reported with the study, supply `--percentages 20 25 30 100` to both scripts as shown below.

Use `rule_reduction_conf.py` in the no-argument baseline pipeline. The comparison script `rule_reduction.py` also defaults to confidence reduction with `delta=0.05`, but writes `Rules_For_Python_REDUCED_CONFIDENCE.csv`, whereas `graph_utils.py` without arguments expects `Rules_For_Python_REDUCED.csv`.

The auxiliary scripts also have reproducible defaults: sensitivity analysis runs `delta=0.01–0.10` and all seven support/confidence scenarios; edge-weight sensitivity compares `Lift`, `Confidence`, and `Lift × Confidence` on the same baseline reduced rule set; segment analysis uses the baseline parameters; statistical validation uses `maxlen=3–6` and BH-FDR `alpha=0.05`; and the MaxST-variant, MaxST-comparison, and Valle scripts use `maxlen=3–6`. These scripts expect the prerequisite `timing_runs/` outputs described in their respective sections. The Valle comparison reuses its existing pair-count cache unless `--rebuild-pairs` is supplied.

### 1. Mine association rules

```bash
Rscript apriori.R --input-dir=Data --output-dir=timing_runs/maxlen_3 --maxlen=3
```

If `--maxlen` is omitted, the default is `3`. Repeat this command with output directories `maxlen_4`, `maxlen_5`, and `maxlen_6` for the rule-length analysis.

### 2. Apply confidence-improvement reduction

```bash
python rule_reduction_conf.py \
  --input timing_runs/maxlen_3/Rules_For_Python.csv \
  --output timing_runs/maxlen_3/Rules_For_Python_REDUCED_CONFIDENCE.csv \
  --delta 0.05
```

### 3. Project rules into the product graph

```bash
python graph_utils.py \
  --input timing_runs/maxlen_3/Rules_For_Python_REDUCED_CONFIDENCE.csv \
  --output timing_runs/maxlen_3/PRODUCT_GRAPH.csv
```

For every product pair, the strongest retained `Lift × Confidence` edge is used.

### 4. Extract the Maximum Spanning Tree

```bash
python mst_network_analysis.py \
  --input-graph timing_runs/maxlen_3/PRODUCT_GRAPH.csv \
  --output timing_runs/maxlen_3/MST.csv
```

### 5. Generate filtered graphs

```bash
python filtered_graph.py \
  --input-graph timing_runs/maxlen_3/PRODUCT_GRAPH.csv \
  --output-dir timing_runs/maxlen_3 \
  --percentages 20 25 30 100
```

### 6. Generate MaxST centrality and derived network outputs

```bash
python post_reduction_network_analysis.py \
  --product-graph timing_runs/maxlen_3/PRODUCT_GRAPH.csv \
  --mst timing_runs/maxlen_3/MST.csv \
  --filtered-dir timing_runs/maxlen_3 \
  --output-dir timing_runs/maxlen_3 \
  --percentages 20 25 30 100
```

Each stage writes a timing summary and run metadata beside its outputs.

## Reduction-criterion and MaxST comparison

Create confidence-, lift-, and product-based reductions for every `maxlen` directory. For example:

```bash
python rule_reduction.py --input timing_runs/maxlen_3/Rules_For_Python.csv --output timing_runs/maxlen_3/Rules_For_Python_REDUCED_CONFIDENCE.csv --metric confidence --delta 0.05
python rule_reduction.py --input timing_runs/maxlen_3/Rules_For_Python.csv --output timing_runs/maxlen_3/Rules_For_Python_REDUCED_LIFT.csv --metric lift --delta 0.05
python rule_reduction.py --input timing_runs/maxlen_3/Rules_For_Python.csv --output timing_runs/maxlen_3/Rules_For_Python_REDUCED_PRODUCT.csv --metric product --delta 0.05
```

After producing the three reduced files for `maxlen=3–6`, generate all MaxST variants and their comparison summary:

```bash
python generate_mst_variants.py --maxlen 3 4 5 6 --runs-dir timing_runs --output-dir timing_runs/mst_variants
python mst_comparison_analysis.py --maxlen 3 4 5 6 --input-dir timing_runs/mst_variants --output-dir timing_runs/mst_variants
```

## Edge-weight sensitivity of the baseline MaxST

This additional experiment keeps the same baseline reduced rule set and changes only the edge-weighting scheme used for product-pair selection and MaxST extraction. It therefore does not rerun Apriori.

```bash
python mst_weight_sensitivity.py \
  --input timing_runs/maxlen_3/Rules_For_Python_REDUCED_CONFIDENCE.csv \
  --output-dir timing_runs/mst_weight_sensitivity
```

The script compares three weighting schemes:

- `Lift`
- `Confidence`
- `Lift × Confidence` baseline

It writes the three MaxSTs, projected product graphs, an edge-overlap table, and `MST_WEIGHT_SENSITIVITY_SUMMARY.csv`.

## Sensitivity analyses

The following command runs `delta=0.01–0.10` and the seven support/confidence configurations. It reuses `timing_runs/maxlen_3` as the baseline:

```bash
python sensitivity_analysis.py \
  --data-dir Data \
  --baseline-dir timing_runs/maxlen_3 \
  --output-dir sensitivity_experiments
```

On systems where `Rscript` is not at the Windows default used by the script, add `--rscript /path/to/Rscript`.

## Segment analysis

The input files for this analysis must contain `GENDER` and `AGE_GROUP` or `AGE`:

```bash
Rscript segmented_experiments.R \
  --input-dir=Data \
  --output-dir=segmented_experiments \
  --baseline-mst=timing_runs/mst_variants/MST_MAXLEN_3_CONFIDENCE.csv
```

This runs gender, age-group, and combined gender-age analyses using the baseline parameters.

## Fisher exact tests and BH-FDR correction

```bash
python statistical_validation.py \
  --input-dir timing_runs \
  --output-dir statistical_validation \
  --maxlen 3 4 5 6
```

The script reconstructs basket-level 2 × 2 tables from the rule measures, performs one-sided Fisher exact tests, and applies Benjamini-Hochberg correction separately within each `maxlen` set. Basket counts are read from the Apriori metadata.

## Valle-type correlation MST comparison

```bash
python valle_mst_comparison.py \
  --data-dir Data \
  --maxst-dir timing_runs/mst_variants \
  --output-dir valle_mst_comparison \
  --maxlen 3 4 5 6 \
  --rebuild-pairs
```

This builds the correlation-based minimum spanning tree from binary basket-category occurrences and compares it with the `Lift × Confidence` MaxSTs.

## Generate supplementary tables

After completing the analyses above:

```bash
python generate_supplementary_tables.py --base-dir . --output-dir generated_supplementary_tables
```

This generates workbook summaries from the corresponding result CSV files. Additional comparison summaries are written by `mst_comparison_analysis.py` and `mst_weight_sensitivity.py`.

## Citation

If you use the dataset, cite the Zenodo record:

> ARHyMST transaction dataset. Zenodo. https://doi.org/10.5281/zenodo.20788608

Please also cite the accompanying article when its final bibliographic information becomes available.

## License

The source code is distributed under the license in `LICENCE`. Dataset use is governed by the license stated in the Zenodo record.
