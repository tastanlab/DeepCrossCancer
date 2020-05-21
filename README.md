# DeepCrossCancer: a Deep Learning Framework for Discovering Cross-cancer Patients with Similar Transcriptomic Profiles
DeepCrossCancer provides a deep-learning method for discovering cross-cancer patients with similar transcriptomic profiles. It is implemented by deep learning library Keras and Tensorflow-GPU. DeepCrossCancer provides clustering of cancer patients, prediction of cross-cancer patients, and statistical tests to analyze cross-cancer patients in terms of other genomic data. The clustering part is inspired from [DeepType](https://github.com/runpuchen/DeepType) which is a deep learning framework for prediction breast cancer subgroups.
# Installation
•	Download DeepCrossCancer by
```
git clone https://github.com/dyguay/DeepCrossCancer
```
•	Since the package is written in python 3.5, [python 3.5](https://www.python.org/downloads/) with the pip tool must be installed first. DeepCrossCancer uses the following dependencies: numpy, scipy, pandas, talos, sklearn, keras=2.1.6, tensorflow-gpu=1.12.0, shap, statsmodels. You can install these packages first, by the following commands:
```python
pip install pandas
pip install numpy
pip install scipy
pip install git+https://github.com/autonomio/talos
pip install -U scikit-learn
pip install -v keras==2.1.6
pip install -v tensorflow==1.12.0
pip install shap
pip install statsmodels
```
•	For the visualization, it uses the following dependencies:
```python
pip install matplotlib
pip install seaborn
pip install plotly==3.10.0
```
# For general users who want to perform clustering to find cancer subtypes by our provided model:
```python
python3 model.py params.py --cross_cancer False --num_classes [NUMBER OF CLASSES FOR SUPERVISED PART]
```
For details of other parameters, run:
```python
python3 params.py --help
```
or
```python
python3 params.py -h
```
# Train the model with your own data
First run model.py to train your data and find cross-cancer patients:
```python
python3 model.py params.py --data_dir [YOUR DATA_DIR] --train_file [YOUR TRAIN_FILE] --test_file [YOUR TEST_FILE] --dimension [NUMBER OF FEATURES OF YOUR DATA] --num_classes [NUMBER OF CLASSES FOR SUPERVISED PART]
```
Then, run analysis.py to analyze cross-cancer patients that you found previously, statistically.
```python
python3 analysis.py params.py --train_unnorm_file [YOUR TRAIN_UNNORM_FILE] --test_unnorm_file [YOUR TEST_UNNORM_FILE] --mut_data_dir [YOUR MUT_DATA_DIR] --cnv_data_dir [YOUR CNV_DATA_DIR]
```
For details of other parameters, check params.py
# Input data format
The input data consists of train and test data with normalized format. In total, there are 20,536 columns: 20,531 genes, age, gender, labels (cancer types in our case), survival time, and vital status. Age is in the discrete form. We created buckets for it. The data is originally retrieved from http://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html.
## Other genomics datasets (mutation and CNV) for cross-cancer analysis
### Mutation data:
Mutation annotation files (MAFS) were obtained from the Broad Institute [TCGA GDAC Firehose repository](http://firebrowse.org/api-docs/#!/Analyses/MAF) by using the [R/FirebrowseR package](https://github.com/mariodeng/FirebrowseR).
```R
library(FirebrowseR)
all.Found = F
page.Counter = 2
mut = list()
page.Size = 2000 # using a bigger page size is faster
mut[[1]] = Analyses.Mutation.MAF(format = "csv",
                                             cohort = c("KIRC"), page_size = page.Size,
                                             page = 1, tool = "MutSig2.0")
while(all.Found == F){
  mut[[page.Counter]] = Analyses.Mutation.MAF(format = "csv",
                                             cohort = c("KIRC"), page_size = page.Size,
                                             page = page.Counter, tool = "MutSig2.0")
	names(mut[[page.Counter]]) = names(mut[[1]])
  if(nrow(mut[[page.Counter]]) < page.Size)
    all.Found = T
  else
    page.Counter = page.Counter + 1
}
mut = do.call(rbind, mut)
write.table(mut, file = "kidney_mutations.txt", sep = "\t",
            row.names = FALSE)
```
### CNV data:
Copy number thresholded gene-level data from GISTIC2.0 (last analyze date 20160128) were obtained from the Broad Institute TCGA GDAC Firehose repository by using the [RTCGA-Toolbox R/BioConductor package, version 2.16.2](https://www.bioconductor.org/packages/release/bioc/html/RTCGAToolbox.html).
```R
# Get the last run dates
lastRunDate <- getFirehoseRunningDates()[1]

# Download GISTIC results
lastanalyzedate <- getFirehoseAnalyzeDates(1)
gistic <- getFirehoseData("COAD",gistic2Date = lastanalyzedate, clinical = FALSE, GISTIC = TRUE)

# get GISTIC results
gistic.allbygene <- getData(gistic, type = "GISTIC", platform = "ThresholdedByGene")
object_size(gistic.allbygene)
names(gistic.allbygene) = substr(names(gistic.allbygene), start = 1, stop = 15)
df = gistic.allbygene
for (i in names(df)){
	if(str_sub(i, start= -2)!= "01" && i != "Gene.Symbol" && i != "Locus.ID" && i != "Cytoband"){
		print(i)
		df$i <- NULL}
}

write.table(df, file = "colon_cnv.txt", sep = "\t",
            row.names = FALSE)
```

