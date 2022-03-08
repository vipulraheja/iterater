## IteraTeR Crawler

This folder contains the code for data collection and preprocessing.



### 1. Setup Environment
Install required packages under `python==3.8`:
```
pip install -r requirements.txt
```



### 2. Raw Data Collection
Collect raw documents from [arXiv](https://arxiv.org/), [Wikipedia](https://en.wikipedia.org/wiki/Main_Page), and [Wikinews](https://en.wikinews.org/wiki/Main_Page). 


#### 2.1 ArXiv Parser
Collect original documents in arXiv, run parser: 
```
python get_raw_arxiv.py
```


#### 2.2 Wiki & News Parser
Collect original documents in wikipeida and wikinews, run parser:
1. Wiki: 
```
python get_raw_wiki.py --domain wiki --main_cate geography
```
2. News: 
```
python get_raw_wiki.py --domain news --main_cate all
```



### 3. Extract Revision History
Extract revision history of each document, and split the raw corpus into 80% training set, 10% validation set and 10% testing set. Run the following command:
```
python corpus_auto_filter.py --domain arxiv
```



### 4. Extract Edit Actions
1. (Optional) Install LaTex if you don't have one: <https://sourabhbajaj.com/mac-setup/LaTeX/>.
2. Generate latexdiff file of each document pairs in each revision depth using latexdiff, run script: 
```
python corpus_extract_diff.py --domain arxiv
```
3. Parse latexdiff files and get the edit actions for each document:
```
python parse_latexdiff_doc_level.py
```
4. Get the edit actions for each sentence pair from each document:
```
python parse_latexdiff_sentence_level.py
```