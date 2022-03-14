# IteraTeR: Understanding Iterative Revision from Human-Written Text

This repository provides datasets and code for preprocessing, training and testing models for Iterative Text Revision with the official Hugging Face implementation of the following paper:

> [Understanding Iterative Revision from Human-Written Text](https://arxiv.org/abs/2203.03802) <br>
> [Wanyu Du](https://github.com/wyu-du), [Vipul Raheja](https://github.com/vipulraheja), [Dhruv Kumar](https://github.com/ddhruvkr), [Zae Myung Kim](https://github.com/zaemyung), [Melissa Lopez](https://github.com/mlsabthlpz) and [Dongyeop Kang](https://github.com/dykang) <br>
> [ACL 2022](https://www.2022.aclweb.org/) <br>

It is mainly based on `transformers`.


## Installation
The following command installs all necessary packages:
```
pip install -r requirements.txt
```
The project was tested using Python 3.7.


## HuggingFace Integration
We uploaded both our datasets and model checkpoints to Hugging Face. You can directly load our data using `datasets` and load our model using `transformers`.
```python
# load our dataset
from datasets import load_dataset
dataset = load_dataset("wanyu/IteraTeR_human_sent")

# load our model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("wanyu/IteraTeR-PEGASUS")
model = AutoModelForSeq2SeqLM.from_pretrained("wanyu/IteraTeR-PEGASUS")
```

We also provided a [demo code](https://colab.research.google.com/drive/1qv7b2jJSqqMaYOQ5NRvAvoyDB3gvpwcp?usp=sharing) for how to use them to do iterative text revision.



## Datasets
<table>
	<tr>
		<th></th>
		<th colspan='3'>Document-level</th>
		<th colspan='3'>Sentence-level</th>
	</tr>
	<tr>
		<th>Dataset</th>
		<th>Train</th>
		<th>Dev</th>
		<th>Test</th>
		<th>Train</th>
		<th>Dev</th>
		<th>Test</th>
	</tr>
	<tr>
		<td>IteraTeR-FULL</td>
		<td>29848</td>
		<td>856</td>
		<td>927</td>
		<td>157579</td>
		<td>19705</td>
		<td>19703</td>
	</tr>
	<tr>
		<td>IteraTeR-HUMAN</td>
		<td>481</td>
		<td>27</td>
		<td>51</td>
		<td>3254</td>
		<td>400</td>
		<td>364</td>
	</tr>
</table>

All data and detailed description for the data structure can be found under [datasets/](https://github.com/vipulraheja/IteraTeR/tree/main/dataset). 

Code for collecting the revision history data can be found under [code/crawler/](https://github.com/vipulraheja/IteraTeR/tree/main/code/crawler). 



## Models

### Model checkpoints

| Model         | Dataset        |  SARI  |  BLEU  | ROUGE-L|  Avg.  |
| :-------------|:-------------  | :-----:| :-----:| :-----:| :-----:|
| [BART]()      | IteraTeR-FULL  | 37.28  | 77.50  | 86.14  | 66.97  |
| [PEGASUS]()   | IteraTeR-FULL  | 37.11  | 77.60  | 86.84  | 67.18  |


### Train model
To train your model, simply run:
```
bash train.sh
```


### Model inference
To run your model on the test set, use the following command:
```
bash generate.sh
```


## Citation
If you find this work useful for your research, please cite our paper:

#### Understanding Iterative Revision from Human-Written Text
```
@inproceedings{du2022iterater,
    title = "Understanding Iterative Revision from Human-Written Text",
    author = "Du, Wanyu and Raheja, Vipul and Kumar, Dhruv and Kim, Zae Myung and Lopez, Melissa and Kang, Dongyeop",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics",
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```

