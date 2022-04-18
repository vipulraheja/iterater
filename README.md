# IteraTeR: Understanding Iterative Revision from Human-Written Text

This repository provides datasets and code for preprocessing, training and testing models for Iterative Text Revision with the official Hugging Face implementation of the following paper:

> [Understanding Iterative Revision from Human-Written Text](https://arxiv.org/abs/2203.03802) <br>
> [Wanyu Du](https://github.com/wyu-du), [Vipul Raheja](https://github.com/vipulraheja), [Dhruv Kumar](https://github.com/ddhruvkr), [Zae Myung Kim](https://github.com/zaemyung), [Melissa Lopez](https://github.com/mlsabthlpz) and [Dongyeop Kang](https://github.com/dykang) <br>
> [ACL 2022](https://www.2022.aclweb.org/) <br>

This repository also contains the code and data of the following demo paper:

> [Read, Revise, Repeat: A System Demonstration for Human-in-the-loop Iterative Text Revision](https://arxiv.org/abs/2204.03685) <br>
> [Wanyu Du<sup>1](https://github.com/wyu-du), [Zae Myung Kim<sup>1](https://github.com/zaemyung), [Vipul Raheja](https://github.com/vipulraheja), [Dhruv Kumar](https://github.com/ddhruvkr) and [Dongyeop Kang](https://github.com/dykang) <br>
> [First Workshop on Intelligent and Interactive Writing Assistants (ACL 2022)](https://in2writing.glitch.me/) <br>

[<img src="https://yt-embed.herokuapp.com/embed?v=lK08tIpEoaE" width="50%">](https://www.youtube.com/watch?v=lK08tIpEoaE)


Our code is mainly based on HuggingFace's `transformers` libarary.


## Installation
The following command installs all necessary packages:
```
pip install -r requirements.txt
```
The project was tested using Python 3.7.


## HuggingFace Integration
We uploaded both our datasets and model checkpoints to Hugging Face's [repo](https://huggingface.co/wanyu). You can directly load our data using `datasets` and load our model using `transformers`.
```python
# load our dataset
from datasets import load_dataset
dataset = load_dataset("wanyu/IteraTeR_human_sent")

# load our model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("wanyu/IteraTeR-PEGASUS-Revision-Generator")
model = AutoModelForSeq2SeqLM.from_pretrained("wanyu/IteraTeR-PEGASUS-Revision-Generator")
```

You can change the following data and model specifications:
- <a target="_blank" href="https://huggingface.co/datasets/wanyu/IteraTeR_human_sent">"wanyu/IteraTeR_human_sent"</a>: sentence-level IteraTeR-HUMAN dataset;
- <a target="_blank" href="https://huggingface.co/datasets/wanyu/IteraTeR_human_doc">"wanyu/IteraTeR_human_doc"</a>: document-level IteraTeR-HUMAN dataset;
- <a target="_blank" href="https://huggingface.co/datasets/wanyu/IteraTeR_full_sent">"wanyu/IteraTeR_full_sent"</a>: sentence-level IteraTeR-FULL dataset;
- <a target="_blank" href="https://huggingface.co/datasets/wanyu/IteraTeR_full_doc">"wanyu/IteraTeR_full_doc"</a>: document-level IteraTeR-FULL dataset;
- <a target="_blank" href="https://huggingface.co/datasets/wanyu/IteraTeR_v2">"wanyu/IteraTeR_v2"</a>: sentence-level IteraTeR_v2 dataset;
- <a target="_blank" href="https://huggingface.co/wanyu/IteraTeR-PEGASUS-Revision-Generator">"wanyu/IteraTeR-PEGASUS-Revision-Generator"</a>: PEGASUS model fine-tuned on sentence-level IteraTeR-FULL dataset, see usage example [here](https://huggingface.co/wanyu/IteraTeR-PEGASUS-Revision-Generator#usage);
- <a target="_blank" href="https://huggingface.co/wanyu/IteraTeR-BART-Revision-Generator">"wanyu/IteraTeR-BART-Revision-Generator"</a>: BART model fine-tuned on sentence-level IteraTeR-FULL dataset, see usage example [here](https://huggingface.co/wanyu/IteraTeR-BART-Revision-Generator#usage);


[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qv7b2jJSqqMaYOQ5NRvAvoyDB3gvpwcp?usp=sharing)<br>
We also provided a simple [demo code](https://colab.research.google.com/drive/1qv7b2jJSqqMaYOQ5NRvAvoyDB3gvpwcp?usp=sharing) for how to use them to do iterative text revision. 



## Datasets
You can load our dataset using Hugging Face's `datasets`, and you can also download the raw data in [datasets/](https://github.com/vipulraheja/IteraTeR/tree/main/dataset). <br>
We splited IteraTeR dataset as follows:
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

All data and detailed description for the data structure can be found under [datasets/](https://github.com/vipulraheja/IteraTeR/tree/main/dataset). <br>
Code for collecting the revision history data can be found under [code/crawler/](https://github.com/vipulraheja/IteraTeR/tree/main/code/crawler). 



## Models

### Intent classification model

#### Model checkpoints

| Model         | Dataset        |  Edit-Intention  |  Precision  | Recall |  F1  |
| :-------------|:-------------  | :-----:| :-----:| :-----:| :-----:|
| [RoBERTa](https://huggingface.co/wanyu/IteraTeR-ROBERTA-Intention-Classifier)      | IteraTeR-HUMAN  | Clarity  | 0.75  | 0.63  | 0.69  |
| [RoBERTa](https://huggingface.co/wanyu/IteraTeR-ROBERTA-Intention-Classifier)    | IteraTeR-HUMAN  | Fluency  | 0.74  | 0.86  | 0.80  |
| [RoBERTa](https://huggingface.co/wanyu/IteraTeR-ROBERTA-Intention-Classifier)    | IteraTeR-HUMAN  | Coherence  | 0.29 | 0.36 | 0.32 |
| [RoBERTa](https://huggingface.co/wanyu/IteraTeR-ROBERTA-Intention-Classifier)    | IteraTeR-HUMAN  | Style  | 1.00 | 0.07 | 0.13  |
| [RoBERTa](https://huggingface.co/wanyu/IteraTeR-ROBERTA-Intention-Classifier)    | IteraTeR-HUMAN  | Meaning-changed  | 0.44 | 0.69 | 0.53  |

#### Model training and inference
The code and instructions for the training and inference of the intent classifier model can be found under [code/model/intent_classification/](https://github.com/vipulraheja/IteraTeR/tree/main/code/model/intent_classification).


### Generation models

#### Model checkpoints

| Model         | Dataset        |  SARI  |  BLEU  | ROUGE-L|  Avg.  |
| :-------------|:-------------  | :-----:| :-----:| :-----:| :-----:|
| [BART](https://huggingface.co/wanyu/IteraTeR-BART-Revision-Generator)      | IteraTeR-FULL  | 37.28  | 77.50  | 86.14  | 66.97  |
| [PEGASUS](https://huggingface.co/wanyu/IteraTeR-PEGASUS-Revision-Generator)   | IteraTeR-FULL  | 37.11  | 77.60  | 86.84  | 67.18  |


#### Model training and inference
The code and instructions for the training and inference of the Pegasus and BART models can be found under [code/model/generation/](https://github.com/vipulraheja/IteraTeR/tree/main/code/model/generation).



## Human-in-the-loop Iterative Text Revision
This repository also contains the code and data of the [Understanding Iterative Revision from Human-Written Text]().
The `IteraTeR_v2` dataset is larger than `IteraTeR` with around *24K more
unique documents* and *170K more edits*, which is splitted as follows:
<table>
	<tr>
		<th></th>
		<th>Train</th>
		<th>Dev</th>
		<th>Test</th>
	</tr>
	<tr>
		<td>IteraTeR_v2</td>
		<td>292929</td>
		<td>34029</td>
		<td>39511</td>
	</tr>
</table>

**Human-model interaction data in R3**: we also provide our collected human-model interaction data in R3 in [dataset/R3_eval_data.zip](https://github.com/vipulraheja/IteraTeR/tree/main/dataset/R3_eval_data.zip).


## Citation
If you find this work useful for your research, please cite our papers:

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


#### Read, Revise, Repeat: A System Demonstration for Human-in-the-loop Iterative Text Revision
```
@inproceedings{du2022r3,
    title = "Read, Revise, Repeat: A System Demonstration for Human-in-the-loop Iterative Text Revision",
    author = "*Du, Wanyu and *Kim, Zae Myung and Raheja, Vipul and Kumar, Dhruv and Kang, Dongyeop",
    booktitle = "Proceedings of the First Workshop on Intelligent and Interactive Writing Assistants",
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```

