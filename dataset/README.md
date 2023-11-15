# Datasets
Due to license, please acquire first [the Newsela Dataset](https://newsela.com/data/) and email [the first author](mailto:zaemyung@gmail.com) for obtaining the IteraTeR Plus and IteraTeR-v2 datasets.

## IteraTeR Dataset

The IteraTeR dataset is splitted as follows:
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


### Document-level datasets
For `Document-level` dataset, it has a document-level revision per line. Here is a sample of a document-level revision in `IteraTeR-HUMAN`:
```
{
	"doc_id": "1912.10514", 
	"revision_depth": 1, 
	"before_revision": "An effective method to generate a large number of parallel sentences for training improved neural machine translation (NMT) systems is the use of back-translations of the target-side monolingual data.  Tagging, or using gates, has been used to enable translation models to distinguish between synthetic and natural data. This improves standard back-translation and also enables the use of iterative back-translation on language pairs that underperformed using standard back-translation. This work presents a simplified approach of differentiating between the two data using pretraining and finetuning . The approach - tag-less back-translation - trains the model on the synthetic data and finetunes it on the natural data. Preliminary experiments have shown the approach to continuously outperform the tagging approach on low resource English-Vietnamese neural machine translation . While the need for tagging (noising) the dataset has been removed, the approach outperformed the tagged back-translation approach by an average of 0.4 BLEU  .", 
	"after_revision": "An effective method to generate a large number of parallel sentences for training improved neural machine translation (NMT) systems is the use of back-translations of the target-side monolingual data. The method was not able to utilize the available huge amount of monolingual data because of the inability of models to differentiate between the authentic and synthetic parallel data. Tagging, or using gates, has been used to enable translation models to distinguish between synthetic and authentic data, improving standard back-translation and also enabling the use of iterative back-translation on language pairs that under-performed using standard back-translation. This work presents pre-training and fine-tuning as a simplified but more effective approach of differentiating between the two data . The approach - tag-less back-translation - trains the model on the synthetic data and fine-tunes it on the authentic data. Experiments have shown the approach to outperform the baseline and standard back-translation by 4.0 and 0.7 BLEU respectively on low resource English-Vietnamese NMT . While the need for tagging (noising) the dataset has been removed, the technique outperformed tagged back-translation by 0.4 BLEU . The approach reached the best scores in less training time than the standard and tagged back-translation approaches .", 
	"edit_actions": [
		{
			"type": "A", 
			"before": null, 
			"after": "The method was not able to utilize the available huge amount of monolingual data because of the inability of models to differentiate between the authentic and synthetic parallel data.", 
			"start_char_pos": 201, 
			"end_char_pos": 201, 
			"major_intent": "meaning-changed", 
			"raw_intents": ["meaning-changed", "meaning-changed", "meaning-changed"]
		}, 
		{
			"type": "R", 
			"before": "natural data. This improves", 
			"after": "authentic data, improving", 
			"start_char_pos": 307, 
			"end_char_pos": 334, 
			"major_intent": "clarity", 
			"raw_intents": ["coherence", "clarity", "clarity"]
		}, 
		...
	], 
	"sents_char_pos": [0, 200, 320, 486, 602, 722, 882], 
	"domain": "arxiv"
}
```
The document-level revision object contains the following keys:
- `doc_id`: unique document id;
- `revision_depth`: current revision depth for the document, starting from 1;
- `before_revision`: the source document to be revised;
- `after_revision`: the target document that has been revised;
- `edit_actions`: a list of edit actions applied on the `before_revision`, each action is a json object which contains the following keys:
	- `type`: the operation type of the edit action, `A` means add, `D` means delete, `R` means replace;
	- `before`: a span of original text in `before_revision`;
	- `after`: a span of revised text in `after_revision`;
	- `start_char_pos`: the starting index of the current edit action in `before_revision`;
	- `end_char_pos`: the ending index of the current edit action in `before_revision`;
	- `major_intent`: the final edit intention annotation obtained by majority vote (Note: this key only occurs in `IteraTeR-HUMAN`);
	- `raw_intents`: the raw edit intention annotations from 3 human annotators (Note: this key only occurs in `IteraTeR-HUMAN`);
- `sents_char_pos`: a list of starting indices of each original sentence in `before_revision`;
- `domain`: domain of the document.


### Sentence-level datasets
For `Sentence-level` dataset, it has a sentence-level revision per line. Here is a sample of a sentence-level revision in `IteraTeR-FULL`:
```
{
	"before_sent": " While ordinary differential equation (ODE) models remain the conceptual framework for modelling many cellular processes, specific situations demand stochastic models to capture the influence of noise.", 
	"before_sent_with_intent": "<clarity>  While ordinary differential equation (ODE) models remain the conceptual framework for modelling many cellular processes, specific situations demand stochastic models to capture the influence of noise.", 
	"after_sent": " While ordinary differential equations (ODEs) form the conceptual framework for modelling many cellular processes, specific situations demand stochastic models to capture the influence of noise.", 
	"labels": "clarity", 
	"confidence": "0.99741876", 
	"doc_id": "0809.0773", 
	"revision_depth": "1"
}
```
The sentence-level revision object contains the following keys:
- `before_sent`: the original sentence;
- `before_sent_with_intent`: appending a edit intention tag before `before_sent` for conditional text generation;
- `after_sent`: the revised sentence;
- `labels`: edit intention for the current sentence pair; 
- `confidence`: model confidence score for predicting `labels` (Note: this key only occurs in `IteraTeR-FULL`);
- `doc_id`: unique document id;
- `revision_depth`: current revision depth for the document, starting from 1.



## IteraTeR_v2 Dataset

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

The `IteraTeR_v2` dataset shares the same data structure as the sentence-level `IteraTeR-full` dataset.

However, we also incoporate the context sentences into `before_sent_with_intent` for the `IteraTeR_v2` dataset with the following format:<br>
`previous 2 context sentences <S> current sentence to be revised </S> following 2 context sentences`.

Note that if the previous or following sentences do not exist, there will be just a whitespace.


**Human-model interaction data in R3**: we also provide our collected human-model interaction data in R3 in [dataset/R3_eval_data.zip](https://github.com/vipulraheja/IteraTeR/tree/main/dataset/R3_eval_data.zip).


## Code
See [code/crawler/](https://github.com/vipulraheja/IteraTeR/tree/main/code/crawler) for the source code for collecting revision histories and extracting edits.


## Citation
If you find this work useful for your research, please cite our paper:

#### Understanding Iterative Revision from Human-Written Text
```
@inproceedings{du-etal-2022-understanding-iterative,
    title = "Understanding Iterative Revision from Human-Written Text",
    author = "Du, Wanyu  and
      Raheja, Vipul  and
      Kumar, Dhruv  and
      Kim, Zae Myung  and
      Lopez, Melissa  and
      Kang, Dongyeop",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.250",
    pages = "3573--3590"
}
```


#### Read, Revise, Repeat: A System Demonstration for Human-in-the-loop Iterative Text Revision
```
@inproceedings{du-etal-2022-read,
    title = "Read, Revise, Repeat: A System Demonstration for Human-in-the-loop Iterative Text Revision",
    author = "Du, Wanyu  and
      Kim, Zae Myung  and
      Raheja, Vipul  and
      Kumar, Dhruv  and
      Kang, Dongyeop",
    booktitle = "Proceedings of the First Workshop on Intelligent and Interactive Writing Assistants (In2Writing 2022)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.in2writing-1.14",
    pages = "96--108"
}
```

