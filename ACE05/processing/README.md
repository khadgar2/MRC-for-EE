# ACE2005 preprocessing

This preprocessing pipeline is based on https://github.com/nlpcl-lab/ace2005-preprocessing. 

This is a simple code for preprocessing ACE 2005 corpus for Event Extraction task, based on the github repo https://github.com/nlpcl-lab/ace2005-preprocessing. 

To make the entities and the arguments correspond more clearly, we added another processing stage after the original process.

Prerequisites and usage are the same as original code.

## Prerequisites

1. Prepare **ACE 2005 dataset**. 

	 (Download: https://catalog.ldc.upenn.edu/LDC2006T06. Note that ACE 2005 dataset is not free.)

2. Install the packages.
	 ```
	 pip install stanfordcorenlp beautifulsoup4 nltk tqdm
	 ```
		
3. Download stanford-corenlp model.
		```bash
		wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
		unzip stanford-corenlp-full-2018-10-05.zip
		```

## Usage

Run:

```bash
sudo python main.py --data=./data/ace_2005_td_v7/data/English --nlp=./stanford-corenlp-full-2018-10-05
``` 

- Then you can get the parsed data in `output directory`. 

- If it is not executed with the `sudo`, an error can occur when using `stanford-corenlp`.

- It takes about 30 minutes to complete the pre-processing.

## Output

### Format

Each file is a JSON containing a list of instances. Below is a sample of an instance. For each instance, "words" is a list of words in the sentence; "event-mentions" is a list of events in the sentence, where "arguments" in event-mentions correspond with the "entities"; "entities" is a list of entities in the sentence; "event-labels" is a list of event-types corresponding with the words in the sentence.

Below is a sample: 

**`sample.json`**
```json
[
	{
		"words": [
			"Orders",
			"went",
			"out",
			"today",
			"to",
			"deploy",
			"17,000",
			"U.S.",
			"Army",
			"soldiers",
			"in",
			"the",
			"Persian",
			"Gulf",
			"region",
			"."
		],
		"event-mentions": [
			{
				"trigger": {
					"text": "deploy",
					"start": 5,
					"end": 6
				},
				"arguments": [
					"None",
					"Artifact",
					"None",
					"Destination",
					"None",
					"None"
				],
				"event_type": "Movement:Transport"
			}
		],
		"entities": [
			{
				"text": "U.S",
				"entity-type": "GPE:Nation",
				"head": {
					"text": "U.S",
					"start": 7,
					"end": 8
				},
				"entity_id": "CNN_CF_20030303.1900.02-E4-186",
				"start": 7,
				"end": 8
			},
			{
				"text": "17,000 U.S. Army soldiers",
				"entity-type": "PER:Group",
				"head": {
					"text": "soldiers",
					"start": 9,
					"end": 10
				},
				"entity_id": "CNN_CF_20030303.1900.02-E25-1",
				"start": 6,
				"end": 10
			},
			{
				"text": "U.S. Army",
				"entity-type": "ORG:Government",
				"head": {
					"text": "Army",
					"start": 8,
					"end": 9
				},
				"entity_id": "CNN_CF_20030303.1900.02-E66-157",
				"start": 7,
				"end": 9
			},
			{
				"text": "the Persian Gulf region",
				"entity-type": "LOC:Region-International",
				"head": {
					"text": "region",
					"start": 14,
					"end": 15
				},
				"entity_id": "CNN_CF_20030303.1900.02-E76-191",
				"start": 11,
				"end": 15
			},
			{
				"text": "Persian Gulf",
				"entity-type": "LOC:Water-Body",
				"head": {
					"text": "Persian Gulf",
					"start": 12,
					"end": 14
				},
				"entity_id": "CNN_CF_20030303.1900.02-E77-192",
				"start": 12,
				"end": 14
			},
			{
				"text": "today",
				"entity-type": "TIM:time",
				"head": {
					"text": "today",
					"start": 3,
					"end": 4
				},
				"entity_id": "CNN_CF_20030303.1900.02-T6-1",
				"start": 3,
				"end": 4
			}
		],
		"event-labels": [
			"Other",
			"Other",
			"Other",
			"Other",
			"Other",
			"B-Movement:Transport",
			"Other",
			"Other",
			"Other",
			"Other",
			"Other",
			"Other",
			"Other",
			"Other",
			"Other",
			"Other"
		]
	}
]
```


### Data Split

The result of data is divided into test/dev/train as follows.
```
├── output
│		 └── test_process.json
│		 └── dev_process.json
│		 └── train_process.json
│...
```

This project use the same data partitioning as the previous work ([Yang and Mitchell, 2016](https://www.cs.cmu.edu/~bishan/papers/joint_event_naacl16.pdf);	[Nguyen et al., 2016](https://www.aclweb.org/anthology/N16-1034)). The data segmentation is specified in `data_list.csv`.

Below is information about the amount of parsed data when using this project. It is slightly different from the parsing results of the two papers above. The difference seems to have occurred because there are no promised rules for splitting sentences within the sgm format files.

|					| Documents		|	Sentences	 |Triggers		| Arguments | Entity Mentions	|
|-------	 |--------------|--------------|------------|-----------|----------------- |
| Test		 | 40				| 713					 | 422					 | 892						 |	4226						 |
| Dev			| 30				| 875					 | 492					 | 933						 |	4050						 |
| Train		| 529			 | 14724				 | 4312					| 7811						 |	 53045						|
