# VOGUE: Answer Verbalization through Multi-Task Learning

In recent years, there have been significant developments in Question Answering over Knowledge Graphs (KGQA). Despite all the notable advancements, current KGQA systems only focus on answer generation techniques and not on answer verbalization. However, in real-world scenarios (e.g., voice assistants such as Alexa, Siri, etc.), users prefer verbalized answers instead of a generated response. This paper addresses the task of answer verbalization for (complex) question answering over knowledge graphs. In this context, we propose a multi-task-based answer verbalization framework: VOGUE (**V**erbalization thr**O**u**G**h m**U**ltitask l**E**arning). The VOGUE framework attempts to generate a verbalized answer using a hybrid approach through a multi-task learning paradigm. Our framework can generate results based on using questions and queries as inputs concurrently. VOGUE comprises four modules that are trained simultaneously through multi-task learning. We evaluate our framework on all existing datasets for answer verbalization, and it outperforms all current baselines on both BLEU and METEOR scores.

![VOGUE](image/architecture.png?raw=true "VOGUE architecture")

VOGUE's architecture. It consists of four modules: 1) A dual encoder that is responsible to encode both inputs (question, logical form). 2) A similarity threshold module that determines whether the encoded inputs are relevant and determines if both will be used for verbalization. 3) A cross-attention module that performs question and query matching by jointly modeling the relationships of question words and query actions. 4) A hybrid decoder that generates the verbalized answer using the information of both question and logical form representations from the cross-attention module.

## Requirements and Setup

Python version >= 3.7

PyTorch version >= 1.8.1

``` bash
# clone the repository
git clone https://github.com/endrikacupaj/VOGUE.git
cd VOGUE
pip install -r requirements.txt
```

## Answer Verbalization Datasets
Our framework was evaluated on three answer verbalization datasets. You can download the datasets from the links below:
* VQuAnDA: https://figshare.com/projects/VQuAnDa/72488
* ParaQA: https://figshare.com/projects/ParaQA/94010
* VANiLLa: https://sda.tech/projects/vanilla/

After downloading the datasets, please move them under the folder [data](data).

## Preprocess data
You will need to execute three scripts in order to preprocess the datasets and annotate them with gold logical forms. You can do that by running:

``` bash
# preprocess VQuAnDa
python scripts/preprocess_vquanda.py

# preprocess ParaQA
python scripts/preprocess_paraqa.py

# preprocess VANiLLa
python scripts/preprocess_vanilla.py
```

## Train VOGUE
After preprocessing the data, you can train our framework by running:
``` bash
# train framework
python train.py
```
In [args](args.py) file, you can specify which dataset to train and even experiment with different model settings.

## Test VOGUE
For testing our framework, you can run the test file by:
``` bash
# test framework
python test.py
```
Please consider the [args](args.py) file for specifying the desired checkpoint path.

## License
The repository is under [MIT License](LICENCE).

## Cite
Coming Soon!