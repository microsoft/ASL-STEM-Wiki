# ASL STEM Wiki

This repository contains the code supporting our paper titled "ASL STEM Wiki: Dataset and Benchmark for Interpreting STEM Articles". Specifically, it contains code for replicating our fingerspelling detection and alignment benchmarks. Below, you will find instructions on how to use the code. 

Project page: https://www.microsoft.com/en-us/research/project/asl-stem-wiki

## Getting started

Install dependencies: 
```
pip install -r requirements.txt
```
Download data: 
```
wget https://download.microsoft.com/download/4/c/f/4cfec788-7478-4e47-9a15-ace9b6a96198/ASL_STEM_Wiki.zip
unzip ASL_STEM_Wiki.zip ASL_STEM_Wiki_data
```
Preprocess data (note - this may take a while):
```
python src/run_mediapipe.py 
```

## Usage
**Train fingerspelling detection model**
Setup configs in `src/train.py`:
```
if __name__ == '__main__':
    cfg = namedtuple('Config', ['seed', 'meta', 'data', 'model', 'train', 'wandb', 'finetune'])
    cfg.seed = 42
    cfg.meta = namedtuple('Meta', ['device'])
    cfg.meta.device = 'cuda'
    cfg.data = namedtuple('Data', ['datadir', 'label_file', 'clip_length', 'batch_size', 'ft_datadir', 'ft_label_file'])
    cfg.data.datadir = 'ASL_STEM_Wiki_data/videos/'
    cfg.data.label_file = 'ASL_STEM_Wiki_data/videos.csv'
    cfg.data.clip_length = 500 # 4 * clip_length is actual max video length
    cfg.data.batch_size = 4
    cfg.data.seq_length = 600
    cfg.data.ft_datadir = 'ASL_STEM_Wiki_data/videos/'
    cfg.data.ft_label_file = 'fs-annotations/train.csv'
    cfg.data.eval_article = 'Hal Anger' # Replace with the held-out article for cross-validation
    cfg.model = namedtuple('Model', ['type', 'num_keypoints', 'hidden_feature', 'p_dropout', 'num_stages'])
    cfg.model.type = 'multitask'
    cfg.model.num_keypoints = 75
    cfg.model.hidden_feature = 768 # try smaller hidden size
    cfg.model.p_dropout = 0.3
    cfg.model.num_stages = 6
    cfg.train = namedtuple('Train', ['learning_rate', 'num_epochs', 'model_dir'])
    cfg.train.learning_rate = 1e-3
    cfg.train.num_epochs = 0
    cfg.train.model_dir = 'model'
    cfg.wandb = namedtuple('Wandb', ['name'])
    cfg.wandb.name = 'st-Hal_Anger'
    cfg.finetune = namedtuple('Finetune', ['num_epochs', 'model_dir'])
    cfg.finetune.num_epochs = 40
    cfg.finetune.model_dir = 'model'
```


Train:
```
python src/train.py
```
Run the fingerspelling alignment heuristic model:
```
python src/align.py
```

## Citation
If you found the resources in this repository useful, please cite our [paper](https://aclanthology.org/2024.emnlp-main.801/).

```
@inproceedings{yin-etal-2024-asl,
    title = "{ASL} {STEM} {W}iki: Dataset and Benchmark for Interpreting {STEM} Articles",
    author = "Yin, Kayo  and
      Singh, Chinmay  and
      Minakov, Fyodor O  and
      Milan, Vanessa  and
      Daum{\'e} III, Hal  and
      Zhang, Cyril  and
      Lu, Alex Xijie  and
      Bragg, Danielle",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.801/",
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Responsible AI Transparency Documentation

Intended uses: This repo is best suited for detecting and aligning instances of American Sign Language (ASL) fingerspelling with accompanying English text translations. This repo is being shared with the research community to facilitate reproduction of our results and foster further research in this area. This repo is intended to be used by domain experts who are independently capable of evaluating the quality of outputs before acting on them. Further details on use cases can be found on the [ASL STEM Wiki project website](https://www.microsoft.com/en-us/research/project/asl-stem-wiki).  

Out-of-scope uses: This repo is not well suited to be used alone for other sign language processing tasks, like end-to-end translation. In particular, we developed the code with ASL and English in mind, and the code may not immediately apply to other signed languages, as each signed language is unique. We do not recommend using this repo in commercial or real-world applications without further testing and development. It is being released for research purposes. This repo was not designed or evaluated for all possible downstream purposes. Developers should consider its inherent limitations (more below) as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness concerns specific to each intended downstream use. We do not recommend using this repo in the context of high-risk decision making (e.g. in law enforcement, legal, finance, or healthcare).  Please see the [ASL STEM Wiki project page](https://www.microsoft.com/en-us/research/project/asl-stem-wiki) for more information on appropriate and inappropriate use cases. 

Evaluation: This repo was evaluated on its ability to perform fingerspelling detection and alignment with associated text, using the ASL STEM Wiki dataset. We used mean IOU (Intersection Over Union) scores on fingerspelling detection and alignment to measure performance, and compared against a random baseline. We found that our method performed with mean IOU for detection between .19 and .28, and with mean IOU for alignment of 0.13, compared to the random baseline of 0.06 for both tasks. A detailed discussion of our evaluation methods and results can be found in our paper.  

Limitations: This repo was developed for research and experimental purposes. Further testing and validation are needed before considering its application in commercial or real-world scenarios. This repo was designed and tested using American Sign Language and English. Performance in other languages may vary and should be assessed by someone who is both an expert in the expected outputs and a native user of both the signed and written languages. In addition, choice of training data will impact performance, with large amounts of high-quality ASL likely to improve performance. This repo should not be used in highly regulated domains where inaccurate outputs could suggest actions that lead to injury or negatively impact an individual's legal, financial, or life opportunities. Please see the [ASL STEM Wiki project page](https://www.microsoft.com/en-us/research/project/asl-stem-wiki) for more information on important limitations. 

Best Practices: Better performance can be achieved by training on larger datasets for general use, training on specialized datasets for domain-limited use, and developing better machine learning techniques. Users are responsible for sourcing their datasets legally and ethically. This could include securing appropriate copy rights, ensuring consent for use of audio/images, and/or the anonymization of data prior to use in research. Users are reminded to be mindful of data privacy concerns and are encouraged to review the privacy policies associated with any models and data storage solutions interfacing with this repo. It is the user’s responsibility to ensure that the use of this repo complies with relevant data protection regulations and organizational guidelines. 

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
