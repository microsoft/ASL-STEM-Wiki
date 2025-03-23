# ASL STEM Wiki

This repository contains the code and data supporting our paper titled "ASL STEM Wiki: Dataset and Benchmark for Interpreting STEM Articles". Below, you will find instructions on how to use the code and access the data.

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
    cfg.data.eval_article = 'Hal Anger' # Replace 
    cfg.model = namedtuple('Model', ['type', 'num_keypoints', 'hidden_feature', 'p_dropout', 'num_stages'])
    cfg.model.type = 'multitask'
    cfg.model.num_keypoints = 75
    cfg.model.hidden_feature = 768 # try smaller hidden size
    cfg.model.p_dropout = 0.3
    cfg.model.num_stages = 6
    cfg.train = namedtuple('Train', ['learning_rate', 'num_epochs', 'model_dir'])
    cfg.train.learning_rate = 1e-3
    cfg.train.num_epochs = 0
    cfg.train.model_dir = 'test'
    cfg.wandb = namedtuple('Wandb', ['name'])
    cfg.wandb.name = 'st-Hal_Anger'
    cfg.finetune = namedtuple('Finetune', ['num_epochs', 'model_dir'])
    cfg.finetune.num_epochs = 40
    cfg.finetune.model_dir = 'test'
```


Train:
```
python src/train.py
```
Run the fingerspelling alignment heuristic model:

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

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
