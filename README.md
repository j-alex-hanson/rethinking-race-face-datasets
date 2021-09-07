# Rethinking Race Face Datasets

## Setup

Should work for most Python 3.6+, we use Python 3.8.1.

Run `pip install --upgrade pip`.

Install necessary packages: `pip install -r requirements.txt`.

Download `BUPT-Balancedface`: http://www.whdeng.cn/RFW/Trainingdataste.html.

Download `RFW`: http://www.whdeng.cn/RFW/testing.html.

Modify the paths in `configs/data_default.py` according to wherever you put the datasets.

(Optionally) Instead of modifying `data_default.py`, add a new config (`data_[your name].py`) to `configs/`, and add branch logic to `main.py:get_data_config()` such that your config will be loaded.

## Experiments

Bash scripts that run our experiments can be found under the `experiments` directory.

Note that you will need to finish all setup steps before running an experiment!

The settings given in the `experiments` and `configs` directories will reproduce our experimental results, as reported in the paper.

## Citation

Please cite our work if you found this code useful:

```
@inproceedings{gwilliam2021rethinking,
  title={Rethinking Common Assumptions to Mitigate Racial Bias in Face Recognition Datasets},
  author={Gwilliam, Matthew and Hegde, Srinidhi and Tinubu, Lade and Hanson, Alex},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV) Workshops},
  year={2021}
}
```
