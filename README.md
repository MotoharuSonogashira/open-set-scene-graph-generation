# Open-Set Scene Graph Generation

This is the implementation of the paper "[Towards Open-Set Scene Graph Generation with Unknown Objects](https://ieeexplore.ieee.org/abstract/document/9690166)" by Sonogashira et al. (IEEE Access, 2022). It is also intended to be a benchmark for the problem of open-set scene graph generation.

## Installation

### Requirements

- Python 3.8
- CUDA 11.2

```bash
# clone this repository and build binaries
git clone https://github.com/MotoharuSonogashira/open-set-scene-graph-generation.git

# install dependencies
cd open-set-scene-graph-generation
pip3 install --user -r requirements.txt

# install pycocotools
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python3 setup.py build_ext install --user
cd ../..

# install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
sed -i 's/^\(\s*\)\(check_cuda_torch_binary_vs_bare_metal\)/\1#\2/' setup.py # workaround for CUDA and PyTorch version mismatch
python3 setup.py install --cuda_ext --cpp_ext --user
cd ..

# build binaries
python3 setup.py build develop --user
```
At this point, you may continue to follow the Installation and Usage sections of the open-set dataset preprocessor repository [MotoharuSonogashira/open-set-scene-graph-dataset](https://github.com/MotoharuSonogashira/open-set-scene-graph-dataset) in order to prepare the open-set dataset required by this repository. Note that the preprocessing repository must be cloned into the root directory of this repository to perform the visualization described below.

## Dataset

Use [the above-mentioned repository](https://github.com/MotoharuSonogashira/open-set-scene-graph-dataset) to preprocess the Visual Genome dataset and put the following results in the `datasets/vg` directory:
- `VG_100K`: the image data directory. This should be the renamed version of `open-set-scene-graph-dataset/data_tools/VG/images` directory (which contains decompressed files from split zip files) if you follow the preprocessor's Usage. Note that the intermediate image database file `imdb_1024.h5` is not used by this repository.
- `VG-SGG-open.h5` and `VG-SGG-dicts-open.json`: the preprocessed label database files. These should be in `open-set-scene-graph-dataset/data_tools`.

To summarize, assuming you have cloned the preprocessing repository into the root of this repository and followed its Installation and Usage sections, do the following:
```bash
# go back from the root of the preprocessor repository to that of this repository
cd ..

# move the image data
mv open-set-scene-graph-dataset/data_tools/VG/images datasets/vg/VG_100K

# move the label data
mv open-set-scene-graph-dataset/data_tools/{VG-SGG-open.h5,VG-SGG-dicts-open.json} datasets/vg/
```

## Usage

The shell script `run.bash` wraps Python scripts to train and test different methods of scene graph generation. It stores the training and testing results of each method in a subdirectory under `output/open-outer` by default. The script can be configured via environmental variables; for example, to each command executing it,
- Add `GPU=4` to use 4 GPUs. By default without this variable, `torch.distributed` is disabled. Use the standard `CUDA_VISIBLE_DEVICES` variable to choose which GPUs are visible to the script.
- Add `VAL=1` to perform validation (which stores results in `output/open-inner`).

All the following commands are assumed to be executed in the root directory of this repository.

### Faster R-CNN Pretraining

Execute the following command before training any methods:
```
MODEL=faster_rcnn ./run.bash
```
This is required only once since all the methods share the pretrained model.

### Training

Fit a model:
```
MODEL=vctree ./run.bash
```
- Change the `MODEL` value to `freq` or `imp` to use a different model.

All the methods with the same `MODEL` share the trained weights in testing.

### Testing

Evaluate a method based on the trained model:
```
TEST=1 MODEL=vctree THRESHOLD=.1 ./run.bash
```
which predicts scene graphs of testing images using the trained `vctree` model and unknown detection with threshold `.1` (the method named "VCTree+" in the paper), computes open-set SGDet recall metrics, and saves the results in the result directory named `vctree-.1`.

- Change the `MODEL` value in accordance with the training.
- Change the `THRESHOLD` value to use a different threshold. The suffix of the result directory name changes accordingly. In our experiments, the best threshold selected by validation was `.1` for `vctree` and `.3` for `freq` and `imp`. Alternatively, set it to
  - `.0` to use the original method without thresholding-based unknown detection (the method simply named "VCTree" etc. in the paper).
  - The empty string (no value after `=`) to use the closed-set recall metrics. Then, unknown objects in ground-truth labels are ignored and unknown detection is disabled. The directory name simply becomes `vctree` etc.
- Add `PROTOCOL=sgcls` to use SGCls metrics. The directory name becomes `vctree-.1-sgcls` etc.

### Metrics

#### Recalls
Print recall metrics (SGDet or SGCls) after testing:
```
utils/recalls.py output/open-outer/vctree-.1
```

#### Object and Relationship Counts
Compute and print object and relationship count metrics after testing:
```
utils/counts.py output/open-outer/vctree-.1
```

which also creates cache files in the directory, and then the counts can be plotted:
```
utils/plot_counts.py output/open-outer/vctree-.1
```
where multiple result directories may be passed to plot them in a single figure.

### Visualization

Note: this depends on the [the above-mentioned dataset preprocessing repository](https://github.com/MotoharuSonogashira/open-set-scene-graph-dataset) and assumes that it is cloned as the directory `open-set-scene-graph-dataset` under the root directory of this repository.

Visualize the detected objects and relationships after testing:
```
utils/viz.py -U output/open-outer/vctree-.1
```
which shows the object detection result and the scene graph in separate windows for each testing image.
- Remove the `-U` option to show detections not related to ground-truth unknown objects as well.
- Use the `-O` option with a directory name to save the images into the directory instead of showing.

## Acknowledgements

This repository is based on [KaihuaTang/Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), the implementation of the paper "Unbiased Scene Graph Generation from Biased Training" by Tang et al. (CVPR, 2019).


## Citations

```
@article{sonogashira2022towards,
  title={Towards Open-Set Scene Graph Generation with Unknown Objects},
  author={Sonogashira, Motoharu and Iiyama, Masaaki and Kawanishi, Yasutomo},
  journal={IEEE Access},
  volume={10},
  number={},
  pages={11574-11583},
  year={2022},
  publisher={IEEE}
}
```
