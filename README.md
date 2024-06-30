
# Vivid Dreamer : Text to 3D
![](https://github.com/guru-narayana/Vivid_dreamer/blob/master/images/clock.gif)
![](https://github.com/guru-narayana/Vivid_dreamer/blob/master/images/sofa.gif)
![](https://github.com/guru-narayana/Vivid_dreamer/blob/master/images/diamond.gif)
## Get Started
**Installation**
Install [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Shap-E](https://github.com/openai/shap-e#usage) as fellow:
```
conda create -n gdreamer -y python=3.8

git clone https://github.com/hustvl/GaussianDreamer.git 

pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

pip install ninja

cd GaussianDreamer

pip install -r requirements.txt

conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
conda install conda-forge::glm

pip install ./gaussiansplatting/submodules/diff-gaussian-rasterization
pip install ./gaussiansplatting/submodules/simple-knn
pip install plyfile
pip install ipywidgets
pip install open3d

git clone https://github.com/openai/shap-e.git
cd shap-e
pip install -e .

pip install git+https://github.com/bytedance/MVDream
```

Download [finetuned Shap-E](https://huggingface.co/datasets/tiange/Cap3D/resolve/9bfbfe7910ece635e8e3077bed6adaf45186ab48/our_finetuned_models/shapE_finetuned_with_330kdata.pth) by Cap3D, and put it in `./load`


**Quickstart**

Text-to-3D Generation
```
python launch.py --config configs/lucid_dreamer.yaml --train --gpu 0 system.prompt_processor.prompt="a fox"

# if you want to import the generated 3D assets into the Unity game engine.
python launch.py --config configs/lucid_dreamer.yaml --train --gpu 0 system.prompt_processor.prompt="a fox" system.sh_degree=3 
```

## Sample results


## 📑 Citation
Some source code of ours is borrowed from [Threestudio](https://github.com/threestudio-project/threestudio), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [depth-diff-gaussian-rasterization](https://github.com/ingra14m/depth-diff-gaussian-rasterization). We sincerely appreciate the excellent works of these authors.

The current project is modifcation and improvement of the existing work, [GaussianDreamer](https://github.com/hustvl/GaussianDreamer), cited below.

```
@inproceedings{yi2023gaussiandreamer,
  title={GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models},
  author={Yi, Taoran and Fang, Jiemin and Wang, Junjie and Wu, Guanjun and Xie, Lingxi and Zhang, Xiaopeng and Liu, Wenyu and Tian, Qi and Wang, Xinggang},
  year = {2024},
  booktitle = {CVPR}
}
```
