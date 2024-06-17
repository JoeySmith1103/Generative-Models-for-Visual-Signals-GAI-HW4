# Generative-Models-for-Visual-Signals-GAI-HW4

## Run code
### Create conda environment
```shell=
conda create -n GAIHW4 python==3.10
```
### Clone the code
```shell=
git clone git@github.com:JoeySmith1103/Generative-Models-for-Visual-Signals-GAI-HW4.git
```
### Install library
```shell=
pip install torch torchvision torchaudio matplotlib
pip install denoising_diffusion_pytorch
pip install scikit-image
```

### Run code
```python=
python main.py
```

### File 
1. **main.py**: all code including training DIP model and DIP with DDPM model.
2. **best_dip_model_with_ddpm.pth**: best dip with ddpm model pth file.
3. **best_traditional_dip_model.pth**: best dip model pth file.
4. **result**: Including the results of diffent noise level and the improvement of our model and loss comparison.
