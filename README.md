# ruDALL-E 
### Generate images from texts

<a href="https://habr.com/ru/company/sberdevices/blog/586926/" class="logo" title="">
    <svg width="62" height="24" viewBox="0 0 62 24" xmlns="http://www.w3.org/2000/svg">
        <path fill="#CFB2FF" d="M16.875 19L11.075 10.225L16.825 1.4H12.6L8.75 7.4L4.94999 1.4H0.574994L6.32499 10.15L0.524994 19H4.79999L8.64999 12.975L12.525 19H16.875Z"></path>
        <path fill="#CFB2FF" d="M24.2607 5.775C20.8857 5.775 18.9607 7.625 18.6107 9.85H22.0107C22.2107 9.175 22.8607 8.6 24.1107 8.6C25.3357 8.6 26.2357 9.225 26.2357 10.425V11.025H23.4107C20.1107 11.025 18.1107 12.55 18.1107 15.2C18.1107 17.8 20.1107 19.3 22.6107 19.3C24.2857 19.3 25.6357 18.65 26.4357 17.6V19H29.8107V10.55C29.8107 7.4 27.5857 5.775 24.2607 5.775ZM23.6107 16.475C22.4857 16.475 21.7607 15.925 21.7607 15.025C21.7607 14.1 22.5607 13.55 23.6857 13.55H26.2357V14.125C26.2357 15.625 25.0107 16.475 23.6107 16.475Z"></path>
        <path fill="#CFB2FF" d="M39.925 6.3C38.125 6.3 36.65 6.95 35.7 8.275C35.95 5.85 36.925 4.65 39.375 4.275L44.3 3.55V0.375L39.025 1.25C33.925 2.1 32.35 5.5 32.35 11.175C32.35 16.275 34.825 19.3 39.2 19.3C43.125 19.3 45.55 16.3 45.55 12.7C45.55 8.825 43.3 6.3 39.925 6.3ZM39.025 16.25C37.125 16.25 36.075 14.725 36.075 12.675C36.075 10.7 37.175 9.275 39.05 9.275C40.875 9.275 41.9 10.75 41.9 12.7C41.9 14.65 40.9 16.25 39.025 16.25Z"></path>
        <path fill="#CFB2FF" d="M55.2855 5.775C53.3855 5.775 52.1605 6.6 51.5105 7.575V6.075H48.0105V23.775H51.6605V17.75C52.3105 18.65 53.5355 19.3 55.1855 19.3C58.3605 19.3 60.8855 16.8 60.8855 12.55C60.8855 8.225 58.3605 5.775 55.2855 5.775ZM54.4105 16.15C52.7105 16.15 51.5855 14.775 51.5855 12.6V12.5C51.5855 10.325 52.7105 8.925 54.4105 8.925C56.1105 8.925 57.2105 10.35 57.2105 12.55C57.2105 14.75 56.1105 16.15 54.4105 16.15Z"></path>
    </svg>
</a>
&nbsp
<a href="https://instagram.com/rudalle.official">
    <img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white"/>
</a>

[![Apache license](https://img.shields.io/badge/License-Apache-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Coverage Status](https://codecov.io/gh/sberbank-ai/ru-dalle/branch/master/graphs/badge.svg)](https://codecov.io/gh/sberbank-ai/ru-dalle)
[![pipeline](https://gitlab.com/shonenkov/ru-dalle/badges/master/pipeline.svg)](https://gitlab.com/shonenkov/ru-dalle/-/pipelines)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/sberbank-ai/ru-dalle/master.svg)](https://results.pre-commit.ci/latest/github/sberbank-ai/ru-dalle/master)

```
pip install rudalle==0.0.1rc5
```
### 🤗 HF Models:
[ruDALL-E Malevich (XL)](https://huggingface.co/sberbank-ai/rudalle-Malevich)


### Minimal Example:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wGE-046et27oHvNlBNPH07qrEQNE04PQ?usp=sharing)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/shonenkov/rudalle-example-generation)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/anton-l/rudall-e)

**Finetuning example**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Tb7J4PvvegWOybPfUubl5O7m5I24CBg5?usp=sharing)

**English translation example**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12fbO6YqtzHAHemY2roWQnXvKkdidNQKO?usp=sharing)

### generation by ruDALLE:
```python
from rudalle.pipelines import generate_images, show, super_resolution, cherry_pick_by_clip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan, get_ruclip
from rudalle.utils import seed_everything

# prepare models
device = 'cuda'
dalle = get_rudalle_model('Malevich', pretrained=True, fp16=True, device=device)
realesrgan = get_realesrgan('x4', device=device)
tokenizer = get_tokenizer()
vae = get_vae().to(device)
ruclip, ruclip_processor = get_ruclip('ruclip-vit-base-patch32-v5')
ruclip = ruclip.to(device)

text = 'изображение радуги на фоне ночного города'

seed_everything(42)
pil_images = []
scores = []
for top_k, top_p, images_num in [
    (2048, 0.995, 3),
    (1536, 0.99, 3),
    (1024, 0.99, 3),
    (1024, 0.98, 3),
    (512, 0.97, 3),
    (384, 0.96, 3),
    (256, 0.95, 3),
    (128, 0.95, 3), 
]:
    _pil_images, _scores = generate_images(text, tokenizer, dalle, vae, top_k=top_k, images_num=images_num, top_p=top_p)
    pil_images += _pil_images
    scores += _scores

show(pil_images, 6)
```
![](./pics/rainbow-full.png)
### auto cherry-pick by ruCLIP:
```python
top_images, clip_scores = cherry_pick_by_clip(pil_images, text, ruclip, ruclip_processor, device=device, count=6)
show(top_images, 3)
```
![](./pics/rainbow-cherry-pick.png)
### super resolution:
```python
sr_images = super_resolution(top_images, realesrgan)
show(sr_images, 3)
```
![](./pics/rainbow-super-resolution.png)

```python
text, seed = 'красивая тян из аниме', 6955
```
![](./pics/anime-girl-super-resolution.png)


### Image Prompt
see `jupyters/ruDALLE-image-prompts-A100.ipynb`
```python
text, seed = 'Храм Василия Блаженного', 42
skyes = [red_sky, sunny_sky, cloudy_sky, night_sky]
```
![](./pics/russian-temple-image-prompt.png)


### 🚀 Contributors 🚀

- [@neverix](https://www.kaggle.com/neverix) thanks a lot for contributing for speed up of inference
- [@Igor Pavlov](https://github.com/boomb0om) trained model and prepared code with [super-resolution](https://github.com/boomb0om/Real-ESRGAN-colab)
- [@oriBetelgeuse](https://github.com/oriBetelgeuse) thanks a lot for easy API of generation using image prompt 
- [@Alex Wortega](https://github.com/AlexWortega) created first FREE version colab notebook with fine-tuning [ruDALL-E Malevich (XL)](https://huggingface.co/sberbank-ai/rudalle-Malevich) on sneakers domain 💪 
- [@Anton Lozhkov](https://github.com/anton-l) Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio), see [here](https://huggingface.co/spaces/anton-l/rudall-e)
