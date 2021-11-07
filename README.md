# ruDALL-E
### Generate images from texts

[![Apache license](https://img.shields.io/badge/License-Apache-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Coverage Status](https://codecov.io/gh/sberbank-ai/ru-dalle/branch/master/graphs/badge.svg)](https://codecov.io/gh/sberbank-ai/ru-dalle)
[![pipeline](https://gitlab.com/shonenkov/ru-dalle/badges/master/pipeline.svg)](https://gitlab.com/shonenkov/ru-dalle/-/pipelines)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/sberbank-ai/ru-dalle/master.svg)](https://results.pre-commit.ci/latest/github/sberbank-ai/ru-dalle/master)

```
pip install rudalle==0.0.1rc6
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

### Social Media

[![](./pics/habr.svg)](https://habr.com/ru/company/sberdevices/blog/586926/)
[![](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/rudalle.official)
