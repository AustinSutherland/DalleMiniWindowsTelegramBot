import os

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1' # fixes an error if your GPU CUDA runtime is older than the dev toolkit
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.80' # Preallocates 80% of your vram. Adjust if you have issues

import jax.numpy as jnp
import jax
from functools import partial
import random
from dalle_mini import DalleBartProcessor

from IPython.display import display
from PIL import Image, ImageFont, ImageDraw 

# Load models & tokenizer
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from flax.jax_utils import replicate

from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange

## The code here is modified from https://colab.research.google.com/github/borisdayma/dalle-mini/blob/main/tools/inference/inference_pipeline.ipynb

class DalleMini:
    def __init__(self):

        # you can use dalle-mini instead by uncommenting below line
        # This model is not as good
        # DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"

        DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"
        DALLE_COMMIT_ID = None
        # VQGAN model
        VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
        VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

        # check how many devices are available
        print(jax.local_device_count())

        # Load dalle-mini
        self.model, params = DalleBart.from_pretrained(
            DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
        )

        # Load VQGAN
        self.vqgan, vqgan_params = VQModel.from_pretrained(
            VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
        )

        self.params = replicate(params)
        self.vqgan_params = replicate(vqgan_params)

        self.processor = DalleBartProcessor.from_pretrained(
            DALLE_MODEL, revision=DALLE_COMMIT_ID)

        # number of predictions per prompt
        self.n_predictions = 4

        # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
        self.gen_top_k = None
        self.gen_top_p = None
        self.temperature = None
        self.cond_scale = 10.0


    def generate(self, prompt):
        # generate images
        images = []

        # model inference
        @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
        def p_generate(
            tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
        ):
            return self.model.generate(
                **tokenized_prompt,
                prng_key=key,
                params=params,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                condition_scale=condition_scale,
            )

        # decode image
        @partial(jax.pmap, axis_name="batch")
        def p_decode(indices, params):
            return self.vqgan.decode_code(indices, params=params)

        # create a random key
        seed = random.randint(0, 2**32 - 1)
        key = jax.random.PRNGKey(seed)

        tokenized_prompts = self.processor([prompt])
        self.tokenized_prompt = replicate(tokenized_prompts)

        for i in trange(max(self.n_predictions // jax.device_count(), 1)):
            # get a new key
            key, subkey = jax.random.split(key)
            # generate images
            encoded_images = p_generate(
                self.tokenized_prompt,
                shard_prng_key(subkey),
                self.params,
                self.gen_top_k,
                self.gen_top_p,
                self.temperature,
                self.cond_scale,
            )
            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]
            # decode images
            decoded_images = p_decode(encoded_images, self.vqgan_params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            for decoded_img in decoded_images:
                img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                images.append(img)

        title_bar_height = 30
        title_padding = 6

        # Create one big image to hold the sub predictions
        new_im = Image.new('RGB', (512, 512 + title_bar_height))

        # Paste the 4 images on. This should be dynamic but that can be done later :) 
        new_im.paste(images[0], (0,0))
        new_im.paste(images[1], (0,256))
        new_im.paste(images[2], (256,0))
        new_im.paste(images[3], (256,256))


        font = ImageFont.truetype("Helvetica.ttf", 15)
        image_editable = ImageDraw.Draw(new_im)
        image_editable.text((title_padding,512 + title_padding), f'"{prompt}"', (255, 255, 255), font=font)

        new_im.save("result.jpg")

        