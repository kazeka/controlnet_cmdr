#!/bin/env python3

import tempfile

import fire
import requests

import numpy as np
import torch

import cv2
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from transformers import DPTImageProcessor, DPTForDepthEstimation


# OPTIMIZATIONS
torch.backends.cuda.matmul.allow_tf32 = True


class Commander:
    @staticmethod
    def _save_to_tmp(image: Image) -> str:
        fpath = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            fpath = temp_file.name
        return fpath

    def canny(
        self,
        url: str="https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
        prompt: str="futuristic-looking woman") -> int:
        '''ControlNet with Canny edges to guide Stable Diffusion; print path to result'''

        # TODO: add image size check
        image = np.array(load_image(url))

        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)

        # get canny image
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
            )

        # speed up diffusion process with faster scheduler and memory optimization
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # TODO: add check if xformers is not installed use pipe.enable_attention_slicing()
        # pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing(1)
        pipe.enable_model_cpu_offload()

        # generate image
        # generator = torch.manual_seed(0)
        # image = pipe(
        #     prompt, num_inference_steps=20, generator=generator, image=canny_image
        # ).images[0]
        image = pipe(prompt, num_inference_steps=20, image=canny_image).images[0]

        print(self._save_to_tmp(image))

    def depth(
        self,
        url: str="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        prompt: str="elf woman in the woods") -> int:
        '''ControlNet with depth information to guide Stable Diffusion; print path to result'''

        # TODO: add image size check, etc. and refactor out
        image = load_image(url)

        processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

        # prepare image for the model
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # fetch depth image
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth_image = Image.fromarray(formatted)

        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
            )

        # speed up diffusion process with faster scheduler and memory optimization
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # TODO: add check if xformers is not installed use pipe.enable_attention_slicing()
        # pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing(1)
        pipe.enable_model_cpu_offload()

        # generate image
        # generator = torch.manual_seed(0)
        # image = pipe(
        #     prompt, num_inference_steps=20, generator=generator, image=canny_image
        # ).images[0]
        image = pipe(prompt, num_inference_steps=20, image=depth_image).images[0]

        print(self._save_to_tmp(image))


if __name__ == '__main__':
    fire.Fire(Commander)