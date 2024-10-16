import os
import time

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from controlnet_aux import OpenposeDetector
from diffusers import EulerDiscreteScheduler, ControlNetModel
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download

import folder_paths
from .insightface_package import FaceAnalysis2, analyze_faces
from .pipeline import PhotoMakerStableDiffusionXLPipeline
from .pipeline_controlnet import PhotoMakerStableDiffusionXLControlNetPipeline
from .style_template import styles

# global variable
# photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch_dtype = torch.float16
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Photographic (Default)"

face_detector = FaceAnalysis2(providers=['CoreMLExecutionProvider'], allowed_modules=['detection', 'recognition'])
face_detector.prepare(ctx_id=0, det_size=(640, 640))



class LoadLora:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "lora_weight": ("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "display": "slider"}),
                "pipe": ("MODEL",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "📷PhotoMakerV2"

    def load_lora(self, lora_name, lora_weight, pipe):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_name_processed = os.path.basename(lora_path).replace(".safetensors", "")

        # 解融合之前的 LoRA
        pipe.unfuse_lora()

        # 卸载之前加载的 LoRA 权重
        pipe.unload_lora_weights()

        # 重新加载新的 LoRA 权重
        unique_adapter_name = f"photomaker_{int(time.time())}"
        pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=os.path.basename(lora_path),
                               adapter_name=unique_adapter_name)

        # 设置适配器和权重
        adapter_weights = [1.0, lora_weight]
        pipe.set_adapters(["photomaker", unique_adapter_name], adapter_weights=adapter_weights)

        # 融合 LoRA
        pipe.fuse_lora()

        return [pipe]

class Prompt_Process:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": " portrait photo of a man img ",
                    "multiline": True}),
                "negative_prompt": ("STRING", {
                    "default": " worst quality, low quality",
                    "multiline": True}),
                "style_name": (STYLE_NAMES, {"default": DEFAULT_STYLE_NAME})
            }
        }

    RETURN_TYPES = ('STRING', 'STRING',)
    RETURN_NAMES = ('positive_prompt', 'negative_prompt',)
    FUNCTION = "prompt_process"
    CATEGORY = "HyPhotoMaker"

    def prompt_process(self, style_name, prompt, negative_prompt):
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)
        return prompt, negative_prompt


class PortraitPhotographyMakeCn:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": (folder_paths.get_filename_list("photomaker"),),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "lora": (folder_paths.get_filename_list("loras"),),
                "cn_name": ("STRING", {"multiline": True, "forceInput": True},),
                "prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "negative_prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "pil_image": ("IMAGE",),
                "pose_image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "HyPhotoMaker"

    def generate_image(self,prompt, negative_prompt,
                       pil_image, pose_image,filename,ckpt_name,cn_name):
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        # 获取当前文件所在的目录
        #current_file_path = os.path.abspath(__file__)
        # current_directory = os.path.dirname(current_file_path)+"/examples/pos_ref.png"
        # pose_image = load_image(
        #     current_directory
        # )
        image_np = (255. * pose_image.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8)
        pose_image = Image.fromarray(image_np)
        #pose_image  =tensor_to_image(pose_image.squeeze(0))
        pose_image = openpose(pose_image, detect_resolution=512, image_resolution=1024)
        controlnet_pose = ControlNetModel.from_pretrained(
            cn_name, torch_dtype=torch_dtype,
        ).to(device)
        # download an image

        # initialize the models and pipeline
        controlnet_conditioning_scale = 1.0  # recommended for good generalization
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        ### Load base model
        pipe = PhotoMakerStableDiffusionXLControlNetPipeline.from_single_file(
            ckpt_path,
            controlnet=controlnet_pose,
            torch_dtype=torch_dtype,
        ).to(device)
        photomaker_path = folder_paths.get_full_path("photomaker", filename)
        ### Load PhotoMaker checkpoint
        photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker-V2", filename="photomaker-v2.bin",
                                          repo_type="model")
        pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word="img"  # define the trigger word
        )
        pipe.fuse_lora()
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

        pipe.enable_model_cpu_offload()

        ### define the input ID images
        id_embed_list = []
        if (not isinstance(pil_image, list)):
            pil_image = [pil_image]
        pil_images = []
        for img in pil_image:
            if isinstance(img, torch.Tensor):
                image_np = (255. * img.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(image_np)
            pil_images.append(img)
            img = load_image(img)
            img = np.array(img)
            img = img[:, :, ::-1]
            faces = analyze_faces(face_detector, img)
            if len(faces) > 0:
                id_embed_list.append(torch.from_numpy((faces[0]['embedding'])))

        if len(id_embed_list) == 0:
            raise ValueError(f"No face detected in input image pool")

        id_embeds = torch.stack(id_embed_list)
        pose_image = load_image(
            tensor_to_image(pose_image.squeeze(0))
        )
        # generate image
        output = pipe(
            prompt,
            negative_prompt=negative_prompt,
            input_id_images=pil_images,
            id_embeds=id_embeds,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            image=pose_image,
            num_images_per_prompt=2,
            start_merge_step=10,
        )
        if isinstance(output, tuple):
            # 当返回的是元组时，第一个元素是图像列表
            images_list = output[0]
        else:
            # 如果返回的是 StableDiffusionXLPipelineOutput，需要从中提取图像
            images_list = output.images

            # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
        images_tensors = []
        for img in images_list:
            # 将 PIL.Image 转换为 numpy.ndarray
            img_array = np.array(img)
            # 转换 numpy.ndarray 为 torch.Tensor
            img_tensor = torch.from_numpy(img_array).float() / 255.
            # 转换图像格式为 CHW (如果需要)
            if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            # 添加批次维度并转换为 NHWC
            img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
            images_tensors.append(img_tensor)

        if len(images_tensors) > 1:
            output_image = torch.cat(images_tensors, dim=0)
        else:
            output_image = images_tensors[0]

        return (output_image,)


NODE_CLASS_MAPPINGS = {
    "Load_Lora": LoadLora,
    "Prompt_Process": Prompt_Process,
    "portrait_photography_make_cn": PortraitPhotographyMakeCn,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load_Lora": "LOAD LORA",
    "Prompt_Process": "Prompt Process",
    "portrait_photography_make_cn": "portrait photography make cn",
}
