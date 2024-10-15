import torch
import time
import os
import folder_paths
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, ControlNetModel

from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose import DwposeDetector
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.open_pose import OpenposeDetector
from .pipeline import PhotoMakerStableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from .style_template import styles
from PIL import Image
import numpy as np
from .insightface_package import FaceAnalysis2, analyze_faces
from .pipeline_controlnet import PhotoMakerStableDiffusionXLControlNetPipeline
import comfy.controlnet

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


def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + ' ' + negative


class BaseModelLoader_fromhub_Node_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": ("STRING", {"default": "SG161222/RealVisXL_V3.0"})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "📷PhotoMakerV2"

    def load_model(self, base_model_path):
        # Code to load the base model
        pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(device)
        return [pipe]


class BaseModelLoader_local_Node_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "cn_name": (folder_paths.get_filename_list("controlnet"),)
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "📷PhotoMakerV2"

    def load_model(self, ckpt_name, cn_name):
        # Code to load the base model
        if not ckpt_name:
            raise ValueError("Please provide the ckpt_name parameter with the name of the checkpoint file.")

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        cn_path = folder_paths.get_full_path("controlnet", cn_name)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found.")
        # controlnet = comfy.controlnet.load_controlnet(cn_path)
        controlnet_pose_model = "xinsir/controlnet-openpose-sdxl-1.0"
        controlnet = ControlNetModel.from_pretrained(
            controlnet_pose_model, torch_dtype=torch.float16,
        ).to(device)
        pipe = PhotoMakerStableDiffusionXLControlNetPipeline.from_single_file(
            pretrained_model_link_or_path=ckpt_path,
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to(device)
        return [pipe]


class PhotoMakerAdapterLoader_fromhub_Node_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"default": "TencentARC/PhotoMaker"}),
                "filename": ("STRING", {"default": "photomaker-v2.bin"}),
                "pipe": ("MODEL",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_photomaker_adapter"
    CATEGORY = "📷PhotoMakerV2"

    def load_photomaker_adapter(self, repo_id, filename, pipe):
        # 使用hf_hub_download方法获取PhotoMaker文件的路径
        photomaker_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model"
        )

        # 加载PhotoMaker检查点
        pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word="img",
            pm_version='v2'
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

        return [pipe]


class PhotoMakerAdapterLoader_local_Node_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": (folder_paths.get_filename_list("photomaker"),),
                "pipe": ("MODEL",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_photomaker_adapter"
    CATEGORY = "📷PhotoMakerV2"

    def load_photomaker_adapter(self, filename, pipe):
        # 拼接完整的模型路径
        # photomaker_path = os.path.join(pm_model_path, filename)
        photomaker_path = folder_paths.get_full_path("photomaker", filename)

        # 加载PhotoMaker检查点
        pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word="img",
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

        return [pipe]


class LoRALoader_Node_Zho:
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


class ImagePreprocessingNode_Zho:
    def __init__(self, ref_image=None, ref_images_path=None, mode="direct_Input"):
        self.ref_image = ref_image
        self.ref_images_path = ref_images_path
        self.mode = mode

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_images_path": ("STRING", {"default": "path/to/images"}),  # 图像文件夹路径
                "mode": (["direct_Input", "path_Input"], {"default": "direct_Input"})  # 选择模式
            },
            "optional": {
                "ref_image": ("IMAGE",)  # 直接输入图像（可选）
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess_image"
    CATEGORY = "📷PhotoMakerV2"

    def preprocess_image(self, ref_image=None, ref_images_path=None, mode="direct_Input"):
        # 使用传入的参数更新类属性
        ref_image = ref_image if ref_image is not None else ref_image
        ref_images_path = ref_images_path if ref_images_path is not None else ref_images_path
        mode = mode

        if mode == "direct_Input" and ref_image is not None:
            # 直接图像处理
            pil_images = []
            for image in ref_image:
                image_np = (255. * image.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
                pil_images.append(pil_image)
            return [pil_images]
        elif mode == "path_Input":
            # 路径输入图像
            image_basename_list = os.listdir(ref_images_path)
            image_path_list = [
                os.path.join(ref_images_path, basename)
                for basename in image_basename_list
                if
                not basename.startswith('.') and basename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
            ]
            return [load_image(image_path) for image_path in image_path_list]
        else:
            raise ValueError("Invalid mode. Choose 'direct_Input' or 'path_Input'.")


'''
class CompositeImageGenerationNode_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "sci-fi, closeup portrait photo of a man img wearing the sunglasses in Iron man suit, face, slim body, high quality, film grain", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth", "multiline": True}),
                "style_name": (STYLE_NAMES, {"default": DEFAULT_STYLE_NAME}),
                "style_strength_ratio": ("INT", {"default": 20, "min": 1, "max": 50, "display": "slider"}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "display": "slider"}),
                "guidance_scale": ("FLOAT", {"default": 5, "min": 0, "max": 10}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32, "display": "slider"}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32, "display": "slider"}), 
                "pipe": ("MODEL",),
                "pil_image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "📷PhotoMakerV2"

    def generate_image(self, style_name, style_strength_ratio, steps, seed, prompt, negative_prompt, guidance_scale, batch_size, pil_image, pipe, width, height):
        # Code for the remaining process including style template application, merge step calculation, etc.
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)
        
        start_merge_step = int(float(style_strength_ratio) / 100 * steps)
        if start_merge_step > 30:
            start_merge_step = 30

        generator = torch.Generator(device=device).manual_seed(seed)

        output = pipe(
            prompt=prompt,
            input_id_images=[pil_image],
            negative_prompt=negative_prompt,
            num_images_per_prompt=batch_size,
            num_inference_steps=steps,
            start_merge_step=start_merge_step,
            generator=generator,
            guidance_scale=guidance_scale, 
            width=width,
            height=height,
            return_dict=False
        )

        # 检查输出类型并相应处理
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
'''


# 拆分生成块
class Prompt_Style_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "sci-fi, closeup portrait photo of a man img wearing the sunglasses in Iron man suit, face, slim body, high quality, film grain",
                    "multiline": True}),
                "negative_prompt": ("STRING", {
                    "default": "asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth",
                    "multiline": True}),
                "style_name": (STYLE_NAMES, {"default": DEFAULT_STYLE_NAME})
            }
        }

    RETURN_TYPES = ('STRING', 'STRING',)
    RETURN_NAMES = ('positive_prompt', 'negative_prompt',)
    FUNCTION = "prompt_style"
    CATEGORY = "📷PhotoMakerV2"

    def prompt_style(self, style_name, prompt, negative_prompt):
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

        return prompt, negative_prompt


class NEWCompositeImageGenerationNode_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True, "forceInput": True}),
                "negative": ("STRING", {"multiline": True, "forceInput": True}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4, "display": "slider"}),
                "style_strength_ratio": ("INT", {"default": 20, "min": 1, "max": 50, "display": "slider"}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "display": "slider"}),
                "guidance_scale": ("FLOAT", {"default": 5, "min": 0, "max": 10, "display": "slider"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32, "display": "slider"}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32, "display": "slider"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "pipe": ("MODEL",),
                "pil_image": ("IMAGE",),
                "pose_image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "📷PhotoMakerV2"

    def generate_image(self, steps, seed, positive, negative, style_strength_ratio, guidance_scale, batch_size,
                       pil_image, pose_image, pipe, width, height):
        # Code for the remaining process including style template application, merge step calculation, etc.

        start_merge_step = int(float(style_strength_ratio) / 100 * steps)
        if start_merge_step > 30:
            start_merge_step = 30

        generator = torch.Generator(device=device).manual_seed(seed)

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
       # openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        #pose_image = openpose(pose_image, detect_resolution=512, image_resolution=1024)

        output = pipe(
            prompt=positive,
            input_id_images=pil_images,
            id_embeds=id_embeds,
            negative_prompt=negative,
            num_images_per_prompt=batch_size,
            num_inference_steps=steps,
            start_merge_step=10,
            controlnet_conditioning_scale=1,
            image=pose_image,
            generator=generator,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            return_dict=False
        )

        # 检查输出类型并相应处理
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

class NEWCompositeImageGenerationNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": (folder_paths.get_filename_list("photomaker"),),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "cn_name": (folder_paths.get_filename_list("controlnet"),),
                "prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "negative_prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "pil_image": ("IMAGE",),
                "pose_image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "📷PhotoMakerV2"

    def generate_image(self,prompt, negative_prompt,
                       pil_image, pose_image,filename,ckpt_name,cn_name):
        # Code for the remaining process including style template application, merge step calculation, etc.

        controlnet_pose_model = "xinsir/controlnet-openpose-sdxl-1.0"
        ckpt_path = folder_paths.get_full_path("controlnet", cn_name)
        controlnet_pose = ControlNetModel.from_pretrained(
            ckpt_path, torch_dtype=torch_dtype,
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
    "BaseModel_Loader_fromhub": BaseModelLoader_fromhub_Node_Zho,
    "BaseModel_Loader_local": BaseModelLoader_local_Node_Zho,
    "PhotoMakerAdapter_Loader_fromhub": PhotoMakerAdapterLoader_fromhub_Node_Zho,
    "PhotoMakerAdapter_Loader_local": PhotoMakerAdapterLoader_local_Node_Zho,
    "LoRALoader": LoRALoader_Node_Zho,
    "Ref_Image_Preprocessing": ImagePreprocessingNode_Zho,
    "Prompt_Styler": Prompt_Style_Zho,
    "NEW_PhotoMakerV2_Generation": NEWCompositeImageGenerationNode_Zho,
    "NEW_PhotoMakerV2_Generation1": NEWCompositeImageGenerationNode,
    # "PhotoMaker_Generation": CompositeImageGenerationNode_Zho
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BaseModel_Loader_fromhub": "📷PhotoMaker Base Model Loader from hub🤗",
    "BaseModel_Loader_local": "📷PhotoMakerV2 Base Model Loader",
    "PhotoMakerAdapter_Loader_fromhub": "📷PhotoMaker Adapter Loader from hub🤗",
    "PhotoMakerAdapter_Loader_local": "📷PhotoMakerV2 Adapter Loader",
    "LoRALoader": "📷PhotoMaker LoRA Loader",
    "Ref_Image_Preprocessing": "📷PhotoMakerV2 Ref Image Preprocessing",
    "Prompt_Styler": "📷PhotoMakerV2 Prompt Styler",
    "NEW_PhotoMakerV2_Generation": "📷NEW PhotoMakerV2 Generation",
    "NEW_PhotoMakerV2_Generation1": "📷NEW PhotoMakerV2 Generation1",
    # "PhotoMaker_Generation": "📷PhotoMakerV2 Generation"
}
