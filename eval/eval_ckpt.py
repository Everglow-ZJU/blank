from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from oft_utils.unet_2d_conditional import UNet2DConditionalModel
import torch
from transformers import PretrainedConfig
import numpy as np
from PIL import Image
import os
import argparse

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")
def get_prompt_and_name(idx):
    prompt_idx = idx % 25
    class_idx = idx // 25

    unique_token = "qwe"
    subject_names = [
        "backpack", "backpack_dog", "bear_plushie", "berry_bowl", "can",
        "candle", "cat", "cat2", "clock", "colorful_sneaker",
        "dog", "dog2", "dog3", "dog5", "dog6",
        "dog7", "dog8", "duck_toy", "fancy_boot", "grey_sloth_plushie",
        "monster_toy", "pink_sunglasses", "poop_emoji", "rc_car", "red_cartoon",
        "robot_toy", "shiny_sneaker", "teapot", "vase", "wolf_plushie",
    ]

    class_tokens = [
        "backpack", "backpack", "stuffed animal", "bowl", "can",
        "candle", "cat", "cat", "clock", "sneaker",
        "dog", "dog", "dog", "dog", "dog",
        "dog", "dog", "toy", "boot", "stuffed animal",
        "toy", "glasses", "toy", "toy", "cartoon",
        "toy", "sneaker", "teapot", "vase", "stuffed animal",
    ]

    class_token = class_tokens[class_idx]
    selected_subject = subject_names[class_idx]

    if class_idx in [0, 1, 2, 3, 4, 5, 8, 9, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
        prompt_list = [
            f"a {unique_token} {class_token} in the jungle",
            f"a {unique_token} {class_token} in the snow",
            f"a {unique_token} {class_token} on the beach",
            f"a {unique_token} {class_token} on a cobblestone street",
            f"a {unique_token} {class_token} on top of pink fabric",
            f"a {unique_token} {class_token} on top of a wooden floor",
            f"a {unique_token} {class_token} with a city in the background",
            f"a {unique_token} {class_token} with a mountain in the background",
            f"a {unique_token} {class_token} with a blue house in the background",
            f"a {unique_token} {class_token} on top of a purple rug in a forest",
            f"a {unique_token} {class_token} with a wheat field in the background",
            f"a {unique_token} {class_token} with a tree and autumn leaves in the background",
            f"a {unique_token} {class_token} with the Eiffel Tower in the background",
            f"a {unique_token} {class_token} floating on top of water",
            f"a {unique_token} {class_token} floating in an ocean of milk",
            f"a {unique_token} {class_token} on top of green grass with sunflowers around it",
            f"a {unique_token} {class_token} on top of a mirror",
            f"a {unique_token} {class_token} on top of the sidewalk in a crowded street",
            f"a {unique_token} {class_token} on top of a dirt road",
            f"a {unique_token} {class_token} on top of a white rug",
            f"a red {unique_token} {class_token}",
            f"a purple {unique_token} {class_token}",
            f"a shiny {unique_token} {class_token}",
            f"a wet {unique_token} {class_token}",
            f"a cube shaped {unique_token} {class_token}",
        ]
    else:
        prompt_list = [
            f"a {unique_token} {class_token} in the jungle",
            f"a {unique_token} {class_token} in the snow",
            f"a {unique_token} {class_token} on the beach",
            f"a {unique_token} {class_token} on a cobblestone street",
            f"a {unique_token} {class_token} on top of pink fabric",
            f"a {unique_token} {class_token} on top of a wooden floor",
            f"a {unique_token} {class_token} with a city in the background",
            f"a {unique_token} {class_token} with a mountain in the background",
            f"a {unique_token} {class_token} with a blue house in the background",
            f"a {unique_token} {class_token} on top of a purple rug in a forest",
            f"a {unique_token} {class_token} wearing a red hat",
            f"a {unique_token} {class_token} wearing a santa hat",
            f"a {unique_token} {class_token} wearing a rainbow scarf",
            f"a {unique_token} {class_token} wearing a black top hat and a monocle",
            f"a {unique_token} {class_token} in a chef outfit",
            f"a {unique_token} {class_token} in a firefighter outfit",
            f"a {unique_token} {class_token} in a police outfit",
            f"a {unique_token} {class_token} wearing pink glasses",
            f"a {unique_token} {class_token} wearing a yellow shirt",
            f"a {unique_token} {class_token} in a purple wizard outfit",
            f"a red {unique_token} {class_token}",
            f"a purple {unique_token} {class_token}",
            f"a shiny {unique_token} {class_token}",
            f"a wet {unique_token} {class_token}",
            f"a cube shaped {unique_token} {class_token}",
        ]

    validation_prompt = prompt_list[prompt_idx]
    name = f"{selected_subject}-{prompt_idx}"
    weight_folder_name=f"{selected_subject}-{0}"
    return validation_prompt,name,weight_folder_name
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="a eval script.")
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="idx",
    )   
    parser.add_argument( 
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=8,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )    
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--epoch",
        type=int,
        default=4,
        help="epoch of image to be tested",
    )
    parser.add_argument(
        "--output_root_path",
        type=str,
        default=None,
        required=True,
        help="root Path to lora/oft/fft validation output,like"./log_lora",
    )
    parser.add_argument( 
        "--weight_filename",
        type=str,
        default=None,
        required=True,
        help="weight file name",
    )   
def main(args):
    idx=args.idx
    model_base=args.pretrained_model_name_or_path
    epoch=args.epoch
    
    num_validation_images=args.num_validation_images
    seed=args.seed
    
    device="cuda"
    # unique_token="qwe"
    
    validation_prompt,name,weight_folder_name=get_prompt_and_name(idx)
    weight_path=os.path.join(args.output_root_path,weight_folder_name,str(epoch),args.weight_filename) #model is under this directory
    
    text_encoder_cls =import_model_class_from_model_name_or_path(model_base, None)
    text_encoder = text_encoder_cls.from_pretrained(model_base, subfolder="text_encoder",revision=None)
    vae=AutoencoderKL.from_pretrained(model_base, subfolder="vae",revision=None)
    unet=UNet2DConditionalModel.from_pretrained(model_base, subfolder="unet",revision=None)
    weight_dtype=torch.float32
    unet.to(device=device, dtype=weight_dtype)
    vae.to(device=device, dtype=weight_dtype)
    text_encoder.to(device=device, dtype=weight_dtype) 

    pipe=DiffusionPipeline(
        model_base,
        unet=unet,
        text_encoder=text_encoder,
        revision=None,
        torch_dtype=weight_dtype,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe=pipe.to(device)

    pipe.unet.load_attn_procs(weight_path,use_safetensors=False) 
    pipe.safety_checker = None
    pipe.requires_safety_checker =False

    
    output_dir=os.path.join(args.output_root_path,name,str(epoch))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #set seed
    if seed is not None:
        generator=torch.Generator(device).manual_seed(seed)

    #save pics
    images=[
        pipe(validation_prompt,num_inference_steps=25,generator=generator).images[0]
        for _ in range(num_validation_images)
    ]
    for i,image in enumerate(images):
        np_image = np.array(image)
        pil_image = Image.fromarray(np_image)
        pil_image.save(os.path.join(output_dir,f"image_{i}.png"))
if __name__ == "__main__":
    args = parse_args()
    main(args)
