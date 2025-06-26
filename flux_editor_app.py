#!/usr/bin/env python3
"""
Comprehensive Flux Image Editor - Streamlit Application
Creates and edits images using various Flux models including text-to-image, 
image-to-image, and image editing with text prompts.
"""

import os
import re
import time
from glob import iglob
from io import BytesIO

import streamlit as st
import torch
from einops import rearrange
from fire import Fire
from PIL import ExifTags, Image
from st_keyup import st_keyup
from torchvision import transforms
from transformers import pipeline

from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, prepare_kontext, unpack
from flux.util import (
    configs,
    embed_watermark,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    track_usage_via_api,
    PREFERED_KONTEXT_RESOLUTIONS,
)

NSFW_THRESHOLD = 0.85

# Set page config
st.set_page_config(
    page_title="Flux Image Editor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource()
def get_models(name: str, device: torch.device, offload: bool, is_schnell: bool):
    """Load and cache the models"""
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu" if offload else device)
    ae = load_ae(name, device="cpu" if offload else device)
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
    return model, ae, t5, clip, nsfw_classifier

def get_image_tensor(uploaded_file) -> torch.Tensor | None:
    """Convert uploaded file to tensor"""
    if uploaded_file is None:
        return None
    image = Image.open(uploaded_file).convert("RGB")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2.0 * x - 1.0),
    ])
    img: torch.Tensor = transform(image)
    return img[None, ...]

def generate_image(model, ae, t5, clip, nsfw_classifier, opts, torch_device, init_image=None, 
                  image2image_strength=0.0, offload=False, is_schnell=False):
    """Generate image using text-to-image or image-to-image"""
    
    if init_image is not None:
        if opts.width and opts.height:
            init_image = torch.nn.functional.interpolate(init_image, (opts.height, opts.width))
        else:
            h, w = init_image.shape[-2:]
            init_image = init_image[..., : 16 * (h // 16), : 16 * (w // 16)]
            opts.height = init_image.shape[-2]
            opts.width = init_image.shape[-1]
        
        if offload:
            ae.encoder.to(torch_device)
        init_image = ae.encode(init_image.to(torch_device))
        if offload:
            ae = ae.cpu()
            torch.cuda.empty_cache()

    # prepare input
    x = get_noise(
        1,
        opts.height,
        opts.width,
        device=torch_device,
        dtype=torch.bfloat16,
        seed=opts.seed,
    )
    
    timesteps = get_schedule(
        opts.num_steps,
        (x.shape[-1] * x.shape[-2]) // 4,
        shift=(not is_schnell),
    )
    
    if init_image is not None:
        t_idx = int((1 - image2image_strength) * opts.num_steps)
        t = timesteps[t_idx]
        timesteps = timesteps[t_idx:]
        x = t * x + (1.0 - t) * init_image.to(x.dtype)

    if offload:
        t5, clip = t5.to(torch_device), clip.to(torch_device)
    inp = prepare(t5=t5, clip=clip, img=x, prompt=opts.prompt)

    # offload TEs to CPU, load model to gpu
    if offload:
        t5, clip = t5.cpu(), clip.cpu()
        torch.cuda.empty_cache()
        model = model.to(torch_device)

    # denoise initial noise
    x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

    # offload model, load autoencoder to gpu
    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x.device)

    # decode latents to pixel space
    x = unpack(x.float(), opts.height, opts.width)
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        x = ae.decode(x)

    if offload:
        ae.decoder.cpu()
        torch.cuda.empty_cache()

    return x

def edit_image_with_kontext(model, ae, t5, clip, nsfw_classifier, prompt, input_image_path, 
                           num_steps, guidance, seed, torch_device, offload=False):
    """Edit image using Kontext model with text prompt"""
    
    if offload:
        t5, clip, ae = t5.to(torch_device), clip.to(torch_device), ae.to(torch_device)
    
    inp, height, width = prepare_kontext(
        t5=t5,
        clip=clip,
        prompt=prompt,
        ae=ae,
        img_cond_path=input_image_path,
        target_width=None,
        target_height=None,
        bs=1,
        seed=seed,
        device=torch_device,
    )
    
    if offload:
        t5, clip = t5.cpu(), clip.cpu()
        torch.cuda.empty_cache()
        model = model.to(torch_device)

    x = get_noise(1, height, width, device=torch_device, dtype=torch.bfloat16, seed=seed)
    timesteps = get_schedule(num_steps, (x.shape[-1] * x.shape[-2]) // 4, shift=True)
    
    x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x.device)

    x = unpack(x.float(), height, width)
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        x = ae.decode(x)

    if offload:
        ae.decoder.cpu()
        torch.cuda.empty_cache()

    return x

def process_generated_image(x, nsfw_classifier, prompt, seed, name, save_samples=False, 
                           add_sampling_metadata=True, output_dir="output"):
    """Process and return the generated image"""
    
    # bring into PIL format
    x = x.clamp(-1, 1)
    x = embed_watermark(x.float())
    x = rearrange(x[0], "c h w -> h w c")
    
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]
    
    if nsfw_score < NSFW_THRESHOLD:
        buffer = BytesIO()
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = f"AI generated;{name};flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = name
        if add_sampling_metadata:
            exif_data[ExifTags.Base.ImageDescription] = prompt
        img.save(buffer, format="jpeg", exif=exif_data, quality=95, subsampling=0)
        
        img_bytes = buffer.getvalue()
        
        if save_samples:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_name = os.path.join(output_dir, "img_{idx}.jpg")
            fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
            if len(fns) > 0:
                idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
            else:
                idx = 0
            fn = output_name.format(idx=idx)
            with open(fn, "wb") as file:
                file.write(img_bytes)
        
        return img, img_bytes, True
    else:
        return None, None, False

@torch.inference_mode()
def main(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    offload: bool = False,
    output_dir: str = "output",
    track_usage: bool = False,
):
    torch_device = torch.device(device)
    
    # Title and description
    st.title("🎨 Flux Image Editor")
    st.markdown("""
    **Generate and edit images with Flux models**
    - **Text-to-Image**: Create images from text descriptions
    - **Image-to-Image**: Transform existing images  
    - **Image Editing**: Edit images with text instructions using Kontext
    """)
    
    # Sidebar for model selection and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        names = list(configs.keys())
        available_models = [name for name in names if name in ["flux-dev", "flux-schnell", "flux-dev-kontext"]]
        
        model_name = st.selectbox("Select Model", available_models, 
                                 help="Choose the Flux model to use")
        
        if not st.checkbox("Load Model", False):
            st.warning("Please check 'Load Model' to continue")
            return
            
        is_schnell = model_name == "flux-schnell"
        is_kontext = model_name == "flux-dev-kontext"
        
        # Load models
        with st.spinner("Loading models..."):
            model, ae, t5, clip, nsfw_classifier = get_models(
                model_name,
                device=torch_device,
                offload=offload,
                is_schnell=is_schnell,
            )
        
        st.success("✅ Models loaded successfully!")
        
        # Common settings
        st.subheader("Generation Settings")
        
        if not is_kontext:
            width = int(16 * (st.number_input("Width", min_value=128, value=1024, step=16) // 16))
            height = int(16 * (st.number_input("Height", min_value=128, value=1024, step=16) // 16))
        else:
            st.info("Resolution will be automatically determined for Kontext model")
            width, height = None, None
            
        num_steps = int(st.number_input("Number of steps", min_value=1, 
                                       value=(4 if is_schnell else 50 if not is_kontext else 30)))
        guidance = float(st.number_input("Guidance", min_value=1.0, 
                                        value=3.5 if not is_kontext else 2.5, 
                                        disabled=is_schnell))
        
        seed_str = st.text_input("Seed", disabled=is_schnell)
        if seed_str.isdecimal():
            seed = int(seed_str)
        else:
            seed = None
            
        save_samples = st.checkbox("Save samples?", not is_schnell)
        add_sampling_metadata = st.checkbox("Add metadata?", True)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            if st.button("Clear Session State"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # Main content area
    mode = st.radio("Choose Mode", 
                   ["Text-to-Image", "Image-to-Image", "Image Editing"] if not is_kontext 
                   else ["Image Editing"], 
                   horizontal=True)
    
    # Initialize session state
    if "seed" not in st.session_state:
        rng = torch.Generator(device="cpu")
        st.session_state.seed = rng.seed()
    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
        
    # Mode-specific interfaces
    if mode == "Text-to-Image" and not is_kontext:
        st.header("📝 Text-to-Image Generation")
        
        # Prompt input
        default_prompt = ("a photo of a forest with mist swirling around the tree trunks. "
                         'The word "FLUX" is painted over it in big, red brush strokes with visible texture')
        prompt = st_keyup("Enter your prompt", value=default_prompt, debounce=300, key="txt2img_prompt")
        
        # Seed controls for schnell
        if is_schnell:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col2:
                if st.button("⬅️ Previous Seed"):
                    if st.session_state.seed > 0:
                        st.session_state.seed -= 1
            with col3:
                if st.button("➡️ Next Seed"):
                    st.session_state.seed += 1
        
        # Generate button
        if is_schnell or st.button("🎨 Generate Image", type="primary"):
            if prompt:
                if is_schnell:
                    actual_seed = st.session_state.seed
                elif seed is None:
                    rng = torch.Generator(device="cpu")
                    actual_seed = rng.seed()
                else:
                    actual_seed = seed
                    
                opts = SamplingOptions(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_steps=num_steps,
                    guidance=guidance,
                    seed=actual_seed,
                )
                
                with st.spinner("Generating image..."):
                    start_time = time.perf_counter()
                    x = generate_image(model, ae, t5, clip, nsfw_classifier, opts, torch_device, 
                                     offload=offload, is_schnell=is_schnell)
                    end_time = time.perf_counter()
                    
                    img, img_bytes, is_safe = process_generated_image(
                        x, nsfw_classifier, prompt, actual_seed, model_name, 
                        save_samples, add_sampling_metadata, output_dir
                    )
                    
                    if is_safe:
                        st.success(f"✅ Generated in {end_time - start_time:.1f}s")
                        result = {
                            "prompt": prompt,
                            "image": img,
                            "seed": actual_seed,
                            "bytes": img_bytes,
                            "mode": "text-to-image"
                        }
                        st.session_state.generated_images.append(result)
                        st.session_state.current_image = result
                        
                        if track_usage:
                            track_usage_via_api(model_name, 1)
                    else:
                        st.warning("⚠️ Generated image may contain NSFW content.")
            else:
                st.warning("Please enter a prompt")
    
    elif mode == "Image-to-Image" and not is_kontext:
        st.header("🖼️ Image-to-Image Generation")
        
        # Image upload
        uploaded_file = st.file_uploader("Upload Input Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            init_image = get_image_tensor(uploaded_file)
            if init_image is not None:
                h, w = init_image.shape[-2:]
                st.write(f"📏 Input image size: {w}x{h} ({h * w / 1e6:.2f}MP)")
                st.image(uploaded_file, caption="Input Image", width=300)
                
                # Settings
                col1, col2 = st.columns(2)
                with col1:
                    prompt = st_keyup("Transformation prompt", value="", debounce=300, key="img2img_prompt")
                with col2:
                    strength = st.slider("Transformation Strength", 0.0, 1.0, 0.8, 
                                       help="Higher values = more transformation")
                
                if st.button("🔄 Transform Image", type="primary"):
                    if prompt:
                        actual_seed = seed if seed is not None else torch.Generator(device="cpu").seed()
                        
                        opts = SamplingOptions(
                            prompt=prompt,
                            width=width,
                            height=height,
                            num_steps=num_steps,
                            guidance=guidance,
                            seed=actual_seed,
                        )
                        
                        with st.spinner("Transforming image..."):
                            start_time = time.perf_counter()
                            x = generate_image(model, ae, t5, clip, nsfw_classifier, opts, torch_device,
                                             init_image=init_image, image2image_strength=strength,
                                             offload=offload, is_schnell=is_schnell)
                            end_time = time.perf_counter()
                            
                            img, img_bytes, is_safe = process_generated_image(
                                x, nsfw_classifier, prompt, actual_seed, model_name,
                                save_samples, add_sampling_metadata, output_dir
                            )
                            
                            if is_safe:
                                st.success(f"✅ Transformed in {end_time - start_time:.1f}s")
                                result = {
                                    "prompt": prompt,
                                    "image": img,
                                    "seed": actual_seed,
                                    "bytes": img_bytes,
                                    "mode": "image-to-image"
                                }
                                st.session_state.generated_images.append(result)
                                st.session_state.current_image = result
                            else:
                                st.warning("⚠️ Generated image may contain NSFW content.")
                    else:
                        st.warning("Please enter a transformation prompt")
            else:
                st.error("Could not process the uploaded image")
        else:
            st.info("👆 Upload an image to start transforming")
    
    elif mode == "Image Editing" or is_kontext:
        st.header("✏️ Image Editing with Text Instructions")
        
        # Image source selection
        image_source = st.radio("Image Source", ["Upload New Image", "Use Generated Image"], horizontal=True)
        
        input_image_path = None
        current_display_image = None
        
        if image_source == "Upload New Image":
            uploaded_file = st.file_uploader("Upload Image to Edit", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                # Save uploaded file temporarily
                temp_dir = "/tmp/flux_editor"
                os.makedirs(temp_dir, exist_ok=True)
                input_image_path = os.path.join(temp_dir, uploaded_file.name)
                with open(input_image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                current_display_image = Image.open(input_image_path)
                st.image(current_display_image, caption="Image to Edit", width=400)
        
        elif image_source == "Use Generated Image":
            if st.session_state.current_image:
                current_display_image = st.session_state.current_image["image"]
                st.image(current_display_image, caption="Current Generated Image", width=400)
                
                # Save current image temporarily
                temp_dir = "/tmp/flux_editor"
                os.makedirs(temp_dir, exist_ok=True)
                input_image_path = os.path.join(temp_dir, "current_image.jpg")
                current_display_image.save(input_image_path, quality=95)
            else:
                st.info("📄 No generated image available. Generate one first or upload an image.")
        
        if input_image_path and os.path.exists(input_image_path):
            # Editing interface
            st.subheader("Edit Instructions")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                edit_prompt = st_keyup("What would you like to change?", 
                                     value="", debounce=300, key="edit_prompt",
                                     placeholder="e.g., 'replace the sky with a sunset', 'add flowers to the field'")
            
            with col2:
                if st.button("✨ Apply Edit", type="primary"):
                    if edit_prompt:
                        actual_seed = seed if seed is not None else torch.Generator(device="cpu").seed()
                        
                        with st.spinner("Editing image..."):
                            start_time = time.perf_counter()
                            x = edit_image_with_kontext(
                                model, ae, t5, clip, nsfw_classifier, edit_prompt,
                                input_image_path, num_steps, guidance, actual_seed,
                                torch_device, offload=offload
                            )
                            end_time = time.perf_counter()
                            
                            img, img_bytes, is_safe = process_generated_image(
                                x, nsfw_classifier, edit_prompt, actual_seed, model_name,
                                save_samples, add_sampling_metadata, output_dir
                            )
                            
                            if is_safe:
                                st.success(f"✅ Edited in {end_time - start_time:.1f}s")
                                result = {
                                    "prompt": edit_prompt,
                                    "image": img,
                                    "seed": actual_seed,
                                    "bytes": img_bytes,
                                    "mode": "image-editing",
                                    "original_image": current_display_image
                                }
                                st.session_state.generated_images.append(result)
                                st.session_state.current_image = result
                                
                                if track_usage:
                                    track_usage_via_api(model_name, 1)
                            else:
                                st.warning("⚠️ Edited image may contain NSFW content.")
                    else:
                        st.warning("Please enter editing instructions")
            
            # Quick edit suggestions
            if st.button("🎯 Quick Suggestions"):
                suggestions = [
                    "change the lighting to golden hour",
                    "add dramatic clouds to the sky", 
                    "make it look like a painting",
                    "add snow to the scene",
                    "change the season to autumn",
                    "make it look futuristic"
                ]
                st.write("**💡 Try these editing ideas:**")
                for suggestion in suggestions:
                    if st.button(f"'{suggestion}'", key=f"suggestion_{suggestion}"):
                        st.session_state.edit_prompt = suggestion
                        st.rerun()
        else:
            st.info("👆 Upload an image or generate one first to start editing")
    
    # Display results
    if st.session_state.current_image:
        st.header("🖼️ Latest Result")
        
        result = st.session_state.current_image
        
        # Show before/after for editing
        if result["mode"] == "image-editing" and "original_image" in result:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Before")
                st.image(result["original_image"], caption="Original", use_container_width=True)
            with col2:
                st.subheader("After")
                st.image(result["image"], caption=f"Edited: {result['prompt']}", use_container_width=True)
        else:
            st.image(result["image"], caption=result["prompt"], use_container_width=True)
        
        # Download and info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "💾 Download Image",
                result["bytes"],
                file_name=f"flux_{result['mode']}_{result['seed']}.jpg",
                mime="image/jpeg",
            )
        with col2:
            st.write(f"**Seed:** {result['seed']}")
        with col3:
            st.write(f"**Mode:** {result['mode']}")
            
        # Continue editing
        if result["mode"] != "image-editing" and not is_kontext:
            if st.button("✏️ Continue Editing This Image"):
                st.session_state.edit_mode = True
                st.rerun()
    
    # Image history
    if len(st.session_state.generated_images) > 1:
        st.header("📚 Generation History")
        
        # Show recent images in a grid
        cols = st.columns(min(4, len(st.session_state.generated_images)))
        for i, result in enumerate(reversed(st.session_state.generated_images[-8:])):  # Show last 8
            with cols[i % 4]:
                st.image(result["image"], caption=f"{result['mode']}: {result['prompt'][:50]}...", 
                        use_container_width=True)
                if st.button(f"🔄 Use This", key=f"use_{i}"):
                    st.session_state.current_image = result
                    st.rerun()

def app():
    Fire(main)

if __name__ == "__main__":
    app()