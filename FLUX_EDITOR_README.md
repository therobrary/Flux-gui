# 🎨 Flux Image Editor - Comprehensive Streamlit App

A unified Streamlit application for generating and editing images using Flux models. This app combines text-to-image generation, image-to-image transformation, and advanced image editing capabilities in a single, intuitive interface.

![Flux Image Editor Interface](https://github.com/user-attachments/assets/20d86a80-2ac6-4605-a92b-1d2d6400e43b)

## Features

### 🖼️ Three Generation Modes

1. **Text-to-Image**: Create images from text descriptions
   - Uses `flux-dev` or `flux-schnell` models
   - Adjustable resolution, steps, and guidance
   - Seed control for reproducible results

2. **Image-to-Image**: Transform existing images
   - Upload an image and provide transformation instructions
   - Adjustable transformation strength
   - Preserves image structure while applying changes

3. **Image Editing**: Edit images with text instructions
   - Uses `flux-dev-kontext` model for precise editing
   - Natural language editing commands
   - Iterative editing workflow

### ✨ Key Features

- **Unified Interface**: All Flux capabilities in one app
- **Model Selection**: Choose between flux-dev, flux-schnell, and flux-dev-kontext
- **Interactive Editing**: Continue editing generated images
- **Generation History**: Track and reuse previous generations
- **Download Support**: Save images with metadata
- **NSFW Filtering**: Built-in content safety
- **Responsive Design**: Clean, modern interface

## Installation & Setup

### Prerequisites

```bash
# Clone the repository
cd $HOME && git clone https://github.com/therobrary/Flux-gui
cd Flux-gui

# Install dependencies
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"

# Install additional dependencies for the editor app
pip install streamlit streamlit-keyup streamlit-drawable-canvas torchvision
```

### Running the Application

```bash
# Start the Flux Image Editor
streamlit run flux_editor_app.py
```

The app will be available at `http://localhost:8501`

## Usage Guide

### 1. Getting Started

1. **Select Model**: Choose from available Flux models in the sidebar
2. **Load Model**: Check the "Load Model" checkbox to initialize
3. **Choose Mode**: Select your desired generation mode

### 2. Text-to-Image Generation

```python
# Example prompts:
"a majestic mountain landscape at sunset with golden clouds"
"a futuristic city with flying cars and neon lights"
"a cat wearing a wizard hat, digital art style"
```

**Settings:**
- **Width/Height**: Image dimensions (must be multiples of 16)
- **Steps**: Number of denoising steps (4 for flux-schnell, 50+ for flux-dev)
- **Guidance**: How closely to follow the prompt (higher = more adherence)
- **Seed**: For reproducible results

### 3. Image-to-Image Transformation

1. **Upload Image**: Drag and drop or click to upload
2. **Enter Prompt**: Describe the transformation you want
3. **Adjust Strength**: Control how much the image changes (0.8 recommended)
4. **Generate**: Click "Transform Image"

```python
# Example transformation prompts:
"turn this photo into a watercolor painting"
"make this landscape look like it's in winter"
"add dramatic lighting to this portrait"
```

### 4. Image Editing with Kontext

This is the most powerful feature - edit specific parts of images using natural language:

1. **Source Image**: Upload new image or use a generated one
2. **Edit Instructions**: Describe what you want to change
3. **Apply Edit**: The Kontext model will make precise changes

```python
# Example editing instructions:
"replace the sky with a dramatic sunset"
"add flowers to the foreground"
"change the person's shirt to red"
"remove the building in the background"
"make the lighting more dramatic"
```

**Quick Suggestions**: The app provides common editing ideas to get you started.

### 5. Iterative Workflow

- **Continue Editing**: Use generated images as input for further editing
- **History**: Access previous generations from the history panel
- **Download**: Save images with embedded metadata

## Model Comparison

| Model | Best For | Speed | Quality | Editing |
|-------|----------|-------|---------|---------|
| `flux-schnell` | Quick generations | ⚡ Fast | Good | ❌ |
| `flux-dev` | High-quality images | 🐌 Slower | Excellent | ❌ |
| `flux-dev-kontext` | Image editing | 🐌 Slower | Excellent | ✅ |

## Tips for Best Results

### Text-to-Image
- Be specific and descriptive in your prompts
- Include style information ("photorealistic", "digital art", "painting")
- Mention lighting, composition, and mood
- Use negative prompts for unwanted elements

### Image-to-Image
- Start with transformation strength around 0.7-0.8
- Lower values preserve more of the original image
- Higher values create more dramatic changes

### Image Editing
- Be specific about what to change: "replace X with Y"
- Mention spatial relationships: "in the background", "on the left"
- Use action words: "add", "remove", "change", "replace"
- Try multiple small edits rather than one large change

## Advanced Features

### Seed Control
- Use specific seeds for reproducible results
- Increment/decrement seeds to explore variations
- Copy seeds from successful generations

### Batch Processing
- Generate multiple variations by changing seeds
- Compare different guidance values
- Test various prompt modifications

### Quality Settings
- Adjust steps based on your time/quality needs
- Higher guidance for more prompt adherence
- Use appropriate resolution for your use case

## Technical Details

### Supported Models
- **FLUX.1 [schnell]**: Fast text-to-image (Apache 2.0 license)
- **FLUX.1 [dev]**: High-quality text-to-image (Non-commercial license)
- **FLUX.1 Kontext [dev]**: Image editing (Non-commercial license)

### System Requirements
- **GPU**: CUDA-compatible GPU recommended
- **RAM**: 16GB+ system RAM
- **VRAM**: 8GB+ GPU memory for optimal performance
- **Storage**: 20GB+ for model weights

### Performance Optimization
- **Model Offloading**: Automatically moves models between GPU/CPU
- **Memory Management**: Efficient VRAM usage
- **Caching**: Models loaded once and cached

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce image resolution
- Enable model offloading in settings
- Close other GPU-intensive applications

**"Model not found"**
- Ensure HuggingFace authentication is set up
- Check internet connection for model downloads
- Verify model names in configs

**"Generation too slow"**
- Use flux-schnell for faster results
- Reduce number of steps
- Lower image resolution

### Getting Help

1. Check the [main repository documentation](../README.md)
2. Review model-specific guides in the `docs/` folder
3. Submit issues to the GitHub repository

## License

This application uses Flux models with different licenses:
- **flux-schnell**: Apache 2.0 (commercial use allowed)
- **flux-dev** and **flux-dev-kontext**: Non-commercial license

See the `model_licenses/` directory for full license texts.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## Acknowledgments

Built on the excellent Flux models by [Black Forest Labs](https://bfl.ai) and powered by:
- [Streamlit](https://streamlit.io) for the web interface
- [PyTorch](https://pytorch.org) for model inference
- [Hugging Face](https://huggingface.co) for model hosting