"""Gradio app to test prompting and segmenting images with different encoders.

Make sure you have gradio installed `pip install gradio`

The app uses the rayfronts hydra configs to initialize the encoder.

Typical Usage:
  python scripts/encoder_semseg_app.py encoder=naradio encoder.model_version=radio_v2.5-b
"""

import sys
import os
import logging
import base64
from io import BytesIO
import colorsys
from functools import partial
from dataclasses import dataclass

import gradio as gr
from PIL import Image
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import hydra
from hydra.core.config_store import ConfigStore
import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
import torch.nn.functional as F
from einops import rearrange

sys.path.insert(
  0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
import rayfronts.utils as utils

logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
  # Chunk size to compute cos similarity. Reduce if getting OOM error.
  chunk_size: int = 10000

cs = ConfigStore.instance()
cs.store(name="extras", node=AppConfig)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Store prompts and colors
prompt_list = []
color_list = []

def apply_colormap(image: np.ndarray, cmap_name='turbo') -> np.ndarray:
  """Apply a colormap to a grayscale image and return an RGB uint8 image."""
  # Ensure image is normalized to [0, 1]
  if image.dtype != np.float16 and image.dtype != np.float32 and image.dtype != np.float64:
      image = image.astype(np.float32) / 255.0
  image = np.clip(image, 0, 1)
  cmap = cm.get_cmap(cmap_name)
  colored = cmap(image)[:, :, :3]  # Drop alpha channel
  return (colored * 255).astype(np.uint8)

def numpy_to_base64(img_array):
  """Convert a NumPy array image to base64 string."""
  if img_array.dtype != np.uint8:
      img_array = (img_array * 255).astype(np.uint8)  # normalize if needed
  pil_img = Image.fromarray(img_array)
  buffered = BytesIO()
  pil_img.save(buffered, format="PNG")
  return base64.b64encode(buffered.getvalue()).decode()

def make_grid_output(images, labels, show_colorbar=True):
  html = """
  <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;'>
  """
  for img_array, label in zip(images, labels):
    img_str = numpy_to_base64(img_array)
    html += f"""
    <div style='text-align: center;'>
      <div style='font-weight: bold; margin-bottom: 5px;'>{label}</div>
      <img src='data:image/png;base64,{img_str}' style='width: 100%; height: auto; border: 1px solid #ccc;' />
    </div>
    """
  html += "</div>"

  # Append colorbar legend if requested
  if show_colorbar:
      colorbar_str = make_colorbar_image('turbo', vmin=-1, vmax=1, orientation='horizontal')
      html += f"""
      <div style='margin-top: 20px; text-align: center;'>
        <img src='data:image/png;base64,{colorbar_str}' style='width: 60%; height: auto;' />
      </div>
      """
  return html

def make_colorbar_image(cmap_name='turbo', vmin=-1, vmax=1, orientation='horizontal'):
    """Return a base64-encoded colorbar image."""
    cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    fig, ax = plt.subplots(figsize=(5, 0.5) if orientation == 'horizontal' else (0.5, 5))
    fig.subplots_adjust(bottom=0.5 if orientation == 'horizontal' else 0.1)
    
    cb = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation=orientation
    )
    cb.set_label('Cosine similarity', fontsize=10)
    cb.ax.tick_params(labelsize=8)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    
    # Convert buffer to base64 string
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def generate_distinct_color(index):
  """Generate visually distinct colors using HSV color space."""
  hue = (index * 0.61803398875) % 1  # golden ratio for spacing hues
  r, g, b = colorsys.hsv_to_rgb(hue, 0.5, 0.95)
  return '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

def add_prompt(prompts):
  for prompt in prompts.split("\n"):
    if not prompt.strip() or prompt in prompt_list:
      continue
    color = generate_distinct_color(len(prompt_list))
    prompt_list.append(prompt)
    color_list.append(color)

  # Format prompt display
  colored_prompts = [
    f"<span style='background-color:{color}; color:#FFFFFF'>{p}</span>" 
    for p, color in zip(prompt_list, color_list)
  ]
  return gr.update(value=""), gr.update(value="<br>".join(colored_prompts))

def clear_prompts():
  prompt_list.clear()
  color_list.clear()
  return gr.update(value=""), gr.update(value="")

def on_page_load():
  prompt_list.clear()
  color_list.clear()
  return gr.update(value="")

@torch.inference_mode()
def process_all(input_image, use_templates, softmax, resolution,
                cradio_model, image_processor, lang_adaptor, chunk_size):
  N = len(prompt_list)
  resolution = (resolution, resolution)
  if N == 0:
    raise gr.Error("You must add some prompts", duration=5)
  elif softmax and N == 1:
    raise gr.Error("With softmax enabled, you need at least two prompts", duration=5)

  logger.info("Prompts submitted: %s", str(prompt_list))
  m = "Computing feature map.."
  logger.info(m)
  yield m
  # Image processing
  image = Image.fromarray(input_image)
  pixel_values = image_processor(images=image, return_tensors='pt', do_resize=True).pixel_values
  pixel_values = pixel_values.cuda()
  supported_res = cradio_model.get_nearest_supported_resolution(pixel_values.shape[2], pixel_values.shape[3])
  resized_pixel_values = F.interpolate(
                            pixel_values,
                            size=supported_res,
                            mode='bilinear',
                            align_corners=False
                          )
  #backbone_summary, backbone_features = cradio_model(resized_pixel_values)
  out_dict = cradio_model(resized_pixel_values)
  backbone_summary, backbone_features = out_dict['backbone']
  #sig2_vis_summary, sig2_vis_features = out_dict['siglip2-g']
  # aligment:
  lang_feat = lang_adaptor.head_mlp(backbone_features)
  
  lang_feat = rearrange(lang_feat, 'b (h w) d -> b d h w', 
                               h=resized_pixel_values.shape[-2] // cradio_model.patch_size, 
                               w=resized_pixel_values.shape[-1] // cradio_model.patch_size)
  lang_feat  = F.interpolate(
                        lang_feat,
                        size=resolution,
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(0)

  m = "Computing prompt embeddings.."
  logger.info(m)
  yield m
  # Text encoding
  sig2_adaptor = cradio_model.adaptors['siglip2-g']
  with torch.autocast("cuda", dtype=torch.float16, enabled=True):
    text_input = sig2_adaptor.tokenizer(prompt_list).to("cuda")
    # Normalized text/prompts features
    text_tokens = sig2_adaptor.encode_text(text_input, normalize=True)

  m = "Computing cosine similarity.."
  logger.info(m)
  yield m
  #sim = F.cosine_similarity(sig2_upsampled_feat, text_tokens)
  C, H, W = lang_feat.shape
  # Move features last so spatial layout is preserved
  lang_feat = lang_feat.permute(1, 2, 0)  # (H, W, C)
  # Flatten
  lang_feat = lang_feat.reshape(-1, C)    # (HW, C)
  # Normalize for computing cossim
  lang_feat = F.normalize(lang_feat, dim=-1)    # (HW, C)
  logits = lang_feat @ text_tokens.T      # (HW, N)
  # Softmax across prompts
  if softmax:
    logits = torch.softmax(100 * logits, dim=-1)
  # Reshape back to original image shape
  cos_sim = logits.reshape(H, W, N)

  m = "Visualizing.."
  logger.info(m)
  yield m
  if not softmax:
    cos_sim = utils.norm_img_01(cos_sim.permute(2, 0, 1).unsqueeze(0))
    cos_sim = cos_sim.squeeze(0).permute(1, 2, 0)

  yield make_grid_output(
    [apply_colormap(x) for x in cos_sim.permute(2, 0, 1).cpu().numpy()],
    prompt_list)


@hydra.main(version_base="1.2",
            config_path="../rayfronts/configs",
            config_name="default")
@torch.inference_mode()
def main(cfg=None):
  #encoder_kwargs = dict()
  #if "NARadioEncoder" in cfg.encoder._target_:
  #  encoder_kwargs["input_resolution"] = [224, 224]
  #  encoder_kwargs["compile"] = False # Compiling will make resolution changes slow
#
  #encoder = hydra.utils.instantiate(cfg.encoder, **encoder_kwargs)
  step = 16

  with gr.Blocks() as demo:
    desc = gr.HTML(
    """
    <p align="center"><img src="/gradio_api/file=assets/logo.gif" width="400" alt="RayFronts"/></p>
    <h1 align="center">Test the RayFronts encoders in 2D !</h1>
    <h3 align="center"><a href="https://arxiv.org/abs/2504.06994">Paper</a> | <a href="https://RayFronts.github.io/">Project Page</a> | <a href="https://www.youtube.com/watch?v=fFSKUBHx5gA">Video</a></h3>
    <p align="center">Note that results may be noisy in 2D and get smoothed out as you aggregate features in 3D giving RayFronts its robust 3D open-vocabulary semantic segmentation performance.</p>
    """)
    with gr.Row():
      with gr.Column(scale=1):
        input_image = gr.Image(label="Input Image", type="numpy")

        with gr.Row():
          use_templates = gr.Checkbox(label="Use templates", value=True)
          softmax = gr.Checkbox(label="Use softmax", value=True)
          res_slider = gr.Slider(224, 1024, 224, step=step, label="Resolution", )

        with gr.Row(equal_height=True):
          prompt = gr.Textbox(label="Prompt", placeholder="Type a prompt...",
                              scale=15)
          add_button = gr.Button("+", scale=1)

        prompt_display = gr.HTML()  # To show added prompts
        with gr.Row():
          clear_button = gr.Button("Clear")
          process_button = gr.Button("Run")

      with gr.Column(scale=2):
        # output_image = gr.Image(label="Output Image", type="numpy")
        output_image = gr.HTML()

      hf_repo = "nvidia/C-RADIOv4-H"
      image_processor = CLIPImageProcessor.from_pretrained(hf_repo)
      #cradio_model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
      lang_adaptor_name = 'siglip2-g'
      cradio_model = torch.hub.load('NVlabs/RADIO', 
                                    'radio_model', 
                                    version="c-radio_v4-h", 
                                    progress=True, 
                                    skip_validation=True, 
                                    adaptor_names=[lang_adaptor_name])
      cradio_model.eval().cuda()
      lang_adaptor = cradio_model.adaptors[lang_adaptor_name]

      add_button.click(
        fn=add_prompt,
        inputs=prompt,
        outputs=[prompt, prompt_display])
      process_button.click(
        fn=partial(process_all, cradio_model=cradio_model, image_processor=image_processor, lang_adaptor=lang_adaptor, chunk_size=cfg.chunk_size),
        inputs=[input_image, use_templates, softmax, res_slider],
        outputs=output_image)
      clear_button.click(
        fn=clear_prompts,
        inputs=None,
        outputs=[prompt, prompt_display])

    examples = gr.Examples(
      examples=[
        ["assets/example1.jpg", True, True, "Pothole\nRoad\nSky\nCar\nWater", 224],
        ["assets/example2.jpg", False, True, "Person\nShoes\nGrey Jacket\nRed overalls\nRoad\nCrosswalk\nCar", 512],
        ["assets/example3.jpg", True, True, "Paved ground\nFlood lights\nRed Container\nBuilding\nTanker\nSky\nClouds\nTreeline", 768]
      ],
      inputs=[input_image, use_templates, softmax, prompt, res_slider],
    )

    demo.load(fn=on_page_load, inputs=None, outputs=prompt_display)
  demo.launch(allowed_paths=["assets/logo.gif"])


if __name__ == "__main__":
  main()