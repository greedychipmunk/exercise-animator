import gradio as gr
from gradio_client import Client
import os
from PIL import Image
import pydantic
import gradio_client
import PIL

print(f"STARTUP - gradio: {gr.__version__}")
print(f"STARTUP - pydantic: {pydantic.__version__}")
print(f"STARTUP - gradio_client: {gradio_client.__version__}")
print(f"STARTUP - Pillow: {PIL.__version__}")

# Using 'black-forest-labs/FLUX.1-schnell' for fast, reliable image generation.
# Can be overridden via environment variables if needed.
T2I_SPACE = os.environ.get("T2I_SPACE", "black-forest-labs/FLUX.1-schnell")

def extract_image_path(result):
    if not result:
        return None
    if isinstance(result, tuple):
        first = result[0]
        if isinstance(first, dict) and 'path' in first:
            return first['path']
        return first
    if isinstance(result, dict) and 'path' in result:
        return result['path']
    return result

def generate_exercise_video(exercise_name):
    if not exercise_name or len(exercise_name.strip()) == 0:
        return None
    
    # Construct an optimized storyboard prompt
    prompt = (
        f"A 3-panel horizontal storyboard showing a fit person performing the exercise: {exercise_name}. "
        "Panel 1: starting position, Panel 2: midpoint action, Panel 3: starting position. "
        "Three side-by-side horizontal panels, clean white background, high quality, consistent character."
    )
    
    try:
        import inspect
        import time
        
        hf_token = os.environ.get("HF_TOKEN")
        client_kwargs = {}
        if hf_token:
            sig = inspect.signature(Client.__init__)
            if "token" in sig.parameters:
                client_kwargs["token"] = hf_token
            elif "hf_token" in sig.parameters:
                client_kwargs["hf_token"] = hf_token
                
        print(f"Connecting to {T2I_SPACE}...")
        client = Client(T2I_SPACE, **client_kwargs)
        
        print("Generating storyboard image...")
        result = client.predict(
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            width=1024,
            height=512,
            num_inference_steps=4,
            api_name="/infer"
        )
        
        image_path = extract_image_path(result)
        if not image_path or not os.path.exists(image_path):
            raise ValueError(f"Could not retrieve generated image from path: {image_path}")
            
        print(f"Storyboard generated at: {image_path}. Processing frames...")
        img = Image.open(image_path)
        w, h = img.size
        
        # Split into 3 horizontal frames
        frame_width = w // 3
        frames = []
        for i in range(3):
            box = (i * frame_width, 0, (i + 1) * frame_width, h)
            frame = img.crop(box)
            frames.append(frame)
            
        # Compile into a looping GIF (Frame 1 -> Frame 2 -> Frame 3 -> Frame 2)
        loop_frames = [frames[0], frames[1], frames[2], frames[1]]
        
        gif_path = os.path.join(os.path.dirname(image_path), f"exercise_animation_{int(time.time())}.gif")
        loop_frames[0].save(
            gif_path,
            save_all=True,
            append_images=loop_frames[1:],
            duration=500,  # 500ms per frame
            loop=0
        )
        print(f"Successfully saved looping GIF to: {gif_path}")
        return gif_path
        
    except Exception as e:
        print(f"Error generating animation: {e}")
        raise gr.Error(f"Failed to animate exercise. The API returned an error: {e}")

# Build the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏋️ Exercise Animator")
    gr.Markdown("Type the name of an exercise to see an AI-generated animation of how to perform it.")
    
    with gr.Row():
        with gr.Column():
            exercise_input = gr.Textbox(
                label="Exercise Name", 
                placeholder="e.g., Jumping Jacks, Squats, Pushups...",
                lines=1
            )
            submit_btn = gr.Button("Animate", variant="primary")
        
        with gr.Column():
            video_output = gr.Image(label="Generated Animation", type="filepath")
            
    submit_btn.click(
        fn=generate_exercise_video,
        inputs=[exercise_input],
        outputs=[video_output]
    )
    
    gr.Examples(
        examples=["Jumping Jacks", "Squats", "Pushups", "Burpees", "Yoga Downward Dog"],
        inputs=[exercise_input]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
