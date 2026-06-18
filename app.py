import gradio as gr
import os
from PIL import Image
import time
from huggingface_hub import InferenceClient

# Using 'black-forest-labs/FLUX.1-schnell' for fast, reliable image generation.
# Can be overridden via environment variables if needed.
T2I_SPACE = os.environ.get("T2I_SPACE", "black-forest-labs/FLUX.1-schnell")

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
        hf_token = os.environ.get("HF_TOKEN")
        print(f"Connecting to serverless Inference Client for {T2I_SPACE}...")
        client = InferenceClient(T2I_SPACE, token=hf_token)
        
        print("Generating storyboard image via serverless API...")
        img = client.text_to_image(prompt, width=1024, height=512)
        
        w, h = img.size
        print(f"Storyboard generated with size: {w}x{h}. Processing frames...")
        
        # Split into 3 horizontal frames
        frame_width = w // 3
        frames = []
        for i in range(3):
            box = (i * frame_width, 0, (i + 1) * frame_width, h)
            frame = img.crop(box)
            frames.append(frame)
            
        # Compile into a looping GIF (Frame 1 -> Frame 2 -> Frame 3 -> Frame 2)
        loop_frames = [frames[0], frames[1], frames[2], frames[1]]
        
        # Save GIF in a temporary directory inside workspace
        tmp_dir = os.path.join(os.getcwd(), "tmp_animations")
        os.makedirs(tmp_dir, exist_ok=True)
        gif_path = os.path.join(tmp_dir, f"exercise_animation_{int(time.time())}.gif")
        
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
with gr.Blocks() as demo:
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
    demo.launch(theme=gr.themes.Soft(), server_name="0.0.0.0")
