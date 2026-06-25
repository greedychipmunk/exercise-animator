import gradio as gr
import os
from PIL import Image
import time
from huggingface_hub import InferenceClient
import imageio
import numpy as np

# Using 'black-forest-labs/FLUX.1-schnell' for fast, reliable image generation.
# Can be overridden via environment variables if needed.
T2I_SPACE = os.environ.get("T2I_SPACE", "black-forest-labs/FLUX.1-schnell")

def generate_exercise_video(exercise_name, output_format):
    if not exercise_name or len(exercise_name.strip()) == 0:
        return None, None, gr.Image(), gr.Video()
    
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
            
        # Compile into a looping frame sequence (Frame 1 -> Frame 2 -> Frame 3 -> Frame 2)
        loop_frames = [frames[0], frames[1], frames[2], frames[1]]
        
        # Ensure temporary directory exists
        tmp_dir = os.path.join(os.getcwd(), "tmp_animations")
        os.makedirs(tmp_dir, exist_ok=True)
        timestamp = int(time.time())
        
        # 1. Save as Animated GIF
        gif_path = os.path.join(tmp_dir, f"exercise_animation_{timestamp}.gif")
        loop_frames[0].save(
            gif_path,
            save_all=True,
            append_images=loop_frames[1:],
            duration=500,  # 500ms per frame
            loop=0
        )
        print(f"Successfully saved looping GIF to: {gif_path}")
        
        # 2. Save as MP4 Video
        mp4_path = os.path.join(tmp_dir, f"exercise_animation_{timestamp}.mp4")
        # Resize to even dimensions for standard video codec compatibility
        w_even = (frame_width // 2) * 2
        h_even = (h // 2) * 2
        
        processed_video_frames = []
        for f in loop_frames:
            if f.size != (w_even, h_even):
                f = f.resize((w_even, h_even), Image.Resampling.LANCZOS)
            processed_video_frames.append(np.array(f))
            
        # Write MP4 (fps=2 matches 500ms duration per frame)
        # macro_block_size=None keeps the exact resized dimensions (e.g. 340x512)
        writer = imageio.get_writer(mp4_path, fps=2, macro_block_size=None)
        for frame in processed_video_frames:
            writer.append_data(frame)
        writer.close()
        print(f"Successfully saved looping MP4 to: {mp4_path}")
        
        # Return files and visibility configurations depending on chosen format
        if output_format == "Animated GIF":
            return (
                gif_path,
                mp4_path,
                gr.Image(visible=True, value=gif_path),
                gr.Video(visible=False, value=None)
            )
        else:
            return (
                gif_path,
                mp4_path,
                gr.Image(visible=False, value=None),
                gr.Video(visible=True, value=mp4_path, autoplay=True, loop=True)
            )
        
    except Exception as e:
        print(f"Error generating animation: {e}")
        raise gr.Error(f"Failed to animate exercise. The API returned an error: {e}")

# Build the Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🏋️ Exercise Animator")
    gr.Markdown("Type the name of an exercise to see an AI-generated animation of how to perform it.")
    
    # Store file paths so switching format doesn't require regeneration
    gif_state = gr.State(value=None)
    mp4_state = gr.State(value=None)
    
    with gr.Row():
        with gr.Column():
            exercise_input = gr.Textbox(
                label="Exercise Name", 
                placeholder="e.g., Jumping Jacks, Squats, Pushups...",
                lines=1
            )
            format_choice = gr.Radio(
                choices=["Animated GIF", "Video (MP4)"],
                value="Animated GIF",
                label="Output Format"
            )
            submit_btn = gr.Button("Animate", variant="primary")
        
        with gr.Column():
            video_output_image = gr.Image(label="Generated Animation (GIF)", type="filepath", visible=True)
            video_output_video = gr.Video(label="Generated Animation (Video)", visible=False)
            
    submit_btn.click(
        fn=generate_exercise_video,
        inputs=[exercise_input, format_choice],
        outputs=[gif_state, mp4_state, video_output_image, video_output_video]
    )
    
    def toggle_format(choice, gif_path, mp4_path):
        if choice == "Animated GIF":
            return (
                gr.Image(visible=True, value=gif_path),
                gr.Video(visible=False, value=None)
            )
        else:
            return (
                gr.Image(visible=False, value=None),
                gr.Video(visible=True, value=mp4_path, autoplay=True, loop=True)
            )
            
    format_choice.change(
        fn=toggle_format,
        inputs=[format_choice, gif_state, mp4_state],
        outputs=[video_output_image, video_output_video]
    )
    
    gr.Examples(
        examples=["Jumping Jacks", "Squats", "Pushups", "Burpees", "Yoga Downward Dog"],
        inputs=[exercise_input]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), server_name="0.0.0.0")
