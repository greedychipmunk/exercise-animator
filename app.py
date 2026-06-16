import gradio as gr
from gradio_client import Client
import os

# Initialize the Gradio Client
# Using 'fffiloni/Wan2.1' for a simple and direct API
T2V_SPACE = "fffiloni/Wan2.1"

def generate_exercise_video(exercise_name):
    if not exercise_name or len(exercise_name.strip()) == 0:
        return None
    
    # Construct an optimized prompt
    prompt = (
        f"A highly detailed, clear, instructional video of a fit person perfectly performing the exercise: {exercise_name}. "
        "Clean studio background, full body visible, smooth fluid motion, high quality, 4k."
    )
    
    try:
        client = Client(T2V_SPACE)
        result = client.predict(
            prompt=prompt,
            api_name="/infer"
        )
        # The result is a dictionary with a 'video' key according to view_api()
        if isinstance(result, dict) and 'video' in result:
            return result['video']
        return result
    except Exception as e:
        print(f"Error calling API: {e}")
        return None

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
            video_output = gr.Video(label="Generated Animation")
            
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
