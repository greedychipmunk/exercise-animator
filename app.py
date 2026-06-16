import gradio as gr
from gradio_client import Client
import os

# Initialize the Gradio Client
# Using 'fffiloni/Wan2.1' for a simple and direct API
T2V_SPACE = "fffiloni/Wan2.1"

def extract_video_path(result):
    if not result:
        return None
    
    # If result is a dictionary
    if isinstance(result, dict):
        if 'video' in result:
            video_val = result['video']
            if isinstance(video_val, dict) and 'path' in video_val:
                return video_val['path']
            elif isinstance(video_val, str):
                return video_val
        if 'path' in result:
            return result['path']
            
    # If result is a list or tuple
    elif isinstance(result, (list, tuple)):
        if len(result) > 0:
            return extract_video_path(result[0])
            
    # If result is a string
    elif isinstance(result, str):
        return result
        
    return None

def generate_exercise_video(exercise_name):
    if not exercise_name or len(exercise_name.strip()) == 0:
        return None
    
    # Construct an optimized prompt
    prompt = (
        f"A highly detailed, clear, instructional video of a fit person perfectly performing the exercise: {exercise_name}. "
        "Clean studio background, full body visible, smooth fluid motion, high quality, 4k."
    )
    
    try:
        # Pass HF_TOKEN if available in the environment, checking parameter name dynamically
        import inspect
        hf_token = os.environ.get("HF_TOKEN")
        client_kwargs = {}
        if hf_token:
            sig = inspect.signature(Client.__init__)
            if "token" in sig.parameters:
                client_kwargs["token"] = hf_token
            elif "hf_token" in sig.parameters:
                client_kwargs["hf_token"] = hf_token
                
        client = Client(T2V_SPACE, **client_kwargs)
        
        result = client.predict(
            prompt=prompt,
            api_name="/infer"
        )
        
        video_path = extract_video_path(result)
        if video_path:
            return video_path
            
        raise ValueError(f"Could not extract video path from result: {result}")
        
    except Exception as e:
        print(f"Error calling API: {e}")
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
