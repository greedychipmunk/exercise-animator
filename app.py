import gradio as gr
from gradio_client import Client
import os

# Initialize the Gradio Client
# Using the official 'Wan-AI/Wan2.1' as default for stable and active GPU generation.
# Can be overridden via environment variables if the user duplicates the Space.
T2V_SPACE = os.environ.get("T2V_SPACE", "Wan-AI/Wan2.1")

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
        import time
        
        hf_token = os.environ.get("HF_TOKEN")
        client_kwargs = {}
        if hf_token:
            sig = inspect.signature(Client.__init__)
            if "token" in sig.parameters:
                client_kwargs["token"] = hf_token
            elif "hf_token" in sig.parameters:
                client_kwargs["hf_token"] = hf_token
                
        print(f"Connecting to {T2V_SPACE}...")
        client = Client(T2V_SPACE, **client_kwargs)
        
        # 1. Attempt direct /infer endpoint first (for spaces that support direct sync calls)
        try:
            print("Attempting direct /infer endpoint...")
            result = client.predict(
                prompt=prompt,
                api_name="/infer"
            )
            video_path = extract_video_path(result)
            if video_path:
                print(f"Success! Video generated directly: {video_path}")
                return video_path
        except Exception as infer_err:
            err_msg = str(infer_err).lower()
            # If the endpoint doesn't exist, fall back to the async polling workflow
            if "api_name" in err_msg or "not found" in err_msg or "invalid" in err_msg:
                print("Direct /infer endpoint not found or invalid. Falling back to async polling flow...")
            else:
                # If it's a real runtime error on a direct space, raise it
                raise infer_err
        
        # 2. Async polling workflow (for Wan-AI/Wan2.1 official space)
        print("Switching to text-to-video tab...")
        try:
            client.predict(api_name="/switch_t2v_tab")
        except Exception as tab_err:
            print(f"Warning: failed to switch tab: {tab_err}")
            
        print("Submitting text-to-video request...")
        max_queue_retries = 5
        for retry_idx in range(max_queue_retries):
            try:
                client.predict(
                    prompt=prompt,
                    size="1280*720",
                    watermark_wan=False,
                    seed=-1,
                    api_name="/t2v_generation_async"
                )
                break  # Succeeded submitting!
            except Exception as queue_err:
                # If queue is full, wait and retry
                if "queue is full" in str(queue_err).lower() and retry_idx < max_queue_retries - 1:
                    wait_sec = (retry_idx + 1) * 10
                    print(f"Upstream queue is full. Retrying in {wait_sec} seconds...")
                    time.sleep(wait_sec)
                else:
                    raise queue_err
        
        print("Starting status polling...")
        # Poll up to 60 times (5 minutes)
        for i in range(60):
            time.sleep(5)
            status = client.predict(api_name="/status_refresh")
            
            # Extract video from status_refresh response
            # status[0] is the generated_video update dictionary
            video_update = status[0] if isinstance(status, (list, tuple)) and len(status) > 0 else None
            
            video_path = None
            if isinstance(video_update, dict):
                if 'value' in video_update:
                    video_path = extract_video_path(video_update['value'])
                else:
                    video_path = extract_video_path(video_update)
            else:
                video_path = extract_video_path(video_update)
                
            if video_path:
                print(f"Success! Video generated at: {video_path}")
                return video_path
                
            progress_val = status[3] if isinstance(status, (list, tuple)) and len(status) > 3 else "Unknown"
            print(f"Polling check {i+1}/60: progress={progress_val}")
            
        raise TimeoutError("Video generation timed out on the upstream space.")
        
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
