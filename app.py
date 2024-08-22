import gradio as gr
from faster_whisper import WhisperModel

model = WhisperModel("tiny")

def generate_response(correction_intensity,
                      language_level,
                      buddy_personality,
                      language_choice,
                      user_query_audio
                      ):
    # Convert input audio to text
    # Ask llm for response to text
    # Convert llm response to audio
    # Return converted llm response
    user_query_transcribed_segments, info = model.transcribe(user_query_audio)
    user_query_transcribed = list(user_query_transcribed_segments)[0].text.strip()
    return user_query_audio, user_query_transcribed

demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Slider(
            minimum=1,
            maximum=5,
            step=1,
            label='Grammar Correction Intensity'
        ),
        gr.Dropdown(
            choices=['Beginner', 'Intermediate', 'Advanced'],
            label='Language Level'),
        gr.Dropdown(
            choices=['Formal Teacher', 'Flirty Friend', 'Sarcastic Bro'],
            label='Language Buddy Personality'),
        gr.Dropdown(
            choices=['English', 'Urdu', 'Japanese'],
            label='Language Choice'),
        gr.Audio(
            # format='mp3',
            show_download_button=True,
            type='filepath'
        )],
    outputs=[
        gr.Audio(label='User Query'),
        gr.Textbox(label='AI Buddy Response')
    ],
    title="AI Language Buddy"
)

demo.launch()