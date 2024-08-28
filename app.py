import gradio as gr
from faster_whisper import WhisperModel
import edge_tts
import tempfile
import asyncio

model = WhisperModel("tiny", compute_type="float32")

# Text-to-speech function
async def text_to_speech(text, voice):   
    communicate = edge_tts.Communicate(text, voice)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name
        await communicate.save(tmp_path)
    return tmp_path, None

def generate_response(
    language_level, buddy_personality,
    language_choice, user_query_audio,
    chatbot_history
):
    # Convert input audio to text

    language_codes = {'English':'en',
                     'Spanish':'es',
                     'Japanese':'ja'}

    user_query_transcribed_segments, info = model.transcribe(
        audio=user_query_audio,
        language=language_codes[language_choice]
        )
    user_query_transcribed = list(user_query_transcribed_segments)[0].text.strip()
    user_message = 'User: ' + user_query_transcribed

    # Ask llm for response to text

    bot_message = 'Bot: ' + user_query_transcribed
    chatbot_history.append((user_message, bot_message))

    # Convert llm response to audio
    # Return None to reset user input audio and
    # llm response + user inputs in chatbot_history object to be displayed 
    if language_choice == "Spanish":
        voice_short_name =  "es-MX-JorgeNeural"
    elif language_choice == "Japanese":
        voice_short_name = "ja-JP-KeitaNeural"
    else: 
        # default to an english voice otherwise
        voice_short_name = "en-US-BrianNeural"
    bot_message_audio, warning = asyncio.run(text_to_speech(text=bot_message, voice=voice_short_name))
    
    return None, chatbot_history, bot_message_audio

with gr.Blocks() as demo:

    header_section = gr.Markdown(
    """
    # AI Language Buddy!
    Click the **converse** button to practice your language skills!
    """)
    
    language = gr.Dropdown(
        choices=['English', 'Spanish', 'Japanese'],
        label='Language Choice',
        value='English'
    )
    
    language_level = gr.Dropdown(
        choices=['Beginner', 'Intermediate', 'Advanced'],
        label='Language Level',
        value='Beginner'
    )
    
    personality = gr.Dropdown(
        choices=['Formal Teacher', 'Flirty Friend', 'Sarcastic Bro'],
        label='Language Buddy Personality',
        value='Flirty Friend'
    )

    chatbot = gr.Chatbot()
    
    user_input = gr.Audio(
        sources='microphone',
        show_download_button=True,
        type='filepath'
    )

    ai_response = gr.Audio(
        autoplay=True
    )

    converse_button = gr.Button("Send Message")

    clear_button = gr.Button("Clear Convo History")

    converse_button.click(
        fn=generate_response,
        inputs=[
            language_level,
            personality,
            language,
            user_input,
            chatbot
        ],
        outputs=[user_input,
                 chatbot,
                 ai_response]
    )

demo.launch()