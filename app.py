import gradio as gr
from faster_whisper import WhisperModel
import edge_tts
import tempfile
import asyncio
import yaml
import os
import openai

open_ai_client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

model = WhisperModel("tiny", compute_type="float32")

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

def generate_prompt(personality: str, user_query: str) -> str:
    
    prompt = f'''
{config['prompts']['base']}
{config['prompts'][personality]}

User query:

{user_query} -> '''

    return prompt


def gpt_answer(prompt, personality, chatbot_history):

    print(f'going to send the prompt: {prompt}')

    history_for_gpt_call = [
        {"role": "system", "content": f"You are a helpful assistant, with the personality of a {personality}."}
    ] + chatbot_history + [
        {"role": "user", "content": prompt}
    ]

    completion =  open_ai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history_for_gpt_call
    )

    # Extract the generated response from the API response
    generated_text = completion.choices[0].message.content.strip()

    return generated_text

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

    # Ask llm for response to text
    prompt = generate_prompt(
        personality=buddy_personality,
        user_query=user_query_transcribed
    )

    bot_message = gpt_answer(prompt=prompt,
                             personality=buddy_personality,
                             chatbot_history=chatbot_history)

    chatbot_history.append(gr.ChatMessage(role="user", content=user_query_transcribed))
    chatbot_history.append(gr.ChatMessage(role="assistant", content=bot_message))

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

    chatbot = gr.Chatbot(type='messages')
    
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