import whisper
import gradio as gr
import time

model = whisper.load_model("base")

def transcribe(audio):

    #time.sleep(3)
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text

gr.Interface(
    title="OpenAI-Whisper Audio to Text Web UI",
    fn=transcribe,
    inputs=[gr.components.Audio(type="filepath")],
    outputs=["textbox"],
    live=True
).launch()
