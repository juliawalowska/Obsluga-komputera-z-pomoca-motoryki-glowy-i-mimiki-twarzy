import vosk
import pyaudio
import json

model_path = "s2t_model/vosk-model-small-pl-0.22"

model = vosk.Model(model_path)

rec = vosk.KaldiRecognizer(model, 16000) # Sample rating 16000Hz

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8192)

output_file_path = "recognized_text.txt"

with open(output_file_path, "w") as output_file:
    print("Listening for speech. Say 'Terminate' to stop.")

    while True:
        data = stream.read(4096)
        if rec.AcceptWaveform(data):
             # Parse the JSON result and get the recognized text
            result = json.loads(rec.Result())
            recognized_text = result['text']
            
            # Write recognized text to the file
            output_file.write(recognized_text + "\n")
            print(recognized_text)
            
            # Check for the termination keyword
            if "zakończ" in recognized_text.lower():
                print("Termination keyword detected. Stopping...")
                break
            