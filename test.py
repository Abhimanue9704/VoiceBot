import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QComboBox
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import speech_recognition as sr
from gtts import gTTS
import io
import pygame
from meta_ai_api import MetaAI
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import uuid

# Initialize Pinecone
pc = Pinecone(api_key="dd8c611f-86d4-4528-ad39-e6cb0db9e04e")
index_name = "voice-assistant-index"

# Check if the index already exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Use a smaller dimension for 'all-MiniLM-L6-v2'
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

print(index)
# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize pygame
pygame.mixer.init()

# Initialize the recognizer
recognizer = sr.Recognizer()

def speak(text):
    tts = gTTS(text, lang='en', tld='com')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)

    # Load the speech into pygame
    pygame.mixer.music.load(fp, 'mp3')
    pygame.mixer.music.play()

    # Wait for the speech to finish playing
    while pygame.mixer.music.get_busy():
        QApplication.processEvents()  # Process GUI events to keep the UI responsive

class RecordingThread(QThread):
    command_received = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_recording = False

    def run(self):
        text=""
        print("Recording")
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            while self.is_recording:
                try:
                    audio = recognizer.listen(source,timeout=sys.maxsize, phrase_time_limit=30)
                    text = text+" "+recognizer.recognize_google(audio)
                    print(text)
                    if text.lower() == 'bye':
                        speak("Bye. Take care.")
                        if self.main_window:
                            self.main_window.close()
                except sr.RequestError:
                    speak("Speech service is down. Please try again later.")
                except Exception as e:
                    print(f"Error during recording: {e}")
            self.command_received.emit(text)        

    def stop(self):
        self.is_recording = False

class VoiceAssistantApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.response_thread = None
        self.identify_id={}
        self.history = []  # Initialize the history list
        self.current_session_id = str(uuid.uuid4())  # Initialize the first session ID
        self.histories = {}  # Dictionary to store session histories
        self.load_histories()
        self.recording_thread = None

    def initUI(self):
        self.setWindowTitle('Voice Assistant')
        self.layout = QVBoxLayout()

        self.label = QLabel('Ask something:', self)
        self.layout.addWidget(self.label)

        self.history_dropdown = QComboBox(self)
        self.history_dropdown.activated.connect(self.load_selected_history)
        self.layout.addWidget(self.history_dropdown)

        self.new_session_button = QPushButton("+", self)
        self.new_session_button.clicked.connect(self.start_new_session)
        self.layout.addWidget(self.new_session_button)

        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_recording)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_recording)
        self.layout.addWidget(self.stop_button)

        self.response_label = QLabel(self)
        self.layout.addWidget(self.response_label)

        self.setLayout(self.layout)

    def load_histories(self):
        try:
            metadata={}
            result = index.query(namespace='history',vector=[0]*384,top_k=100,include_values=True)  # Specify the namespace here
            self.histories = [item['id'] for item in result['matches']]
            for id in self.histories:
                metadata[id]=(index.query(namespace='history',id=id,top_k=1,include_values=True,include_metadata=True)['matches'][0].get('metadata', {})).get('history_text')
                self.identify_id[id]=metadata[id].split('#')[1].strip('\n')
            self.history_dropdown.clear()
            print(self.identify_id)
            self.history_dropdown.addItems([(metadata[id]).split('#')[1].strip('\n') for id in metadata])
        except Exception as e:
            print(f"Error loading histories: {e}")

    def load_selected_history(self):
        drop_down_value = self.history_dropdown.currentText()
        for key in self.identify_id:
            if self.identify_id[key] == drop_down_value:
                session_id = key
                break
        self.current_session_id = session_id
        self.history = []
        if session_id:
            result1 = index.query(namespace='history', id=session_id, top_k=1, include_values=True, include_metadata=True)
            metadata = result1['matches'][0].get('metadata', {})
            history_text = metadata.get('history_text')
            if history_text:
                user_assistant_pairs = history_text.split('User: ')
                for pair in user_assistant_pairs:
                    if 'Assistant: ' in pair:
                        user_text, assistant_text = pair.split('Assistant: ', 1)
                        self.history.append({'user': user_text.strip(), 'assistant': assistant_text.strip()})
                    else:
                        self.history.append({'user': pair.strip(), 'assistant': ''})
            print(self.history)

    def start_new_session(self):
        self.history = []
        self.response_label.clear()
        self.current_session_id = str(uuid.uuid4())  # Generate a new unique session ID

    def process_command(self, prompt):
        text = prompt
        if text:
            if self.response_thread is None or not self.response_thread.isRunning():
                self.response_thread = ResponseThread(text, self.history, self.current_session_id)
                self.response_thread.response_ready.connect(lambda response, text=text: self.show_response(response, text))
                self.response_thread.start()
        else:
            self.show_response({"message": "Please enter a command."})

    def show_response(self, response, text):
        message = response.get('message', 'No response')
        print(message)
        lines = message.split('. ')  # Split the response into sentences
        self.response_label.setText("")  # Clear the label
        self.speak_and_update_label(lines, 0)
        self.history.append({"user": text, "assistant": message})  # Update the history

    def speak_and_update_label(self, lines, index):
        if index < len(lines):
            line = lines[index]
            self.response_label.setText(self.response_label.text() + line + ".\n")
            QTimer.singleShot(100, lambda: self.speak_line(line, lines, index))

    def speak_line(self, line, lines, index):
        speak(line)
        self.speak_and_update_label(lines, index + 1)

    def start_recording(self):
        if self.recording_thread is None or not self.recording_thread.isRunning():
            self.recording_thread = RecordingThread()
            self.recording_thread.command_received.connect(self.process_command)
            self.recording_thread.is_recording = True
            self.recording_thread.start()
            print("Recording started")

    def stop_recording(self):
        if self.recording_thread is not None:
            self.recording_thread.stop()
            self.recording_thread.wait()
            print("Recording stopped")

class ResponseThread(QThread):
    response_ready = pyqtSignal(dict)

    def __init__(self, prompt, history, session_id):
        super().__init__()
        self.prompt = prompt
        self.history = history
        self.session_id = session_id

    def run(self):
        try:
            ai = MetaAI()
            # Include the history in the prompt
            history_text = ' '.join([f"User: {entry['user']} Assistant: {entry['assistant']}" for entry in self.history]).split('#')[0]
            full_prompt = history_text + f" User: {self.prompt}"
            response = ai.prompt(message=full_prompt)
            summary_prompt = (
                "Based on the following conversation, provide a concise 3-word heading:\n"
                f"{full_prompt}"
                "Only provide a 3-word heading."
            )
            summary = ai.prompt(message=summary_prompt)
            message = response.get('message', 'No response')
            summary_message = summary.get('message', 'No summary available')

            print(message)
            print(summary_message)
            id = self.session_id

            self.response_ready.emit(response)
            full_prompt = full_prompt + f" Assistant: {message}#{summary_message}"
            vector = model.encode(full_prompt)
            index.upsert(vectors=[(id, vector, {'history_text': full_prompt})], namespace='history')  # Specify the namespace here
        except Exception as e:
            print(f"Error executing Meta AI command: {e}")
            self.response_ready.emit({"message": f"Error: {e}"})

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VoiceAssistantApp()
    ex.show()
    sys.exit(app.exec_())
