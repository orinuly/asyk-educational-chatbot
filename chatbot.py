import tkinter as tk
from tkinter import scrolledtext
import random
import json
import torch
from model import NeuralNets
from nltk_utils import bag_of_words, tokenize


class ChatGUI:
    def __init__(self, master):
        self.master = master
        master.title("Asyk")

        self.chat_display = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=100, height=30)
        self.chat_display.pack(padx=10, pady=10)

        self.user_input = tk.Entry(master, width=40)
        self.user_input.pack(padx=10, pady=10)

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.pack(padx=10, pady=10)

        self.user_input.bind("<Return>", lambda event: self.send_message())

        self.bot_name = "Asyk"

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open('intents.json', 'r') as f:
            self.intents = json.load(f)

        self.FILE = "data.pth"
        self.data = torch.load(self.FILE)

        self.input_size = self.data["input_size"]
        self.hidden_size = self.data["hidden_size"]
        self.output_size = self.data["output_size"]
        self.all_words = self.data["all_words"]
        self.tags = self.data["tags"]
        self.model_state = self.data["model_state"]

        self.model = NeuralNets(self.input_size, self.hidden_size, self.output_size).to(device)
        self.model.load_state_dict(self.model_state)
        self.model.eval()

        initial_message = "Say 'Hello!' to Asyk!"
        self.display_message(f'{initial_message}')

    def send_message(self):
        user_message = self.user_input.get()
        self.display_message(f'You: {user_message}')

        self.get_chatbot_response(user_message, self.handle_chatbot_response)

        self.user_input.delete(0, tk.END)

    def get_chatbot_response(self, user_message, callback):
        sentence = tokenize(user_message)
        x = bag_of_words(sentence, self.all_words)
        x = x.reshape(1, x.shape[0])
        x = torch.from_numpy(x)

        output = self.model(x)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in self.intents["intents"]:
                if tag == intent["tag"]:
                    response = random.choice(intent["responses"])

                    callback(response)
                    return
        else:
            callback("I can not give you an answer, please ask again")
            return

    def handle_chatbot_response(self, response):
        self.display_message(f'{self.bot_name}: {response}')

    def display_message(self, message):
        self.chat_display.insert(tk.END, message + '\n')
        self.chat_display.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    chat_gui = ChatGUI(root)
    root.mainloop()
