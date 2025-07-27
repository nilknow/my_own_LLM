import numpy as np
import os

from ai.core import LinearLayer, Softmax, CrossEntropyLoss
from ai.text.CharTokenizer import CharTokenizer
from ai.text.EmbeddingLayer import EmbeddingLayer


class BigramModel:
    def __init__(self, tokenizer: CharTokenizer, embedding_dim, learning_rate=0.01):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        self.embedding_layer = EmbeddingLayer(self.vocab_size, self.embedding_dim)
        self.linear_layer = LinearLayer(self.embedding_dim, self.vocab_size)
        self.softmax = Softmax()
        self.loss_func = CrossEntropyLoss()

    def forward(self, input_ids):
        output_embeds = self.embedding_layer.forward(input_ids)

        reshaped_embeds = output_embeds.squeeze(axis=1)

        logits = self.linear_layer.forward(reshaped_embeds)

        predictions = self.softmax.forward(logits)

        return predictions, logits

    def backward(self, predictions, true_next_char_ids):
        d_logits = self.loss_func.backward()

        d_reshaped_embeds = self.linear_layer.backward(d_logits)

        d_output_embeds = d_reshaped_embeds[:, np.newaxis, :]
        self.embedding_layer.backward(d_output_embeds)

    def update_parameters(self):
        self.embedding_layer.E -= self.learning_rate * self.embedding_layer.dE
        self.linear_layer.W -= self.learning_rate * self.linear_layer.dW
        self.linear_layer.b -= self.learning_rate * self.linear_layer.db

    def train(self, text_sequence, num_epochs, batch_size):
        inputs = np.array(text_sequence[:-1]).reshape(-1, 1)
        targets = np.array(text_sequence[1:])

        num_samples = len(inputs)

        for epoch in range(num_epochs):
            total_loss = 0
            permutation = np.random.permutation(num_samples)

            for i in range(0, num_samples, batch_size):
                batch_indices = permutation[i:i + batch_size]

                batch_inputs = inputs[batch_indices]
                batch_targets = targets[batch_indices]

                predictions, _ = self.forward(batch_inputs)
                loss = self.loss_func.forward(predictions, batch_targets)
                total_loss += loss

                self.backward(predictions, batch_targets)

                self.update_parameters()

                avg_loss = total_loss / (num_samples / batch_size)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    def generate(self, start_char, num_generate=50, temperature=1.0):
        generated_text = [start_char]
        current_id = self.tokenizer.encode(start_char)[0]

        for _ in range(num_generate):
            input_id_batch = np.array([[current_id]])

            predictions, _ = self.forward(input_id_batch)

            scaled_predictions = predictions[0] / temperature
            exp_preds = np.exp(scaled_predictions - np.max(scaled_predictions))
            probs = exp_preds / np.sum(exp_preds)

            next_id = np.random.choice(self.vocab_size, p=probs)

            if next_id == self.tokenizer.char_to_idx.get('<EOS>', -1):
                break

            next_char = self.tokenizer.decode([next_id])
            generated_text.append(next_char)
            current_id = next_id

        return "".join(generated_text)


if __name__ == "__main__":
    corpus_file = "small_text_corpus.txt"

    tokenizer = CharTokenizer(corpus_file, special_tokens=['<PAD>', '<UNK>', '<BOS>', '<EOS>'])

    with open(corpus_file, 'r', encoding='utf-8') as f:
        full_text = f.read()

    encoded_text_sequence = tokenizer.encode(full_text)

    encoded_text_array = np.array(encoded_text_sequence)

    embedding_dim = 32
    num_epochs = 20000
    batch_size = 16
    learning_rate = 0.05

    model = BigramModel(tokenizer, embedding_dim, learning_rate)

    print("\n--- Starting Bigram Model Training ---")
    model.train(encoded_text_array, num_epochs, batch_size)

    print("\n--- Generating Text ---")
    start_char = "h"
    print(f"Starting with '{start_char}':")
    generated_sequence = model.generate(start_char, num_generate=50, temperature=0.1)
    print(generated_sequence)

    start_char = "t"
    print(f"\nStarting with '{start_char}':")
    generated_sequence = model.generate(start_char, num_generate=50, temperature=0.1)
    print(generated_sequence)

    start_char = "p"
    print(f"\nStarting with '{start_char}':")
    generated_sequence = model.generate(start_char, num_generate=50, temperature=0.1)
    print(generated_sequence)
