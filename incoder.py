#!/usr/bin/env python
# coding: utf-8



# get_ipython().system('nvidia-smi')

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, Model, losses, optimizers

import torch
torch.cuda.is_available()

# After training, you can use vae.predict(x_test) to reconstruct inputs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import nltk
from nltk.tokenize import word_tokenize

# Download the necessary NLTK models (if you haven't already)
nltk.download('punkt')


import seaborn as sns
from pprint import pprint

from tensorflow.keras.callbacks import Callback

class VerboseEveryTenEpochs(Callback):
    def __init__(self):
        super(VerboseEveryTenEpochs, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        # Check if the epoch is one where we want to print the logs (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: {logs}")


# ASCII character set
characters = [chr(i) for i in range(128)]

# Create a mapping from characters to their index in ASCII
char_to_index = {char: idx for idx, char in enumerate(characters)}

pprint(char_to_index)

# Create a mapping from index in ASCII to characters
index_to_char = {idx: char for char, idx in char_to_index.items()}

def visualize_components(model, input_matrix, input_word=None):
    reconstructed, z_3_vector, z_mu, sigma, z_scalar, v = model(input_matrix, return_components=True)

    fig, axes = plt.subplots(1, 7, figsize=(21, 3))

    # Input word (matrix)
    axes[0].imshow(input_matrix[0, :, :, 0], cmap='gray')
    axes[0].set_title(f"Input Word\n{input_word if input_word else ''}")

    # 3-vector v
    axes[2].bar(['Dim1', 'Dim2', 'Dim3'], v[0, :])
    axes[2].set_title("3-Vector v\n" + '\n'.join([f"{val:.4f}" for val in v[0, :]]))

    # 3-vector z_3_vector
    axes[1].bar(['Dim1', 'Dim2', 'Dim3'], z_3_vector[0, :])
    axes[1].set_title("3-Vector z\n" + '\n'.join([f"{val:.4f}" for val in z_3_vector[0, :]]))

    # Scalar mu
    axes[3].bar(['Scalar Mu'], z_mu[0, 0])
    axes[3].set_title(f"Scalar Mu\n{z_mu[0, 0]:.4f}")

    # Scalar sigma
    axes[4].bar(['Scalar Sigma'], sigma[0, 0])
    axes[4].set_title(f"Scalar Sigma\n{sigma[0, 0]:.4f}")

    # Sampled z
    axes[5].bar(['Sampled Z'], z_scalar[0, 0])
    axes[5].set_title(f"Sampled Z\n{z_scalar[0, 0]:.4f}")

    # Reconstructed word (matrix)
    axes[6].imshow(reconstructed[0, :, :, 0], cmap='gray')
    axes[6].set_title(f"Reconstructed\n{decode_matrix(reconstructed[0, :, :, 0])}")
    
    pprint(reconstructed[0, :, :, 0])

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{input_word}.png")




def decode_matrix(matrix):
    """Decode a 128x128 matrix into a word, ignoring NUL characters."""
    word = ''
    for row in matrix:
        if row.ndim > 1:  # If row is 2D (for batched input), we take the first axis
            row = row[0]
        char_index = np.argmax(row)
        char = index_to_char[char_index]
        if char == '\x00':  # Skip if NUL character is found
            continue
        word += char
    return word


def one_hot_encode(char):
    """One hot encode a single character."""
    vector = np.zeros(len(characters)) - 1
    vector[char_to_index[char]] = 1
    return vector

def gaussian_encode(char, sigma=10):
    """Gaussian encode a single character."""
    vector = np.zeros(len(characters))
    center_index = char_to_index[char]
    indices = np.arange(len(characters))
    vector = np.exp(-((indices - center_index)**2) / (2 * sigma**2))
    return vector

def encode_word(word):
    """Encode a word into a 128x128 matrix using Numpy optimizations."""
    # Create an empty matrix filled with the one hot encoding of NUL character
    matrix = np.tile(gaussian_encode('\x00'), (128, 1))

    # Calculate the indices where each character's one-hot encoding will be placed
    step = max(1, int(128 / len(word)))
    indices = np.arange(len(word)) * step

    # Create one-hot encodings for all characters in the word at once
    char_vectors = np.array([gaussian_encode(char) for char in word])

    # Place the character one-hot encodings into the matrix
    matrix[indices] = char_vectors

    return matrix


def decode_matrix(matrix):
    """Decode a 128x128 matrix into a word, ignoring NUL characters, using Numpy optimizations."""
    # Find the index with the maximum value (one-hot encoded character) in each row
    char_indices = np.argmax(matrix, axis=1)

    # Convert indices to characters
    chars = np.array([index_to_char[idx] for idx in char_indices])

    # Filter out NUL characters and join the rest into a word
    word = ''.join(chars[chars != '\x00'])

    return word


class CustomVAE(Model):
    def __init__(self, **kwargs):
        super(CustomVAE, self).__init__(**kwargs)
        self.encoder = self.build_encoder()
        self.intermediate = self.build_intermediate()
        self.decoder = self.build_decoder()

        # Temperature for SoftFloor bijector
        self.temperature = 0.1
        self.softfloor = tfp.bijectors.Softfloor(temperature=self.temperature)

    def build_encoder(self):
        inputs = tf.keras.Input(shape=(128, 128, 1))
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        z_3_vector = layers.Dense(3, name='z_3_vector')(x)
        encoder = tf.keras.Model(inputs, z_3_vector, name='encoder')
        return encoder

    def build_intermediate(self):
        inputs = tf.keras.Input(shape=(3,))
        h = layers.Dense(64, activation='relu')(inputs)
        v = 9.9 * tf.nn.sigmoid(layers.Dense(3)(h))  # 3-vector v
        z_mu = v[:, 0] / 10.0 + v[:, 1] / 100.0 + v[:, 2] / 1000.0
        z_mu = tf.expand_dims(z_mu, -1)  # Ensure z_mu has the right shape
        z_log_sigma = layers.Dense(1, name='z_log_sigma')(h)
        intermediate = tf.keras.Model(inputs, [z_mu, z_log_sigma, v], name='intermediate')
        return intermediate

    def build_decoder(self):
        latent_inputs = tf.keras.Input(shape=(3,))
        x = layers.Dense(64 * 32 * 32, activation='relu')(latent_inputs)
        x = layers.Reshape((32, 32, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling2D()(x)
        x = layers.Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling2D()(x)
        outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
        decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
        return decoder

    def sample(self, z_mu, z_log_sigma):
        epsilon = tf.random.normal(shape=(tf.shape(z_mu)[0], 1))
        return tf.nn.relu(z_mu + tf.exp(0.5 * z_log_sigma) * epsilon)  # Ensure sampled z is always > 0

    def softfloor_series(self, z_scalar, n_terms=3):
        # Adjusted to produce a 3-vector
        results = []
        for _ in range(n_terms):
            results.append(self.softfloor.forward(z_scalar))
            z_scalar = (z_scalar - self.softfloor.forward(z_scalar)) * 10
        return tf.stack(results, axis=1)

    def call(self, inputs, return_components=False):
        z_3_vector = self.encoder(inputs)
        z_mu, sigma, v = self.intermediate(z_3_vector)
        z_scalar = self.sample(z_mu, sigma)
        z_processed = self.softfloor_series(z_scalar)
        reconstructed = self.decoder(z_processed)

        if return_components:
            return reconstructed, z_3_vector, z_mu, sigma, z_scalar, v
        else:
            return reconstructed, z_mu, sigma

    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction, z_mu, z_log_sigma = self(data, training=True)  # Forward pass
            reconstruction_loss = tf.reduce_mean(
                losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 128 * 128

            # Compute KL divergence loss using z_log_sigma
            kl_loss = 1 + z_log_sigma - tf.square(z_mu) - tf.exp(z_log_sigma)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}




if __name__=="__main__":

    # The sentences
    sentences = [
        "Dr. Smith's email, j.smith@university.edu, includes a period and an \"at\" symbol (@), and his office number is 456-7890.",
        "\"The quick brown fox jumps over the lazy dog,\" exclaimed Jane, marveling at how the sentence includes every letter of the alphabet!",
        "In her recipe book, Emily noted: \"Mix 2 cups of flour, 1.5 cups of sugar, and 3/4 teaspoon of salt; then, add 100g of butter.\"",
        "The company's revenue increased by 15% in Q4 2023; however, the CEO warned, \"Expect a -5% adjustment due to market fluctuations.\"",
        "\"Did you know,\" asked Tom, \"that '#' is called a 'hash', '&' is an 'ampersand', and '~' is a 'tilde'?\"",
        "The mathematician wrote, \"When x > 3 and y < 5, the function f(x, y) = 3x + 2y yields values between 27 and 49.\"",
        "In his novel, the author described a peculiar device: \"It's called the 'Chrono-Synclastic Infundibulum'a portal to parallel universes!\"",
        "The file path C:\\Users\\Lachlan\\Documents\\Project_XYZ\\Report_v3.1.txt includes various punctuation marks used in computer directories.",
        "On the sign-up form, fields such as name, email (e.g., john.doe@example.com), and password (e.g., Pa$$w0rd!) were mandatory.",
        "The biologist noted, \"Species 'Homo sapiens' (human) is distinct in its use of complex symbols, such as alphabets (A-Z) and numbers (0-9).\""
    ]

    # Tokenizing each sentence
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

    # Create a set of unique tokens
    unique_tokens = set(token for sentence in tokenized_sentences for token in sentence)

    # Encode each unique token into a 128x128 matrix
    token_matrices = {token: encode_word(token) for token in unique_tokens}

    # Optional: Convert the dictionary of matrices to a list or array if needed
    matrix_dataset = list(token_matrices.values())

    # Printing the tokenized sentences
    for i, tokens in enumerate(tokenized_sentences):
        print(f"Sentence {i+1} tokens:")
        print(tokens)
        print("\n")

    # Test the functions
    word = 'a'
    encoded_matrix = encode_word(word)
    decoded_word = decode_matrix(encoded_matrix)

    print(f'Encoded Matrix for "{word}":\n{encoded_matrix}')
    print(f'Decoded Word: {decoded_word}')
    # sns.heatmap(encoded_matrix)


    # Print some of the encoded matrices
    for token, matrix in list(token_matrices.items())[:5]:  # print first 5 for brevity
        print(f'Encoded Matrix for "{token}":\n{matrix}\n')

    # Decoding back (optional)
    for token, matrix in list(token_matrices.items())[:5]:  # print first 5 for brevity
        decoded_word = decode_matrix(matrix)
        print(f'Decoded Word for matrix of "{token}": {decoded_word}')

    # Instantiate and compile the VAE
    vae = CustomVAE()
    vae.compile(optimizer=optimizers.Adam())

    # Assuming x_train is your dataset of 128x128 matrices
    x_train = np.array(matrix_dataset).astype('float32') # / 255.0  # Normalize to [0, 1]
    x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension


    # Instantiate your custom callback
    verbose_callback = VerboseEveryTenEpochs()
    # Train the VAE
    vae.fit(
        x_train,
        verbose=0,  # This disables the default epoch-by-epoch output 
        epochs=1000, batch_size=128, callbacks=[verbose_callback])

    print("x_train shape: ", x_train.shape)


    # Choose an example
    example_index = 99  # Change this index to see different examples
    example_input = x_train[example_index:example_index+1]  # Get a single example
    print(example_input.shape)
    input_word = decode_matrix(example_input[0, :,:, 0])  # Replace with the actual word or leave it as None

    # Visualize the components
    visualize_components(vae, example_input, input_word=input_word)




