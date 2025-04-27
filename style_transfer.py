import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import soundfile as sf
from keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image
import os

# ======== USER INPUT ========
# Give your own paths for content and style
content_path = "D:/Final-year-project/Music-Generator-APP/relaxing-trumpet-music-for-stress-relief-and-deep-relaxation-281069.wav"
style_path = "D:/Final-year-project/Music-Generator-APP/style.wav"
# =============================

# Helper to play audio in VSCode
def play_audio(data, sr=22050):
    import IPython.display as ipd
    display(ipd.Audio(data, rate=sr))

# === Set image size
size = (1025, 430)

# === Function to load and play audio
def load_and_play(path):
    x, sr = librosa.load(path)
    # Commented because VSCode terminal can't display Audio widget
    # play_audio(x, sr)
    return x, sr

# === Audio to image conversion
def audio_to_img(path, size):
    x, sr = librosa.load(path)
    stft = librosa.stft(x)
    mag, phase = librosa.magphase(stft)
    mag = np.log1p(mag)

    mag_min, mag_max = mag.min(), mag.max()
    mag_norm = (mag - mag_min) / (mag_max - mag_min)

    mag_norm = mag_norm[:size[0], :size[1]]

    data = (mag_norm * 255).astype(np.uint8)
    img = Image.fromarray(data, mode='L')

    return img, mag_min, mag_max, phase

# === Image back to audio
def image_to_audio(img, mag_min, mag_max):
    mag_norm = np.array(img, dtype=np.float32) / 255
    mag = mag_norm * (mag_max - mag_min) + mag_min
    mag = np.exp(mag) - 1
    return librosa.griffinlim(mag)

# === Main process ===
# Load content and style
x_content, sr_content = load_and_play(content_path)
x_style, sr_style = load_and_play(style_path)

# Convert audio to images
content_img, content_min, content_max, content_phase = audio_to_img(content_path, size)
style_img, style_min, style_max, style_phase = audio_to_img(style_path, size)

# Prepare tensors
content_np = np.array(content_img).T[None, None, :, :]
style_np = np.array(style_img).T[None, None, :, :]

content_tensor = tf.convert_to_tensor(content_np, dtype=tf.float32)
style_tensor = tf.convert_to_tensor(style_np, dtype=tf.float32)

# Model parameters
BATCH, HEIGHT, WIDTH, CHANNELS = content_tensor.shape
FILTERS = 4096

input_shape = (HEIGHT, WIDTH, CHANNELS)

# === Custom kernel initializer
def custom_kernel_initializer(shape, dtype=None):
    std = np.sqrt(2) * np.sqrt(2.0 / ((CHANNELS + FILTERS) * 11))
    kernel = np.random.randn(1, 11, shape[-2], shape[-1]) * std
    return tf.constant(kernel, dtype=dtype)

# === Create model
def create_model(input_shape):
    inputs = Input(shape=input_shape)

    outputs = Conv2D(
        filters=FILTERS,
        kernel_size=(1, 11),
        padding='same',
        activation='relu',
        kernel_initializer=custom_kernel_initializer
    )(inputs)

    return Model(inputs=inputs, outputs=outputs)

# === Build model
model = create_model(input_shape)
model.summary()

# === Loss functions
def gram_matrix(x):
    feats = tf.reshape(x, (-1, x.shape[-1]))
    return tf.matmul(tf.transpose(feats), feats)

def get_style_loss(A, B):
    gram_A = gram_matrix(A)
    gram_B = gram_matrix(B)
    return tf.sqrt(tf.reduce_sum(tf.square(gram_A - gram_B)))

def get_content_loss(A, B):
    return tf.sqrt(tf.reduce_sum(tf.square(A - B)))

# Extract features
content_features = model(content_tensor)
style_features = model(style_tensor)

# Create random initial generation
gen_np = tf.random.normal((1, *input_shape))
gen = tf.Variable(gen_np)

steps_counter = 0
STEPS = 1000

optimizer = Adam(learning_rate=1.0)

# === Training loop
for i in range(STEPS):
    with tf.GradientTape() as tape:
        tape.watch(gen)

        gen_features = model(gen)

        content_loss = get_content_loss(gen_features, content_features)
        style_loss = get_style_loss(gen_features, style_features) * 0.001

        loss = content_loss + style_loss

    gradients = tape.gradient(loss, [gen])
    optimizer.apply_gradients(zip(gradients, [gen]))

    if i % 50 == 0:
        print(f"Step: {i} | loss: {loss.numpy()} | Content Loss: {content_loss.numpy()} | Style Loss: {style_loss.numpy()}")

steps_counter += STEPS

# === Final results
gen_np = np.squeeze(gen.numpy()).T
gen_img = Image.fromarray(gen_np).convert('L')

# Save and display
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# Save output images
gen_img.convert('RGB').save('outputs/output.jpg')

# Plot
plt.figure(figsize=(10, 8))

plt.subplot(1, 3, 1)
plt.title("Content")
plt.imshow(content_img)

plt.subplot(1, 3, 2)
plt.title("Style")
plt.imshow(style_img)

plt.subplot(1, 3, 3)
plt.title("Generated")
plt.imshow(gen_img)

plt.tight_layout()
plt.savefig('outputs/comparison.png')
plt.show()

# Convert generated image to audio
x = image_to_audio(gen_img, content_min, content_max)

# Save output audio
sf.write('outputs/output.wav', x, 22050)

# Save weights
np.save('outputs/weights.npy', gen.numpy())

print("✅ Process completed! Outputs saved inside 'outputs/' folder.")
