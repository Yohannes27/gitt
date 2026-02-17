import streamlit as st
import tensorflow as tf
import numpy as np
import soundfile as sf
from tensorflow.keras.layers import GRU

# =============================
# PARAMETERS
# =============================
frame_length = 256
frame_step = 160
fft_length = 384
sample_rate = 16000

# =============================
# CUSTOM GRU (LEGACY H5 FIX)
# =============================
@tf.keras.utils.register_keras_serializable()
class CustomGRU(GRU):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "final_modelfi.h5",
        compile=False,
        custom_objects={"GRU": CustomGRU}
    )

model = load_model()

# =============================
# VOCAB
# =============================
characters = [x for x in 
"ሀሁሂሃሄህሆለሉሊላሌልሎሏቨቩቪቫቬቭቮቯጨጩጪጫጬጭጮጯሐሑሒሓሔሕሖመሙሚማሜምሞሟ"
"ሠሡሢሣሤሥሦሰሱሲሳሴስሶሷረሩሪራሬርሮሯሸሹሺሻሼሽሾሿ"
"ቀቁቂቃቄቅቆቋበቡቢባቤብቦቧ"
"ተቱቲታቴትቶቷቸቹቺቻቼችቾ"
"ነኑኒናኔንኖኗኘኙኚኛኜኝኞ"
"ገጉጊጋጌግጎጐጓ"
"ደዱዲዳዴድዶዷ"
"ኀኁኂኃኄኅኆኋ"
"ጀጁጂጃጄጅጆ"
"ዘዙዚዛዜዝዞዟ"
"ዠዡዢዣዤዥዦ"
"ጠጡጢጣጤጥጦጧ"
"ጰጱጲጴጵጶ"
"ጸጹጺጻጼጽጾ"
"አኡኢኣኤእኦ"
"ከኩኪካኬክኮኳ"
"ወዉዊዋዌውዎ"
"ዐዑዒዓዔዕዖ"
"የዩዪያዬይዮ"
"ፈፉፊፋፌፍፎፏ"
"ፐፑፒፓፔፕፖ'?! "]

characters = list(dict.fromkeys(characters))

char_to_num = tf.keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    invert=True,
    oov_token=""
)

# =============================
# AUDIO PREPROCESSING
# =============================
def preprocess_audio(audio):
    # 0.5s padding to prevent last-word cutoff
    pad = int(0.5 * sample_rate)
    audio = np.pad(audio, (0, pad))

    audio = tf.cast(audio, tf.float64)

    spectrogram = tf.signal.stft(
        audio,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length
    )

    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)

    mean = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    std = tf.math.reduce_std(spectrogram, 1, keepdims=True)

    spectrogram = (spectrogram - mean) / (std + 1e-10)
    return tf.expand_dims(spectrogram, axis=0)

# =============================
# CTC DECODER + HEURISTIC SPACING
# =============================
def decode_prediction(pred):
    # Softmax probabilities
    probs = tf.nn.softmax(pred[0]).numpy()

    # Sequence lengths
    input_len = np.full((1,), pred.shape[1], dtype=np.int32)

    # Greedy CTC decode
    decoded = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

    # Remove padding (-1)
    decoded = tf.boolean_mask(decoded[0], decoded[0] != -1)

    # Map to characters
    chars = num_to_char(decoded).numpy()
    text_chars = [c.decode("utf-8") for c in chars]

    # Confidence-based spacing
    confidence = np.max(probs, axis=1)
    output = []
    buffer = ""

    for i, ch in enumerate(text_chars):
        buffer += ch
        if i < len(confidence) and confidence[i] < 0.55:
            output.append(buffer)
            buffer = ""

    if buffer:
        output.append(buffer)

    # Join with single spaces
    text = " ".join(output)
    text = text.replace("  ", " ").strip()
    return text

# =============================
# STREAMLIT UI
# =============================
st.title("Code for Africa_Yohannes A. – Amharic ASR Demo 🎙️")

audio_file = st.file_uploader(
    "Upload or record Amharic speech (.wav, 16kHz)",
    type=["wav"]
)

if audio_file:
    audio, sr = sf.read(audio_file)

    if sr != sample_rate:
        st.warning("Audio must be 16kHz (same as training)")
        st.stop()

    st.audio(audio_file)

    features = preprocess_audio(audio)
    preds = model.predict(features)

    transcription = decode_prediction(preds)
    st.subheader("Transcription")
    st.success(transcription)
