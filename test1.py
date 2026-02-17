import streamlit as st
import tensorflow as tf
import numpy as np
import soundfile as sf
from tensorflow.keras.layers import GRU

# =============================
# PARAMETERS (FROM TRAINING)
# =============================
frame_length = 256
frame_step = 160
fft_length = 384
sample_rate = 16000

# =============================
# CUSTOM GRU TO FIX LOADING LEGACY H5
# =============================
@tf.keras.utils.register_keras_serializable()
class CustomGRU(GRU):
    def __init__(self, *args, **kwargs):
        # Remove unsupported args from legacy models
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model_checkpoint25.h5",
        compile=False,
        custom_objects={"GRU": CustomGRU}
    )

model = load_model()

# =============================
# VOCAB (COPY EXACTLY AND MAKE UNIQUE)
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

# Make sure vocabulary is unique
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
# CTC DECODE
# =============================
def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    decoded = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    text = tf.strings.reduce_join(num_to_char(decoded[0])).numpy().decode("utf-8")
    return text

# =============================
# STREAMLIT UI
# =============================
st.title("Amharic ASR – DeepSpeech2 Demo 🎙️")

audio_file = st.file_uploader("Upload or record Amharic speech (.wav, 16kHz)", type=["wav"])

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
