import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

def mfccs(wavs):
    import librosa
    embeddings = []
    for f in tqdm(wavs):
        x, fs = sf.read(f)
        y = librosa.feature.mfcc(x, sr=fs, n_mfcc=39)
        embeddings.append(np.mean(y,axis=-1))
    return embeddings

def xvectors(wavs):
    from speechbrain.pretrained import EncoderClassifier
    import torchaudio

    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    def extract_xvector(x):
        signal, fs = torchaudio.load(x)
        embeddings = classifier.encode_batch(signal)
        return embeddings[0].mean(dim=0).detach().cpu().numpy()


    embeddings = [extract_xvector(x) for x in tqdm([str(Path(f).resolve()) for f in wavs])]
    return embeddings

feature_extractors = {
    'mfcc': mfccs,
    'xvector': xvectors
}