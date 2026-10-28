"""Streamlit WSD app using the saved Kaggle classifier, BERT and spaCy POS tagging.

This reconstructs target embeddings with mean pooling of BERT subword hidden states.
Verify this matches the feature extraction and label mapping used during training.
"""
import glob
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import spacy
import streamlit as st
import torch
import nltk
from nltk.corpus import wordnet as wn
from transformers import AutoModel, AutoTokenizer

st.set_page_config(page_title="Word Sense Disambiguation", page_icon="📖", layout="centered")

POS_MAP = {"NOUN": "n", "PROPN": "n", "VERB": "v", "AUX": "v", "ADJ": "a", "ADV": "r"}
POS_LABELS = {"n": "Noun", "v": "Verb", "a": "Adjective", "r": "Adverb"}


def find_artifacts() -> Path:
    # Works with Kaggle attached notebook output or a local copied artifacts/ folder.
    env = os.getenv("WSD_ARTIFACTS_DIR", "")
    candidates = ([Path(env)] if env else []) + [
        Path("artifacts"), Path("wsd_simple/artifacts"), Path("/kaggle/working/wsd_simple/artifacts")
    ]
    candidates += [Path(p).parent for p in glob.glob("/kaggle/input/**/wsd_model.joblib", recursive=True)]
    for directory in candidates:
        if (directory / "wsd_model.joblib").is_file():
            return directory
    raise FileNotFoundError(
        "Could not find artifacts/wsd_model.joblib. In Kaggle, add your saved notebook "
        "output as Input. Locally, copy the output's artifacts folder beside app.py, "
        "or set WSD_ARTIFACTS_DIR."
    )


@st.cache_resource(show_spinner="Loading saved WSD model...")
def load_models():
    directory = find_artifacts()
    for name in ["sense_mappings.json", "run_config.json"]:
        if not (directory / name).exists():
            raise FileNotFoundError(f"Missing {directory / name}")
    bundle = joblib.load(directory / "wsd_model.joblib")
    if not isinstance(bundle, dict) or "models" not in bundle or "scaler" not in bundle:
        raise ValueError("Expected saved bundle containing 'models' and 'scaler'.")
    models = bundle["models"]
    scaler = bundle["scaler"]
    mappings = json.loads((directory / "sense_mappings.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(directory / "tokenizer"), local_files_only=True, use_fast=True)
    encoder = AutoModel.from_pretrained(str(directory / "encoder"), local_files_only=True)
    encoder.eval()
    language_model = spacy.load("en_core_web_sm")
    try:
        wn.synsets("bank")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        wn.synsets("bank")
    return models, scaler, mappings, config, tokenizer, encoder, language_model, directory


def match_target(doc, target, occurrence):
    # Exact token matches, which also provide reliable character spans for BERT alignment.
    matches = [tok for tok in doc if tok.text.casefold() == target.strip().casefold()]
    if not matches:
        raise ValueError("Target word was not found as a complete word in the sentence.")
    if not (1 <= occurrence <= len(matches)):
        raise ValueError(f"Occurrence must be between 1 and {len(matches)}.")
    return matches[occurrence - 1], len(matches)


def encode_target(sentence, start, end, tokenizer, encoder, max_length):
    if not tokenizer.is_fast:
        raise ValueError("A fast Hugging Face tokenizer is required for character-offset alignment.")
    batch = tokenizer(
        sentence, return_tensors="pt", return_offsets_mapping=True,
        truncation=True, max_length=max_length
    )
    offsets = batch.pop("offset_mapping")[0].tolist()
    selected = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    if not selected:
        raise ValueError("Target word was truncated or could not be aligned to BERT tokens.")
    with torch.inference_mode():
        output = encoder(**batch).last_hidden_state[0]
        vec = output[selected].mean(dim=0).cpu().numpy()
    return vec.reshape(1, -1).astype(np.float32)


def extract_label_mapping(mappings):
    """Return numeric classifier label -> WordNet sense string.

    Typical formats: {id_to_sense: ...}, {label_to_sense: ...},
    {sense_to_id: ...}, {sense_to_label: ...}, or direct dictionaries.
    """
    if not isinstance(mappings, dict):
        raise ValueError("sense_mappings.json must contain a JSON object.")
    sense_pattern = re.compile(r"^[\w-]+\.[nvars]\.\d+$")

    def is_sense(v):
        return isinstance(v, str) and bool(sense_pattern.match(v))

    def convert(mapping):
        if not isinstance(mapping, dict) or not mapping:
            return None
        if all(is_sense(v) for v in mapping.values()):
            return {str(k): v for k, v in mapping.items()}
        if all(is_sense(k) for k in mapping.keys()):
            return {str(v): k for k, v in mapping.items()}
        return None

    preference = [
        "id_to_sense", "label_to_sense", "index_to_sense", "idx_to_sense",
        "sense_to_id", "sense_to_label", "sense_to_idx", "label_mapping",
        "sense_to_index", "index_to_label", "label_to_index"
    ]
    for name in preference:
        if name in mappings:
            result = convert(mappings[name])
            if result:
                return result
    result = convert(mappings)
    if result:
        return result
    for value in mappings.values():
        result = convert(value)
        if result:
            return result
    raise ValueError(
        "Unrecognized sense_mappings.json format. Top-level keys: "
        + ", ".join(map(str, mappings.keys()))
        + ". Update extract_label_mapping() to match your training notebook."
    )


def class_to_sense(label, mapping):
    s = str(label)
    if re.fullmatch(r"[\w-]+\.[nvars]\.\d+", s):
        return s
    if s in mapping:
        return mapping[s]
    if isinstance(label, (int, np.integer)) and str(int(label)) in mapping:
        return mapping[str(int(label))]
    return None


def predict_wsd(sentence, target, occurrence, chosen_pos, loaded):
    models, scaler, mappings, config, tokenizer, encoder, language_model, _ = loaded
    doc = language_model(sentence)
    token, count = match_target(doc, target, occurrence)
    detected_pos = POS_MAP.get(token.pos_)
    pos = detected_pos if chosen_pos == "Automatic" else chosen_pos
    if pos is None:
        raise ValueError(f"Unsupported detected grammatical role: {token.pos_}. Try manual override.")

    lemma = token.lemma_.lower()
    key = (lemma, pos)
    if key not in models:
        raise ValueError(
            f"There is no trained classifier for lemma='{lemma}', POS='{pos}'. "
            "This model was trained only for lemma/POS pairs present in SemCor."
        )
    classifier = models[key]
    max_length = config.get("max_length", config.get("bert_max_length", 128))
    try:
        max_length = int(max_length)
    except (TypeError, ValueError):
        max_length = 128

    X = encode_target(sentence, token.idx, token.idx + len(token.text), tokenizer, encoder, max_length)
    if getattr(scaler, "n_features_in_", X.shape[1]) != X.shape[1]:
        raise ValueError("Saved scaler feature dimension does not match the BERT representation.")
    X_scaled = scaler.transform(X)
    if hasattr(classifier, "n_features_in_") and classifier.n_features_in_ != X_scaled.shape[1]:
        raise ValueError("Classifier feature dimension does not match the scaled BERT representation.")

    probabilities = np.asarray(classifier.predict_proba(X_scaled))[0]
    labels = classifier.classes_
    if len(probabilities) != len(labels):
        raise ValueError("Classifier returned inconsistent probabilities and classes.")

    # Per-lemma classifiers generally use full WordNet synset IDs as labels.
    # Keep the local classifier label space, not a global class mapping.
    try:
        label_map = extract_label_mapping(mappings)
    except ValueError:
        label_map = {}

    ranked = []
    for label, probability in zip(labels, probabilities):
        sense = class_to_sense(label, label_map)
        if sense is None:
            raise ValueError(
                f"Could not resolve classifier label {label!r} to a WordNet sense. "
                "Please provide the structure of sense_mappings.json."
            )
        try:
            synset = wn.synset(sense)
        except Exception as exc:
            raise ValueError(f"Classifier returned invalid WordNet synset {sense!r}.") from exc
        ranked.append((synset.name(), synset.definition(), float(probability)))

    ranked.sort(key=lambda item: item[2], reverse=True)
    return {
        "target": token.text, "lemma": lemma, "pos": pos,
        "spacy_pos": token.pos_, "number_of_occurrences": count,
        "candidate_count": len(ranked), "top": ranked[:3],
        "model_type": type(classifier).__name__,
    }


st.title("📖 Word Sense Disambiguation")
st.caption("Predict a word's WordNet meaning with saved BERT embeddings + Logistic Regression")
st.write("Enter a sentence and choose which word to disambiguate. Its grammatical role is detected automatically.")

try:
    loaded = load_models()
except Exception as exc:
    st.error(f"Model setup failed: {exc}")
    st.stop()

sentence = st.text_area("Sentence", "She deposited her money in the bank.", height=90)
target = st.text_input("Target word", "bank").strip()

possible = []
try:
    doc = loaded[6](sentence)
    possible = [t for t in doc if target and t.text.casefold() == target.casefold()]
except Exception:
    pass

occurrence = st.selectbox(
    "Occurrence", options=list(range(1, len(possible) + 1)) or [1],
    format_func=lambda k: f"Occurrence {k}"
)

detected = POS_MAP.get(possible[occurrence - 1].pos_) if possible else None
if possible:
    label = POS_LABELS.get(detected, "Unsupported")
    st.info(f"Automatically detected grammatical role: **{label}**")
else:
    st.warning("Target word not found. Enter a word that appears in the sentence.")

with st.expander("Advanced: correct the detected grammatical role"):
    override = st.selectbox(
        "POS mode", ["Automatic", "n", "v", "a", "r"],
        format_func=lambda p: "Automatic (spaCy)" if p == "Automatic" else POS_LABELS[p]
    )

if st.button("Predict meaning", type="primary", use_container_width=True):
    if not sentence.strip() or not target:
        st.error("Provide a sentence and target word.")
    else:
        try:
            with st.spinner("Predicting..."):
                answer = predict_wsd(sentence, target, occurrence, override, loaded)
            st.success("Prediction complete")
            st.subheader("Predicted meaning")
            sense, definition, probability = answer["top"][0]
            st.markdown(f"**Sense:** `{sense}`")
            st.write(definition)
            st.caption(
                f"Lemma: {answer['lemma']} · POS: {POS_LABELS[answer['pos']]} · "
                f"Trained candidate senses: {answer['candidate_count']}"
            )
            st.metric("Classifier score (not calibrated)", f"{probability:.1%}")
            if len(answer["top"]) > 1:
                st.markdown("**Other candidate senses**")
                for alternate_name, alternate_definition, score in answer["top"][1:]:
                    st.write(f"`{alternate_name}` — {alternate_definition} ({score:.1%})")
            if answer["candidate_count"] == 1:
                st.warning("Only one trained candidate sense is available; this is not proof the sense is correct.")
            st.caption("The displayed probabilities are uncalibrated classifier estimates, not overall accuracy.")
        except Exception as exc:
            st.error(f"Prediction couldn't be completed: {exc}")
            st.info("Check that your sense mapping and BERT embedding extraction match the original training notebook.")
