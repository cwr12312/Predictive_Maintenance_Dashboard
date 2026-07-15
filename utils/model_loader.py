"""
utils/model_loader.py
======================
Loads the six trained models (three .keras, three .pth) exactly as saved.
Handles deserialization of models saved with newer TensorFlow versions.
"""

from __future__ import annotations

import time
import numpy as np
import streamlit as st
import tensorflow as tf
from pathlib import Path
import warnings
import json
warnings.filterwarnings('ignore')

from config import MODEL_REGISTRY, N_CLASSES, N_FEATURES
from utils.torch_architectures import (
    FaultTransformer,
    OptimizedmamlTransformer,
    OptimizedFBCLTransformer,
    count_boosting_heads,
)

# Import custom layers for deserialization
from utils.custom_layers import CUSTOM_LAYERS


# ============================================================
# TENSORFLOW MODEL LOADER WITH CUSTOM DESERIALIZATION
# ============================================================

def safe_deserialize_dtype_policy(config):
    """Handle DTypePolicy deserialization safely."""
    if isinstance(config, dict) and config.get('class_name') == 'DTypePolicy':
        return tf.keras.mixed_precision.Policy(config.get('config', {}).get('name', 'float32'))
    return config


def fix_config_for_loading(config):
    """Recursively fix config for loading by removing unsupported parameters."""
    if isinstance(config, dict):
        # Remove problematic keys
        config.pop('quantization_config', None)
        config.pop('shared_object_id', None)
        
        # Fix DTypePolicy
        for key, value in list(config.items()):
            if isinstance(value, dict):
                fix_config_for_loading(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        fix_config_for_loading(item)
    
    return config


def load_keras_model_compatible(path_str: str):
    """
    Load Keras model with compatibility handling for different TensorFlow versions.
    Uses multiple strategies to load the model.
    """
    path = Path(path_str)
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path_str}")
    
    # ============================================================
    # STRATEGY 1: Load with custom objects and custom layers
    # ============================================================
    try:
        # Register all custom objects
        with tf.keras.utils.custom_object_scope(CUSTOM_LAYERS):
            model = tf.keras.models.load_model(path_str, compile=False)
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"✅ Loaded {path.name} with custom objects")
        return model
    except Exception as e:
        print(f"⚠️ Custom object loading failed: {e}")
    
    # ============================================================
    # STRATEGY 2: Load without compilation (just architecture)
    # ============================================================
    try:
        with tf.keras.utils.custom_object_scope(CUSTOM_LAYERS):
            model = tf.keras.models.load_model(
                path_str, 
                compile=False,
                custom_objects={
                    'DTypePolicy': tf.keras.mixed_precision.Policy,
                    'Dense': tf.keras.layers.Dense,
                    'Conv2D': tf.keras.layers.Conv2D
                }
            )
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"✅ Loaded {path.name} with basic custom objects")
        return model
    except Exception as e:
        print(f"⚠️ Basic custom objects failed: {e}")
    
    # ============================================================
    # STRATEGY 3: Load using H5 format (legacy)
    # ============================================================
    try:
        model = tf.keras.models.load_model(
            path_str, 
            compile=False,
            custom_objects={
                'DTypePolicy': tf.keras.mixed_precision.Policy,
                'GlorotUniform': tf.keras.initializers.GlorotUniform,
                'Zeros': tf.keras.initializers.Zeros,
                'L2': tf.keras.regularizers.L2,
            }
        )
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"✅ Loaded {path.name} with legacy format")
        return model
    except Exception as e:
        print(f"⚠️ Legacy format failed: {e}")
    
    # ============================================================
    # STRATEGY 4: Recreate model from scratch and load weights
    # ============================================================
    try:
        model = recreate_model_from_name(path.stem)
        if model is not None:
            # Try to load weights
            model.load_weights(path_str)
            print(f"✅ Recreated {path.name} from architecture and loaded weights")
            return model
    except Exception as e:
        print(f"⚠️ Recreate and load weights failed: {e}")
    
    # ============================================================
    # STRATEGY 5: Final attempt - use keras.models.load_model with ignore
    # ============================================================
    try:
        # Try loading with safe globals
        model = tf.keras.models.load_model(
            path_str,
            compile=False,
            custom_objects={
                'DTypePolicy': tf.keras.mixed_precision.Policy,
                'Dense': tf.keras.layers.Dense,
                'Conv2D': tf.keras.layers.Conv2D,
                'BatchNormalization': tf.keras.layers.BatchNormalization,
                'Dropout': tf.keras.layers.Dropout,
                'MaxPooling2D': tf.keras.layers.MaxPooling2D,
                'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
                'InputLayer': tf.keras.layers.InputLayer,
                'GlorotUniform': tf.keras.initializers.GlorotUniform,
                'Zeros': tf.keras.initializers.Zeros,
                'Ones': tf.keras.initializers.Ones,
                'L2': tf.keras.regularizers.L2,
            }
        )
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"✅ Loaded {path.name} with comprehensive custom objects")
        return model
    except Exception as e:
        raise RuntimeError(f"All loading strategies failed for {path.name}: {str(e)}")


def recreate_model_from_name(model_name):
    """
    Recreate model architecture from scratch based on model name.
    This is a fallback when loading the saved model fails.
    """
    from models.tensorflow_models import (
        create_cnn_2d_model,
        create_lstm_model,
        create_transformer_model
    )
    
    creators = {
        'cnn_2d_model': create_cnn_2d_model,
        'cnn_2d_model_fixed': create_cnn_2d_model,
        'lstm_model': create_lstm_model,
        'lstm_model_fixed': create_lstm_model,
        'final_transformer_model': create_transformer_model,
        'transformer_model_fixed': create_transformer_model,
    }
    
    # Try exact match
    if model_name in creators:
        model = creators[model_name]()
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    # Try partial match
    for key, creator in creators.items():
        if key in model_name or model_name in key:
            model = creator()
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
    
    return None


class ModelWrapper:
    """Common interface around a loaded Keras or PyTorch model."""

    def __init__(self, name, framework, model, extra=None):
        self.name = name
        self.framework = framework
        self.model = model
        self.extra = extra or {}

    def _reshape_for_model(self, X: np.ndarray) -> np.ndarray:
        """Reshape input based on model requirements."""
        cfg = MODEL_REGISTRY[self.name]
        shape = cfg["input_shape"]
        n = X.shape[0]
        
        if self.name == "2D CNN":
            pad = int(np.prod(shape[:2])) - X.shape[1]
            Xp = np.pad(X, ((0, 0), (0, max(pad, 0))), mode="constant")
            return Xp.reshape((n, shape[0], shape[1], shape[2]))
        
        elif self.name == "LSTM":
            return X.reshape((n, shape[0], shape[1]))
        
        elif self.name == "Transformer":
            return X.reshape((n, shape[0], shape[1]))
        
        return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities from the model."""
        X = np.asarray(X, dtype=np.float32)
        
        if self.framework == "keras":
            try:
                Xr = self._reshape_for_model(X)
                probs = self.model.predict(Xr, verbose=0)
                return np.asarray(probs)
            except Exception as e:
                print(f"Keras prediction error for {self.name}: {e}")
                try:
                    probs = self.model.predict(X, verbose=0)
                    return np.asarray(probs)
                except:
                    raise

        # PyTorch models
        import torch
        Xt = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            if self.name == "FBCL":
                logits = self.model(Xt, use_boosting=True)
            elif self.name == "MAML":
                logits = self.model(Xt, mask=None)
            else:
                logits = self.model(Xt)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray):
        probs = self.predict_proba(X)
        return probs.argmax(axis=1), probs

    def timed_predict(self, X: np.ndarray):
        t0 = time.perf_counter()
        preds, probs = self.predict(X)
        elapsed = (time.perf_counter() - t0) * 1000.0
        per_sample = elapsed / max(1, X.shape[0])
        return preds, probs, per_sample


# ============================================================
# MAIN MODEL LOADER WITH CACHING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_keras_model(path_str: str):
    """Load Keras model with compatibility handling."""
    return load_keras_model_compatible(path_str)


@st.cache_resource(show_spinner=False)
def load_torch_checkpoint(model_name: str, path_str: str):
    """Load PyTorch model checkpoint."""
    import torch
    
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path_str}")
    
    try:
        ckpt = torch.load(path_str, map_location="cpu", weights_only=False)
    except Exception as e:
        try:
            ckpt = torch.load(path_str, map_location="cpu", weights_only=True)
        except:
            raise RuntimeError(f"Failed to load PyTorch checkpoint: {e}")
    
    n_features = ckpt.get("n_features", N_FEATURES)
    n_classes = ckpt.get("n_classes", N_CLASSES)
    state_dict = ckpt["model_state_dict"]

    if model_name == "Meta-SGD":
        arch = ckpt.get("arch", {})
        model = FaultTransformer(
            n_feat=n_features, n_classes=n_classes,
            d_model=arch.get("d_model", 64), n_heads=arch.get("n_heads", 4),
            n_layers=arch.get("n_layers", 3), dropout=arch.get("dropout", 0.15),
        )
        model.load_state_dict(state_dict)

    elif model_name == "MAML":
        cfg = ckpt.get("model_config", {})
        model = OptimizedmamlTransformer(
            n_features=n_features, n_classes=n_classes,
            d_model=cfg.get("d_model", 128), n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 4), dropout=cfg.get("dropout", 0.3),
        )
        model.load_state_dict(state_dict)
        model.switch_to_finetune()

    elif model_name == "FBCL":
        cfg = ckpt.get("model_config", {})
        model = OptimizedFBCLTransformer(
            n_features=n_features, n_classes=n_classes,
            d_model=cfg.get("d_model", 256), n_heads=cfg.get("n_heads", 16),
            n_layers=cfg.get("n_layers", 6), dropout=cfg.get("dropout", 0.15),
        )
        n_heads_saved = count_boosting_heads(state_dict)
        for _ in range(n_heads_saved):
            model.add_boosting_head()
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown torch model: {model_name}")

    model.eval()
    print(f"✅ Loaded {model_name} PyTorch model")
    return model, ckpt


@st.cache_resource(show_spinner=False)
def load_all_models() -> dict:
    """Loads every model in MODEL_REGISTRY once per session and caches them."""
    wrappers = {}
    errors = {}
    
    print("\n" + "="*60)
    print("🚀 LOADING MODELS")
    print("="*60)
    
    for name, cfg in MODEL_REGISTRY.items():
        try:
            print(f"\n📦 Loading {name}...")
            
            if cfg["framework"] == "keras":
                model = load_keras_model_compatible(str(cfg["file"]))
                wrappers[name] = ModelWrapper(name, "keras", model)
                print(f"✅ {name} loaded successfully")
                
            else:
                model, ckpt = load_torch_checkpoint(name, str(cfg["file"]))
                wrappers[name] = ModelWrapper(name, "torch", model, extra={"checkpoint": ckpt})
                print(f"✅ {name} loaded successfully")
                
        except Exception as exc:
            errors[name] = str(exc)
            print(f"❌ {name} failed to load: {exc}")
    
    print("\n" + "="*60)
    print(f"✅ Loaded {len(wrappers)} of {len(MODEL_REGISTRY)} models")
    if errors:
        print(f"⚠️ Failed: {list(errors.keys())}")
    print("="*60 + "\n")
    
    return {"models": wrappers, "errors": errors}


def get_model(name: str) -> ModelWrapper | None:
    registry = load_all_models()
    return registry["models"].get(name)


def get_load_errors() -> dict:
    registry = load_all_models()
    return registry["errors"]


def get_available_models() -> list:
    registry = load_all_models()
    return list(registry["models"].keys())


def is_model_available(name: str) -> bool:
    registry = load_all_models()
    return name in registry["models"]