"""predict() returns a 5-character string and a confidence in [0, 1]."""

import numpy as np

from captcha_recognizer import ModelConfig, create_model, predict


def test_predict_returns_text_and_confidence():
    config = ModelConfig()
    model = create_model(config, learning_rate=1e-3)
    image = np.zeros((config.image_height, config.image_width), dtype=np.uint8)
    text, confidence = predict(image, model, config)
    assert isinstance(text, str)
    assert len(text) == 5
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0
    # Dropout is off outside training, so a second read does not change.
    second_text, second_confidence = predict(image, model, config)
    assert second_text == text
    assert second_confidence == confidence


if __name__ == "__main__":
    test_predict_returns_text_and_confidence()
    print("predict test passed")
