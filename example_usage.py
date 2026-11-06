"""Build the captcha CNN with the training learning rate."""

from captcha_recognizer import Config, create_model


def main():
    config = Config()
    model = create_model(config.model, config.training.learning_rate)
    print("outputs:", ", ".join(model.output_names))
    print("parameters:", model.count_params())


if __name__ == "__main__":
    main()
