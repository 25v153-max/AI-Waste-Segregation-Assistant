def safe_waste_assistant(
    image_class=None,
    image_conf=None,
    audio_class=None,
    audio_conf=None,
    threshold=70
):
    valid_classes = {
        "plastic", "paper", "cardboard",
        "glass", "metal", "organic", "background"
    }

    if image_class is not None:
        image_class = image_class.lower().strip()
        if image_class not in valid_classes or image_conf is None:
            return "Invalid image prediction. Please verify manually."

    if audio_class is not None:
        audio_class = audio_class.lower().strip()
        if audio_class not in valid_classes or audio_conf is None:
            return "Invalid audio prediction. Please verify manually."

    if image_class is None and audio_class is None:
        return "No valid sensor predictions are available."

    if image_class is not None and audio_class is not None:
        if image_class == "background" or audio_class == "background":
            return "Background detected. No reliable waste category can be determined."

        if image_class != audio_class:
            return (
                "The image and audio predictions conflict. "
                "Manual verification is required before disposal."
            )

        if image_conf < threshold or audio_conf < threshold:
            return (
                f"Both sensors predict {image_class}, but confidence is low. "
                "Please verify manually."
            )

        return (
            f"Both sensors agree on {image_class}. "
            f"Place the item in the appropriate {image_class} waste bin."
        )

    if image_class is not None:
        if image_class == "background":
            return "No waste item could be identified."
        if image_conf < threshold:
            return "The image prediction is uncertain. Please verify manually."
        return f"The image sensor predicts {image_class}. Please verify before disposal."

    if audio_class is not None:
        if audio_class == "background":
            return "Background sound detected. No waste category can be determined."
        if audio_conf < threshold:
            return "The audio prediction is uncertain. Please verify manually."
        return f"The audio sensor predicts {audio_class}. Please verify before disposal."


if __name__ == "__main__":
    print("\n========== ECOASSIST SENSOR FUSION ==========")

    image_class = input("Enter image prediction: ").strip()
    image_conf = float(input("Enter image confidence (%): "))

    audio_class = input("Enter audio prediction: ").strip()
    audio_conf = float(input("Enter audio confidence (%): "))

    decision = safe_waste_assistant(
        image_class=image_class,
        image_conf=image_conf,
        audio_class=audio_class,
        audio_conf=audio_conf
    )

    print("\n========== FINAL SAFETY DECISION ==========")
    print(decision)
    print("===========================================")
