import subprocess
import re

def get_prediction(script):
    result = subprocess.run(
        ["python", script],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(script + " failed")

    category = re.search(
        r"Predicted (?:waste|category):\s*(\w+)",
        result.stdout,
        re.IGNORECASE
    )
    confidence = re.search(
        r"Confidence:\s*([\d.]+)%",
        result.stdout
    )

    if not category or not confidence:
        raise ValueError("Could not read prediction from " + script)

    return category.group(1).lower(), float(confidence.group(1))


print("\n========== ECOASSIST AI ==========")

image_class, image_conf = get_prediction("predict_image.py")
audio_class, audio_conf = get_prediction("predict_audio.py")

print("\nIMAGE:", image_class, f"{image_conf:.2f}%")
print("AUDIO:", audio_class, f"{audio_conf:.2f}%")

from sensor_fusion import safe_waste_assistant

decision = safe_waste_assistant(
    image_class=image_class,
    image_conf=image_conf,
    audio_class=audio_class,
    audio_conf=audio_conf
)

print("\n========== FINAL ECOASSIST DECISION ==========")
print(decision)
print("==============================================")
