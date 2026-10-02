import soundfile as sf
from dia.model import Dia

model = Dia.from_pretrained("nari-labs/Dia-1.6B")
index = 0

while True:
    text = input("Enter the command: ")
    if text == "q":
        break

    print("Generating..........")
    output = model.generate(text)
    print("Writing..........")
    sf.write(f"simple{index}.mp3", output, 44100)
    index += 1
