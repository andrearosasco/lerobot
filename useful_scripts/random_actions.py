import time
import random
import platform
import subprocess

def say(text: str, blocking: bool = False):
    system = platform.system()

    if system == "Darwin":
        cmd = ["say", text]
    elif system == "Linux":
        cmd = ["spd-say", text]
        if blocking:
            cmd.append("--wait")
    elif system == "Windows":
        cmd = [
            "PowerShell",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')",
        ]
    else:
        raise RuntimeError("Unsupported operating system for text-to-speech.")


    # On Linux, creationflags doesn't exist, so we set it only for Windows
    kwargs = {}
    if system == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    
    subprocess.Popen(cmd, **kwargs)

if __name__ == "__main__":
    actions = ["stand", "wave", "shake", "handover", "look at the box"]
    last_action = None
    
    print("Starting loop... Press Ctrl+C to stop.")
    count = 0
    try:
        while True:
            # Pick a random action
            current_action = random.choice(actions)
            if current_action == last_action:
                continue
            last_action = current_action
            
            print(f"{count}: {current_action}")
            say(current_action)
            count += 1
            
            # Wait 10 seconds before the next one
            time.sleep(1)
            if count > 100:
                exit()
            
    except KeyboardInterrupt:
        print("\nScript stopped by user.")