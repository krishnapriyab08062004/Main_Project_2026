import os
import pyttsx3
import time

# Create a directory for synthetic samples
OUTPUT_DIR = "data/synthetic_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

engine = pyttsx3.init()
voices = engine.getProperty('voices')

# Find at least two voices (Male and Female)
# Typically: voices[0] is David (Male), voices[1] is Zira (Female)
male_voice = voices[0].id if len(voices) > 0 else None
female_voice = voices[1].id if len(voices) > 1 else male_voice

# Emotional scripts (exactly 20)
scripts = [
    ("angry", "I am absolutely furious about this delay! This is unacceptable!"),
    ("angry", "Stop doing that right now! I told you a thousand times!"),
    ("sad", "I'm feeling very lonely today. It's been a hard week."),
    ("sad", "I really miss the way things used to be. Everything feels empty."),
    ("happy", "I have some amazing news! We finally won the competition!"),
    ("happy", "This is the best day ever! I'm so excited for the party!"),
    ("fearful", "Wait, did you hear that? Someone is outside the door!"),
    ("fearful", "I'm really scared of what might happen next. Please help me."),
    ("neutral", "The weather forecast for tomorrow is partly cloudy."),
    ("neutral", "Please turn left at the next intersection and drive straight."),
    ("disgust", "Ugh, this smells absolutely disgusting. Take it away!"),
    ("disgust", "I can't believe they served this food. It's revolting."),
    ("surprised", "Oh my god! I never expected to see you here!"),
    ("surprised", "Wow! This is a completely unexpected turn of events!"),
    ("calm", "Just take a deep breath and relax. Everything is fine."),
    ("calm", "The ocean waves are very peaceful today. Listen to them."),
    ("angry", "If you don't fix this immediately, I will be reporting you!"),
    ("sad", "It's just too much to handle right now. I need some time."),
    ("happy", "I'm so proud of what we've accomplished together!"),
    ("fearful", "The shadows are moving in the corner of the room. I don't like it.")
]

def generate_samples():
    print(f"Generating 20 synthetic samples (Male & Female) in {OUTPUT_DIR}...")
    
    for i, (emotion, text) in enumerate(scripts):
        # Choose voice: alternate between male and female
        current_voice = male_voice if i % 2 == 0 else female_voice
        voice_label = "Male" if current_voice == male_voice else "Female"
        
        filename = f"{i+1:02d}_{emotion}_{voice_label}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Configure engine
        engine.setProperty('voice', current_voice)
        
        # Vary speech rate for variety
        rate = 150 + (i % 3) * 25 # 150, 175, 200
        engine.setProperty('rate', rate)
        
        print(f"[{i+1}/20] Generating {filename} ({voice_label})...")
        
        # Save to file
        engine.save_to_file(text, filepath)
        engine.runAndWait()
        
        # On Windows, a small sleep helps release file handles
        time.sleep(0.5)
        
    print(f"\n✅ Successfully generated {len(scripts)} samples.")
    print(f"Location: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_samples()
