import asyncio
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-demo")

try:
    import edge_tts
except ImportError:
    print("Error: 'edge-tts' not found. Please install it with: pip install edge-tts")
    exit(1)

# --- Config: Demo Voices ---
DEMO_VOICES = {
    "Swara (Hindi Female)": "hi-IN-SwaraNeural",
    "Madhur (Hindi Male)": "hi-IN-MadhurNeural",
    "Neerja (Indian English Female)": "en-IN-NeerjaNeural",
    "Prabhat (Indian English Male)": "en-IN-PrabhatNeural",
}

DEMO_TEXTS = {
    "hi-IN-SwaraNeural": "नमस्ते! मैं स्वरा हूँ। स्वागत है सुत्रा ए आई इंजन के वॉइस डेमो में। यह आवाज़ पूरी तरह से ए आई द्वारा जनरेट की गई है।",
    "hi-IN-MadhurNeural": "नमस्ते, मैं मधुर हूँ। मैं सुत्रा ए आई का डिजिटल नरेटर हूँ। हम आपकी मार्केटिंग और एजुकेशन के लिए बेहतरीन वॉइस ओवर प्रदान करते हैं।",
    "en-IN-NeerjaNeural": "Hello! I am Neerja, your AI narrator. Today we are demonstrating the high-fidelity speech synthesis capabilities of the Sutra AI Engine.",
    "en-IN-PrabhatNeural": "Hello, this is Prabhat. Our O(1) polymorphic architecture ensures that you always get the best AI provider for every single request.",
}

# --- Core Generator ---
async def generate_sample(name: str, voice_id: str, text: str, output_dir: Path):
    """Generates a single voice sample."""
    logger.info(f"🎤 Generating: {name} ({voice_id})...")
    
    # Sanitize filename
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    output_path = output_dir / f"demo_{safe_name}.mp3"
    
    try:
        communicate = edge_tts.Communicate(text, voice_id)
        await communicate.save(str(output_path))
        logger.info(f"✅ Saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"❌ Failed to generate {name}: {e}")
        return None

async def main():
    # 1. Setup output directory
    output_dir = Path("voice_demos")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*50)
    print("  SutraAI Engine — Voice Demonstration")
    print("="*50 + "\n")
    
    # 2. Iterate and generate
    tasks = []
    for description, voice_id in DEMO_VOICES.items():
        text = DEMO_TEXTS.get(voice_id, "Sample test for AI voice.")
        tasks.append(generate_sample(description, voice_id, text, output_dir))
    
    results = await asyncio.gather(*tasks)
    
    # 3. Final Report
    print("\n" + "="*50)
    print("  Generation Complete!")
    print("="*50)
    for res in results:
        if res:
            print(f" - {res.name}")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
