#!/usr/bin/env python3
"""
Raspberry Pi 5 PWM Audio Test
=============================
For HW-104 (PAM8403) amplifier connected via GPIO PWM.
RPi 5 has NO 3.5mm jack - must use GPIO PWM or USB audio.

WIRING FOR RPI 5:
    Pin 2 (5V)      → VCC on amplifier
    Pin 6 (GND)     → GND on amplifier  
    Pin 12 (GPIO18) → L (Left) on amplifier
    Pin 33 (GPIO13) → R (Right) on amplifier (optional)
"""

import subprocess
import os
import sys
import time
import math
import struct

SAMPLE_RATE = 48000


def check_rpi5():
    """Check if running on Raspberry Pi 5"""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            print(f"Device: {model.strip()}")
            return 'Pi 5' in model or 'Raspberry Pi 5' in model
    except:
        return False


def setup_pwm_audio_rpi5():
    """
    Setup PWM audio on Raspberry Pi 5
    RPi 5 uses different overlay than older Pis
    """
    print("\n" + "=" * 50)
    print("🔧 RASPBERRY PI 5 PWM AUDIO SETUP")
    print("=" * 50)
    
    # Step 1: Check /boot/firmware/config.txt (RPi 5 uses /boot/firmware/)
    config_paths = ['/boot/firmware/config.txt', '/boot/config.txt']
    config_path = None
    
    for path in config_paths:
        if os.path.exists(path):
            config_path = path
            break
    
    if config_path:
        print(f"\n[1] Checking {config_path}...")
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Check for audio settings
            if 'dtparam=audio=on' in content:
                print("  ✓ Audio enabled")
            else:
                print("  ⚠ Audio might not be enabled")
                print(f"    Add 'dtparam=audio=on' to {config_path}")
            
            if 'dtoverlay=pwm' in content:
                print("  ✓ PWM overlay configured")
            else:
                print("  ⚠ PWM overlay not in config")
                print("    Add 'dtoverlay=pwm-2chan,pin=18,func=2,pin2=13,func2=4'")
        except:
            print("  Could not read config file")
    
    # Step 2: Try to load PWM overlay at runtime
    print("\n[2] Loading PWM overlay...")
    result = subprocess.run(
        ['sudo', 'dtoverlay', 'pwm-2chan', 'pin=18', 'func=2', 'pin2=13', 'func2=4'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("  ✓ PWM overlay loaded!")
    else:
        print(f"  ⚠ Could not load overlay: {result.stderr}")
        print("  This is normal - may need reboot after adding to config.txt")
    
    # Step 3: Check audio devices
    print("\n[3] Checking audio devices...")
    result = subprocess.run(['aplay', '-l'], capture_output=True, text=True)
    print(result.stdout if result.stdout else "  No devices found")
    
    # Step 4: Try USB audio as fallback detection
    print("\n[4] Looking for USB audio devices...")
    result = subprocess.run(['aplay', '-L'], capture_output=True, text=True)
    usb_found = 'usb' in result.stdout.lower() if result.stdout else False
    if usb_found:
        print("  ✓ USB audio device detected!")
    else:
        print("  No USB audio device")
    
    return True


def generate_wav(filename, frequency=440, duration=1.0, volume=0.8):
    """Generate a WAV file with a sine wave"""
    import wave
    
    n_samples = int(SAMPLE_RATE * duration)
    amplitude = int(32767 * volume)
    
    with wave.open(filename, 'w') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        
        for i in range(n_samples):
            t = i / SAMPLE_RATE
            sample = int(amplitude * math.sin(2 * math.pi * frequency * t))
            wav.writeframesraw(struct.pack('<hh', sample, sample))
    
    return True


def play_with_aplay(filename, device=None):
    """Play audio file with aplay"""
    cmd = ['aplay']
    if device:
        cmd.extend(['-D', device])
    cmd.append(filename)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False


def find_working_audio_device():
    """Find a working audio output device"""
    print("\n🔍 Searching for audio output devices...")
    
    # Get list of devices
    result = subprocess.run(['aplay', '-L'], capture_output=True, text=True)
    if not result.stdout:
        return None
    
    # Parse device names
    devices = []
    for line in result.stdout.split('\n'):
        if not line.startswith(' ') and line.strip():
            devices.append(line.strip())
    
    # Priority order for devices
    priority = ['hw:0,0', 'plughw:0,0', 'default', 'sysdefault']
    
    # Try devices
    generate_wav('/tmp/test.wav', 440, 0.1)
    
    for device in priority + devices:
        try:
            result = subprocess.run(
                ['aplay', '-D', device, '/tmp/test.wav'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                print(f"  ✓ Found working device: {device}")
                return device
        except:
            pass
    
    return None


def test_audio():
    """Test audio output"""
    print("\n" + "=" * 50)
    print("🔊 AUDIO TEST")
    print("=" * 50)
    
    # Generate test tone
    print("\n[1] Generating 440Hz test tone...")
    generate_wav('/tmp/audio_test.wav', 440, 2.0)
    print("  ✓ Generated /tmp/audio_test.wav")
    
    # Try different playback methods
    print("\n[2] Attempting playback...")
    
    # Method 1: Default aplay
    print("  Trying aplay (default)...")
    if play_with_aplay('/tmp/audio_test.wav'):
        print("  ✓ Playback successful!")
        return True
    
    # Method 2: Try hw:0,0
    print("  Trying aplay -D hw:0,0...")
    if play_with_aplay('/tmp/audio_test.wav', 'hw:0,0'):
        print("  ✓ Playback successful!")
        return True
    
    # Method 3: Try plughw
    print("  Trying aplay -D plughw:0,0...")
    if play_with_aplay('/tmp/audio_test.wav', 'plughw:0,0'):
        print("  ✓ Playback successful!")
        return True
    
    # Method 4: speaker-test
    print("  Trying speaker-test...")
    try:
        subprocess.run(['speaker-test', '-t', 'sine', '-f', '440', '-l', '1'],
                      timeout=5, capture_output=True)
        return True
    except:
        pass
    
    print("  ❌ No audio output detected")
    return False


def show_rpi5_instructions():
    """Show instructions for RPi 5 audio setup"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           RASPBERRY PI 5 AUDIO SETUP INSTRUCTIONS                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  RPi 5 has NO 3.5mm audio jack! You have 3 options:             ║
║                                                                  ║
║  ╭──────────────────────────────────────────────────────────╮    ║
║  │  OPTION 1: USB AUDIO ADAPTER (EASIEST)                   │    ║
║  │  ─────────────────────────────────────────               │    ║
║  │  1. Plug in a USB sound card/adapter                     │    ║
║  │  2. Connect 3.5mm cable to adapter                       │    ║
║  │  3. Wire to amplifier: TIP→L, SLEEVE→GND                 │    ║
║  │  4. Run: speaker-test -t sine -f 440                     │    ║
║  ╰──────────────────────────────────────────────────────────╯    ║
║                                                                  ║
║  ╭──────────────────────────────────────────────────────────╮    ║
║  │  OPTION 2: PWM GPIO AUDIO (REQUIRES REBOOT)              │    ║
║  │  ─────────────────────────────────────────               │    ║
║  │  1. Edit config: sudo nano /boot/firmware/config.txt     │    ║
║  │                                                          │    ║
║  │  2. Add these lines at the end:                          │    ║
║  │     dtparam=audio=on                                     │    ║
║  │     dtoverlay=audremap,pins_18_19                        │    ║
║  │                                                          │    ║
║  │  3. Save and reboot: sudo reboot                         │    ║
║  │                                                          │    ║
║  │  4. Wire: GPIO18→L, GND→GND, 5V→VCC                      │    ║
║  ╰──────────────────────────────────────────────────────────╯    ║
║                                                                  ║
║  ╭──────────────────────────────────────────────────────────╮    ║
║  │  OPTION 3: I2S DAC HAT (BEST QUALITY)                    │    ║
║  │  ─────────────────────────────────────────               │    ║
║  │  Buy an I2S DAC board (e.g., PCM5102, HiFiBerry)         │    ║
║  ╰──────────────────────────────────────────────────────────╯    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

YOUR WIRING (with PWM or USB audio):
────────────────────────────────────
    Audio Source          HW-104 Amplifier
    ─────────────         ─────────────────
    Audio Signal    ────► L (Left Input)
    Ground          ────► GND
    5V (from Pi)    ────► VCC
    
    HW-104 OUT+     ────► Speaker +
    HW-104 OUT-     ────► Speaker -
""")


def main():
    print("\n🔊 Raspberry Pi 5 Audio Test\n")
    
    # Check if RPi 5
    is_rpi5 = check_rpi5()
    
    # Show setup instructions
    show_rpi5_instructions()
    
    # Setup PWM
    setup_pwm_audio_rpi5()
    
    # Test audio
    if not test_audio():
        print("\n" + "=" * 50)
        print("❌ AUDIO NOT WORKING")
        print("=" * 50)
        print("""
The easiest solution for Raspberry Pi 5:
  
  → Buy a cheap USB audio adapter (~$5)
  → Plug it in, it works immediately
  → Connect to your HW-104 amplifier

If you want to use GPIO PWM:
  1. sudo nano /boot/firmware/config.txt
  2. Add: dtoverlay=audremap,pins_18_19
  3. sudo reboot
  4. Then run this script again
""")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
