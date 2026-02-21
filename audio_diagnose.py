#!/usr/bin/env python3
"""
Audio Troubleshooting Script for Raspberry Pi
Run this to diagnose why audio isn't working.
"""

import subprocess
import os
import sys

def run_cmd(cmd, show_output=True):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if show_output and result.stdout:
            print(result.stdout)
        if result.stderr and 'error' in result.stderr.lower():
            print(f"  Error: {result.stderr}")
        return result.returncode == 0, result.stdout
    except Exception as e:
        print(f"  Failed: {e}")
        return False, ""

def main():
    print("=" * 60)
    print("🔧 RASPBERRY PI AUDIO TROUBLESHOOTING")
    print("=" * 60)
    
    # Step 1: Check if we're on a Raspberry Pi
    print("\n[1] CHECKING DEVICE...")
    run_cmd("cat /proc/device-tree/model 2>/dev/null || echo 'Not a Raspberry Pi'")
    
    # Step 2: List audio devices
    print("\n[2] AUDIO DEVICES (aplay -l)...")
    success, output = run_cmd("aplay -l")
    if not success or "no soundcards" in output.lower():
        print("  ❌ No audio devices found!")
        print("  → Try: sudo modprobe snd_bcm2835")
    
    # Step 3: Check current audio routing
    print("\n[3] AUDIO OUTPUT ROUTING...")
    print("  Checking amixer settings...")
    run_cmd("amixer cget numid=3 2>/dev/null || echo 'Cannot check routing'")
    
    # Step 4: Check loaded modules
    print("\n[4] AUDIO KERNEL MODULES...")
    run_cmd("lsmod | grep snd")
    
    # Step 5: Check dtoverlay
    print("\n[5] DEVICE TREE OVERLAYS...")
    run_cmd("sudo dtoverlay -l 2>/dev/null || echo 'Cannot list overlays'")
    
    # Step 6: Check config.txt
    print("\n[6] BOOT CONFIG (/boot/config.txt audio settings)...")
    run_cmd("grep -i 'audio\|pwm\|dtparam' /boot/config.txt 2>/dev/null | head -10")
    
    print("\n" + "=" * 60)
    print("🔧 ATTEMPTING FIXES...")
    print("=" * 60)
    
    # Fix 1: Load audio module
    print("\n[Fix 1] Loading BCM2835 audio module...")
    run_cmd("sudo modprobe snd_bcm2835")
    
    # Fix 2: Force audio to headphone jack (analog output)
    print("\n[Fix 2] Forcing audio to ANALOG output (headphone jack)...")
    # 0=auto, 1=headphone jack, 2=HDMI
    run_cmd("sudo amixer cset numid=3 1 2>/dev/null")
    
    # Fix 3: Unmute and set volume
    print("\n[Fix 3] Unmuting and setting volume to 100%...")
    run_cmd("amixer set PCM unmute 2>/dev/null; amixer set PCM 100% 2>/dev/null")
    run_cmd("amixer set Master unmute 2>/dev/null; amixer set Master 100% 2>/dev/null")
    
    # Fix 4: Try to enable audio in config
    print("\n[Fix 4] Checking if audio is enabled in boot config...")
    success, output = run_cmd("grep 'dtparam=audio=on' /boot/config.txt", show_output=False)
    if not success:
        print("  ⚠ Audio may not be enabled in /boot/config.txt")
        print("  → Add this line to /boot/config.txt and reboot:")
        print("    dtparam=audio=on")
    else:
        print("  ✓ Audio is enabled in boot config")
    
    print("\n" + "=" * 60)
    print("🔊 TESTING AUDIO...")
    print("=" * 60)
    
    # Test 1: speaker-test
    print("\n[Test 1] Running speaker-test (5 seconds)...")
    print("  If you hear a tone, audio is working!")
    run_cmd("timeout 5 speaker-test -t sine -f 440 -l 1 2>&1 || true")
    
    # Test 2: aplay with a generated beep
    print("\n[Test 2] Playing a beep sound...")
    # Generate a simple beep using sox if available
    run_cmd("play -n synth 0.5 sine 880 vol 0.5 2>/dev/null || echo 'sox not installed'")
    
    # Test 3: espeak
    print("\n[Test 3] Text-to-speech test...")
    run_cmd("espeak 'Audio test successful' 2>/dev/null || echo 'espeak not installed'")
    
    print("\n" + "=" * 60)
    print("📋 DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    print("""
If you still hear NO sound, check these things:

1. WIRING - Make sure your connections are correct:
   ┌─────────────────────────────────────────────────────────┐
   │ Raspberry Pi           →       HW-104 Amplifier        │
   ├─────────────────────────────────────────────────────────┤
   │ 3.5mm Jack TIP         →       L (Left Input)          │
   │ 3.5mm Jack SLEEVE      →       GND                     │
   │ Pin 2 or 4 (5V)        →       VCC (if no 3.5mm)       │
   │ Pin 6 (GND)            →       GND                     │
   └─────────────────────────────────────────────────────────┘
   
   NOTE: The EASIEST way is to use the 3.5mm headphone jack
   on the Raspberry Pi, NOT the GPIO pins!

2. POWER - Is the amplifier getting power?
   → Red LED on HW-104 should be ON
   → If not, check 5V and GND connections

3. SPEAKER - Is speaker connected to amplifier?
   → OUT+ to speaker +
   → OUT- to speaker -

4. AUDIO OUTPUT - Make sure Pi outputs to headphone jack:
   → Run: sudo raspi-config
   → Go to: System Options → Audio → Headphones
   
5. REBOOT - After config changes:
   → sudo reboot

To enable PWM audio on GPIO pins (more complex):
   → Edit /boot/config.txt and add:
     dtoverlay=pwm-2chan,pin=18,func=2,pin2=13,func2=4
   → Then reboot
""")
    
    print("\n❓ Did you hear any sound during the tests? (yes/no)")


if __name__ == "__main__":
    main()
