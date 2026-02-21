from gpiozero import OutputDevice
import time

# Replicating pi_motor_server.py EXACTLY with globals
print("Initializing GPIO (Global Scope)...")
IN1 = OutputDevice(17)
IN2 = OutputDevice(27)
IN3 = OutputDevice(22)
IN4 = OutputDevice(23)

def stop():
    IN1.off()
    IN2.off()
    IN3.off()
    IN4.off()

def forward():
    IN1.on()
    IN2.off()
    IN3.on()
    IN4.off()

def backward():
    IN1.off()
    IN2.on()
    IN3.off()
    IN4.on()

if __name__ == "__main__":
    try:
        print("Forward...")
        forward()
        time.sleep(2)
        
        print("Stop...")
        stop()
        time.sleep(1)
        
        print("Backward...")
        backward()
        time.sleep(2)
        
        print("Stop...")
        stop()
        print("Done.")
    finally:
        IN1.close()
        IN2.close()
        IN3.close()
        IN4.close()
