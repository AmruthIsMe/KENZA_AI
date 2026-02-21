import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("motor_test")

class GPIOMotorController:
    def __init__(self):
        self.initialized = False
        self.IN1 = None
        self.IN2 = None
        self.IN3 = None
        self.IN4 = None
        
    def connect(self) -> bool:
        """Initialize GPIO pins for motor control"""
        try:
            from gpiozero import OutputDevice
            
            # Use same pins as pi_motor_server.py
            self.IN1 = OutputDevice(17)
            self.IN2 = OutputDevice(27)
            self.IN3 = OutputDevice(22)
            self.IN4 = OutputDevice(23)
            
            self.stop()
            self.initialized = True
            log.info("🔌 GPIO Initialized (Pins 17, 27, 22, 23)")
            return True
            
        except Exception as e:
            log.error(f"GPIO initialization failed: {e}")
            return False
    
    def send_motor_command(self, direction: str, speed: int):
        log.info(f"Command: {direction} @ {speed}%")
        
        if not self.initialized:
            log.warning("Not initialized!")
            return
        
        if speed == 0 or direction == 'S':
            self.stop()
            return
            
        if direction == 'F':
            self.IN1.on()
            self.IN2.off()
            self.IN3.on()
            self.IN4.off()
        elif direction == 'B':
            self.IN1.off()
            self.IN2.on()
            self.IN3.off()
            self.IN4.on()
        elif direction == 'L':
            self.IN1.off()
            self.IN2.on()
            self.IN3.on()
            self.IN4.off()
        elif direction == 'R':
            self.IN1.on()
            self.IN2.off()
            self.IN3.off()
            self.IN4.on()

    def stop(self):
        if self.IN1: self.IN1.off()
        if self.IN2: self.IN2.off()
        if self.IN3: self.IN3.off()
        if self.IN4: self.IN4.off()

if __name__ == "__main__":
    print("Testing Motors...")
    motors = GPIOMotorController()
    if motors.connect():
        print("Forward for 2 seconds...")
        motors.send_motor_command('F', 100)
        time.sleep(2)
        
        print("Stop...")
        motors.send_motor_command('S', 0)
        time.sleep(1)
        
        print("Backward for 2 seconds...")
        motors.send_motor_command('B', 100)
        time.sleep(2)
        
        print("Stop...")
        motors.stop()
        print("Done.")
    else:
        print("Failed to init motors")
