"""
Raspberry Pi Motor Control Server
Run this on your Raspberry Pi to control motors via HTTP requests.
Works with motor_test.html web controller.

Endpoints:
  /F - Forward
  /B - Backward  
  /L - Left
  /R - Right
  /S - Stop

Usage:
  python3 pi_motor_server.py
  
Note: Uses gpiozero for Pi 5 compatibility
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
from gpiozero import OutputDevice

# ========== GPIO SETUP ==========
# Using gpiozero for Raspberry Pi 5 compatibility
IN1 = OutputDevice(17)  # GPIO17 → Pin 11
IN2 = OutputDevice(27)  # GPIO27 → Pin 13
IN3 = OutputDevice(22)  # GPIO22 → Pin 15
IN4 = OutputDevice(23)  # GPIO23 → Pin 16

def stopCar():
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

def left():
    IN1.on()
    IN2.off()
    IN3.off()
    IN4.on()

def right():
    IN1.off()
    IN2.on()
    IN3.on()
    IN4.off()

stopCar()

# ========== HTTP SERVER ==========
class CarHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/F":
            forward()
            print("→ FORWARD")
        elif self.path == "/B":
            backward()
            print("→ BACKWARD")
        elif self.path == "/L":
            left()
            print("→ LEFT")
        elif self.path == "/R":
            right()
            print("→ RIGHT")
        elif self.path == "/S":
            stopCar()
            print("→ STOP")

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b"OK")
    
    # Suppress logging for cleaner output
    def log_message(self, format, *args):
        pass

# ========== MAIN ==========
if __name__ == "__main__":
    try:
        server = HTTPServer(("", 8080), CarHandler)
        print("=" * 40)
        print(" Raspberry Pi Motor Server Started")
        print(" Listening on port 8080")
        print(" Endpoints: /F /B /L /R /S")
        print("=" * 40)
        server.serve_forever()

    except KeyboardInterrupt:
        print("\nStopping server...")

    finally:
        stopCar()
        IN1.close()
        IN2.close()
        IN3.close()
        IN4.close()
        print("GPIO cleaned up. Goodbye!")
