#!/usr/bin/env python3
"""
KENZA Launcher Service
A lightweight, zero-dependency HTTP server that runs constantly on the Raspberry Pi.
It listens on port 8764 for commands from the web app to start or stop the heavy
kenza_server.py process. This saves CPU and RAM when the robot is not actively in use.
"""

import http.server
import socketserver
import subprocess
import json
import os
import signal
import sys
import argparse

# Global state
server_process = None
PORT = 8764
SERVER_FILE = "kenza_server.py"
VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python3")

class LauncherHandler(http.server.BaseHTTPRequestHandler):
    
    def _set_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()
        
    def _send_json(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_GET(self):
        global server_process
        
        # Check if process died
        if server_process is not None and server_process.poll() is not None:
            server_process = None
            
        if self.path == '/status':
            is_running = server_process is not None
            self._send_json(200, {
                "status": "online",
                "kenza_server_running": is_running,
                "pid": server_process.pid if is_running else None
            })
            
        elif self.path == '/start':
            if server_process is not None:
                self._send_json(200, {"status": "already_running", "pid": server_process.pid})
                return
                
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                script_path = os.path.join(base_dir, SERVER_FILE)
                
                # Determine python executable (prefer venv if it exists)
                python_exe = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable
                
                print(f"[LAUNCHER] Starting {script_path} using {python_exe}...")
                
                # Use preexec_fn=os.setsid to start a new process group, making it easier to kill children
                preexec = os.setsid if hasattr(os, 'setsid') else None
                
                server_process = subprocess.Popen(
                    [python_exe, script_path],
                    cwd=base_dir,
                    preexec_fn=preexec,
                    stdout=subprocess.DEVNULL, # Suppress output to prevent blocking
                    stderr=subprocess.DEVNULL
                )
                
                self._send_json(200, {"status": "started", "pid": server_process.pid})
                print(f"[LAUNCHER] kenza_server.py started (PID: {server_process.pid})")
                
            except Exception as e:
                self._send_json(500, {"error": str(e)})
                print(f"[LAUNCHER] Error starting server: {e}")
                
        elif self.path == '/stop':
            if server_process is None:
                self._send_json(200, {"status": "not_running"})
                return
                
            try:
                print(f"[LAUNCHER] Stopping kenza_server.py (PID: {server_process.pid})...")
                
                # Kill entire process group if posix, else just the process
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                else:
                    server_process.terminate()
                
                try:
                    server_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if hasattr(os, 'killpg'):
                        os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
                    else:
                        server_process.kill()
                
                server_process = None
                self._send_json(200, {"status": "stopped"})
                print("[LAUNCHER] kenza_server.py stopped successfully")
                
            except Exception as e:
                server_process = None # Force reset
                self._send_json(500, {"error": str(e)})
                print(f"[LAUNCHER] Error stopping server: {e}")
                
        else:
            self._send_json(404, {"error": "Not Found"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kenza Launcher HTTP Service")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port to listen on (default: {PORT})")
    args = parser.parse_args()
    
    print(f"===================================================")
    print(f" KENZA LAUNCHER SERVICE")
    print(f" Listening on 0.0.0.0:{args.port}")
    print(f" Waiting for App to command /start or /stop...")
    print(f"===================================================")
    
    # Use ThreadingTCPServer for non-blocking requests
    class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
    
    with ThreadedHTTPServer(("0.0.0.0", args.port), LauncherHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[LAUNCHER] Shutting down...")
            if server_process:
                print(f"[LAUNCHER] Cleaning up kenza_server.py (PID {server_process.pid})")
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
                else:
                    server_process.kill()
            sys.exit(0)
