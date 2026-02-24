/**
 * KENZA AI - Robot Display & Interaction System
 */

class EyeController {
    constructor() {
        this.container = document.getElementById('eye-container');
        this.leftEye = document.getElementById('left-eye');
        this.rightEye = document.getElementById('right-eye');
        this.currentExpression = 'neutral';
        this.isBlinking = false;

        this.init();
    }

    init() {
        // Start idle behavior
        this.startBlinking();
        this.startSaccades();
    }

    setExpression(expression) {
        if (this.currentExpression === expression) return;

        // Remove previous expression classes
        this.container.classList.remove(`expression-${this.currentExpression}`);

        // Add new expression class
        this.currentExpression = expression;
        this.container.classList.add(`expression-${expression}`);

        console.log(`[EYES] Expression set to: ${expression}`);
    }

    startBlinking() {
        const blink = () => {
            if (this.currentExpression === 'sleeping') return;

            this.container.classList.add('blinking');
            setTimeout(() => {
                this.container.classList.remove('blinking');

                // Schedule next blink (random interval between 2-6 seconds)
                const nextBlink = 2000 + Math.random() * 4000;
                setTimeout(blink, nextBlink);
            }, 200);
        };

        setTimeout(blink, 3000);
    }

    startSaccades() {
        const moveEyes = () => {
            if (this.currentExpression === 'sleeping' || this.currentExpression === 'focused') return;

            const x = (Math.random() - 0.5) * 20; // -10 to 10px
            const y = (Math.random() - 0.5) * 10; // -5 to 5px

            const leftIris = this.leftEye.querySelector('.iris');
            const rightIris = this.rightEye.querySelector('.iris');

            const transform = `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`;
            leftIris.style.transform = transform;
            rightIris.style.transform = transform;

            // Schedule next movement
            const nextMove = 1000 + Math.random() * 3000;
            setTimeout(moveEyes, nextMove);
        };

        setTimeout(moveEyes, 2000);
    }

    lookAt(targetX, targetY) {
        // Map target relative to screen to eye movement
        const irisX = (targetX / window.innerWidth - 0.5) * 30;
        const irisY = (targetY / window.innerHeight - 0.5) * 20;

        const leftIris = this.leftEye.querySelector('.iris');
        const rightIris = this.rightEye.querySelector('.iris');

        const transform = `translate(calc(-50% + ${irisX}px), calc(-50% + ${irisY}px))`;
        leftIris.style.transform = transform;
        rightIris.style.transform = transform;
    }
}

class InteractionManager {
    constructor(eyeController) {
        this.eyes = eyeController;
        this.overlay = document.getElementById('ui-overlay');
        this.radialMenu = document.getElementById('radial-menu');
        this.isOverlayOpen = false;

        this.init();
    }

    init() {
        // Tap to awake
        document.body.addEventListener('click', (e) => {
            if (!this.isOverlayOpen && !this.radialMenu.classList.contains('visible')) {
                this.openOverlay();
            }
        });

        // Swipe handling
        let touchStartY = 0;
        document.addEventListener('touchstart', e => {
            touchStartY = e.touches[0].clientY;
        });

        document.addEventListener('touchend', e => {
            const touchEndY = e.changedTouches[0].clientY;
            if (touchEndY - touchStartY > 100) { // Swipe down
                this.closeOverlay();
            }
        });

        // Long press for radial menu
        let pressTimer;
        this.eyes.container.addEventListener('mousedown', e => {
            pressTimer = window.setTimeout(() => this.toggleRadialMenu(true, e.clientX, e.clientY), 1000);
        });
        this.eyes.container.addEventListener('mouseup', () => clearTimeout(pressTimer));
        this.eyes.container.addEventListener('touchstart', e => {
            pressTimer = window.setTimeout(() => this.toggleRadialMenu(true, e.touches[0].clientX, e.touches[0].clientY), 1000);
        });
        this.eyes.container.addEventListener('touchend', () => clearTimeout(pressTimer));
    }

    openOverlay() {
        this.isOverlayOpen = true;
        this.overlay.classList.remove('hidden');
        this.eyes.container.style.transform = 'scale(0.8) translateY(-10%)';

        // Auto-hide after inactivity
        this.resetInactivityTimer();
    }

    closeOverlay() {
        this.isOverlayOpen = false;
        this.overlay.classList.add('hidden');
        this.eyes.container.style.transform = 'scale(1) translateY(0)';
    }

    toggleRadialMenu(show, x, y) {
        if (show) {
            this.radialMenu.classList.remove('hidden');
            // Center eyes apart?
            this.eyes.container.style.gap = '150px';
        } else {
            this.radialMenu.classList.add('hidden');
            this.eyes.container.style.gap = '80px';
        }
    }

    resetInactivityTimer() {
        if (this.inactivityTimeout) clearTimeout(this.inactivityTimeout);
        this.inactivityTimeout = setTimeout(() => {
            if (this.isOverlayOpen) this.closeOverlay();
        }, 10000); // 10 seconds
    }
}

class TelepresenceManager {
    constructor(app) {
        this.app = app;
        this.view = document.getElementById('telepresence-mode');
        this.video = document.getElementById('remote-video');
        this.isActive = false;
    }

    activate() {
        this.isActive = true;
        this.view.classList.remove('hidden');
        document.body.classList.add('telepresence-active');
        document.body.classList.remove('robot-mode');

        // Hide eyes smoothly
        this.app.eyes.container.style.opacity = '0';

        console.log('[TELEPRESENCE] Activated');
    }

    deactivate() {
        this.isActive = false;
        this.view.classList.add('hidden');
        document.body.classList.remove('telepresence-active');
        document.body.classList.add('robot-mode');

        // Show eyes smoothly
        this.app.eyes.container.style.opacity = '1';

        console.log('[TELEPRESENCE] Deactivated');
    }

    handleSignal(data) {
        // Handle WebRTC signaling here
        console.log('[TELEPRESENCE] Signaling data received', data);
    }
}

class RobotApp {
    constructor() {
        this.eyes = new EyeController();
        this.interaction = new InteractionManager(this.eyes);
        this.telepresence = new TelepresenceManager(this);
        this.ws = null;

        this.connect();
    }

    connect() {
        const port = 8765;
        const host = window.location.hostname || 'localhost';
        this.ws = new WebSocket(`ws://${host}:${port}`);

        this.ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            this.handleMessage(msg);
        };

        this.ws.onclose = () => {
            console.log('[WS] Disconnected. Retrying...');
            setTimeout(() => this.connect(), 3000);
        };
    }

    handleMessage(msg) {
        switch (msg.type) {
            case 'telemetry':
                this.updateStatusBar(msg.data);
                break;
            case 'emotion':
                this.eyes.setExpression(msg.data.expression);
                break;
            case 'mode_change':
                document.getElementById('current-mode').textContent = msg.data.mode;
                if (msg.data.mode === 'telepresence') {
                    this.telepresence.activate();
                } else if (this.telepresence.isActive) {
                    this.telepresence.deactivate();
                }
                break;
            case 'telepresence_signal':
                this.telepresence.handleSignal(msg.data);
                break;
        }
    }

    updateStatusBar(data) {
        if (data.battery !== undefined) {
            document.getElementById('battery-level').textContent = `${data.battery}%`;
        }
        if (data.wifi_rssi !== undefined) {
            const status = data.wifi_rssi > -60 ? 'Strong' : 'Weak';
            document.getElementById('wifi-status').textContent = status;
        }
    }
}

// Start the app
window.addEventListener('DOMContentLoaded', () => {
    window.robot = new RobotApp();
});
