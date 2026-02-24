/* ═══════════════════════════════════════════════════════════
   KENZA AI — Complete Display System JS
   ═══════════════════════════════════════════════════════════ */

const $ = id => document.getElementById(id);

/* ── Eye Config (syncs with eyes_display) ─────────────────── */
const eyeCfg = { h: 190, s: 100, l: 50, shape: 'round', brightness: 100 };

const colorPresets = {
    cyan: { h: 190, s: 100, l: 50 }, orange: { h: 30, s: 100, l: 50 },
    teal: { h: 160, s: 100, l: 50 }, pink: { h: 330, s: 100, l: 65 },
    purple: { h: 270, s: 100, l: 60 }, green: { h: 120, s: 100, l: 50 },
    red: { h: 0, s: 100, l: 50 }, gold: { h: 45, s: 100, l: 50 },
    white: { h: 0, s: 0, l: 95 }
};

function applyEyeStyle() {
    const { h, s, l, brightness: b } = eyeCfg;
    const bl = b / 100; const aL = l * bl;
    document.querySelectorAll('.eye').forEach(eye => {
        eye.style.background = `radial-gradient(circle at 30% 30%,
      hsl(${h},${s}%,${Math.min(aL + 28, 100)}%) 0%,
      hsl(${h},${s}%,${aL}%) 50%,
      hsl(${h},${s}%,${Math.max(aL - 22, 5)}%) 100%)`;
        eye.style.boxShadow = `0 0 ${40 * bl}px hsla(${h},${s}%,${aL}%,0.4),
      inset 0 -12px 28px rgba(0,0,0,0.38)`;
    });
    // sync the tp eye icon colour
    const tpEye = $('tp-eye-icon');
    if (tpEye) tpEye.style.background = `radial-gradient(circle at 30% 30%,
    hsl(${h},${s}%,${Math.min(aL + 28, 100)}%) 0%,
    hsl(${h},${s}%,${aL}%) 50%,
    hsl(${h},${s}%,${Math.max(aL - 22, 5)}%) 100%)`;
    applyShape();
}

function applyShape() {
    const eyes2 = document.querySelectorAll('.eye');
    const pupils = document.querySelectorAll('.eye .pupil');
    eyes2.forEach(e => { e.style.width = '140px'; e.style.height = '140px'; e.style.borderRadius = '50%'; e.style.transform = ''; });
    pupils.forEach(p => { p.style.width = '50px'; p.style.height = '50px'; p.style.borderRadius = '50%'; });
    switch (eyeCfg.shape) {
        case 'oval': eyes2.forEach(e => { e.style.width = '115px'; e.style.height = '148px'; }); break;
        case 'anime': eyes2.forEach(e => { e.style.width = '150px'; e.style.height = '158px'; e.style.borderRadius = '50% 50% 38% 38%'; }); pupils.forEach(p => { p.style.width = '60px'; p.style.height = '60px'; }); break;
        case 'robot': eyes2.forEach(e => { e.style.width = '132px'; e.style.height = '108px'; e.style.borderRadius = '16px'; }); pupils.forEach(p => { p.style.borderRadius = '8px'; p.style.width = '46px'; p.style.height = '46px'; }); break;
        case 'line': eyes2.forEach(e => { e.style.width = '162px'; e.style.height = '28px'; e.style.borderRadius = '16px'; }); pupils.forEach(p => { p.style.width = '22px'; p.style.height = '20px'; }); break;
        case 'rectangle': eyes2.forEach(e => { e.style.width = '158px'; e.style.height = '88px'; e.style.borderRadius = '14px'; }); break;
        case 'diamond': eyes2.forEach(e => { e.style.width = '110px'; e.style.height = '110px'; e.style.borderRadius = '12px'; e.style.transform = 'rotate(45deg)'; }); pupils.forEach(p => { p.style.transform = 'translate(-50%,-50%) rotate(-45deg)'; }); break;
    }
}

/* ── App State ────────────────────────────────────────────── */
const st = {
    expr: 'neutral', panelOpen: false, radialOpen: false,
    sleeping: false, tpActive: false,
    battery: 95, mode: 'Auto', ai: 'Online',
    micOn: true, camOn: false, tpConn: false,
    panelTab: 'actions', speaking: false,
};

let blinkTmr = null, saccTmr = null, sleepTmr = null, panTmr = null;
let notifTmr = null, stxtTmr = null, tpTimerIv = null, zzzIv = null;
let lpTmr = null; // long press
let ws = null;
let tpSecs = 0;

/* ── Boot Sequence ────────────────────────────────────────── */
function boot() {
    const bootEl = $('boot');
    // Step 1: glow
    setTimeout(() => { bootEl.classList.add('glow-on'); }, 200);
    // Step 2: fade in eyes + bar
    setTimeout(() => {
        bootEl.classList.add('fade-out');
        $('scene').classList.add('show');
        $('sb').classList.add('show');
    }, 1400);
    // Step 3: scan pupil expand
    setTimeout(() => {
        document.querySelectorAll('.eye .pupil').forEach(p => { p.style.width = '70px'; p.style.height = '70px'; });
        setTimeout(() => { document.querySelectorAll('.eye .pupil').forEach(p => { p.style.width = ''; p.style.height = ''; }); }, 400);
    }, 2200);
    // Step 4: blink
    setTimeout(() => { doBlink(); }, 2700);
    // Step 5: idle
    setTimeout(() => {
        bootEl.style.display = 'none';
        startBlink();
        scheduleSacc();
        resetSleepTimer();
        setStatusText('System ready', 2500);
        connectWS();
    }, 3200);
}

/* ── Expressions ──────────────────────────────────────────── */
const EXPRS = ['neutral', 'happy', 'curious', 'listening', 'thinking', 'alert', 'sleepy', 'sad', 'excited', 'error', 'searching', 'speaking', 'focused'];

function setExpr(ex, durationMs = 0) {
    if (!EXPRS.includes(ex)) ex = 'neutral';
    EXPRS.forEach(e => $('eyes').classList.remove(e));
    $('eyes').classList.add(ex);
    st.expr = ex;
    if (['sleepy', 'tired'].includes(ex)) activateSleep();
    else deactivateSleep();
    if (ex === 'alert') notify('⚠️ Alert!', 'warn');
    if (durationMs > 0) setTimeout(() => setExpr('neutral'), durationMs);
}

/* ── Blink ────────────────────────────────────────────────── */
function doBlink() {
    document.querySelectorAll('.eye').forEach(e => e.classList.add('blink'));
    setTimeout(() => document.querySelectorAll('.eye').forEach(e => e.classList.remove('blink')), 185);
}
function startBlink() {
    const go = () => {
        if (st.sleeping || st.tpActive) return;
        doBlink();
        blinkTmr = setTimeout(go, 4000 + Math.random() * 2500);
    };
    blinkTmr = setTimeout(go, 2500);
}

/* ── Saccades ─────────────────────────────────────────────── */
function movePupils(dx, dy, dur = 250) {
    document.querySelectorAll('.eye .pupil').forEach(p => {
        if (['searching', 'excited', 'error'].includes(st.expr)) return;
        p.style.transition = `transform ${dur}ms cubic-bezier(0.25,0.46,0.45,0.94)`;
        p.style.transform = `translate(calc(-50% + ${dx}px),calc(-50% + ${dy}px))`;
    });
}
function scheduleSacc() {
    saccTmr = setTimeout(() => {
        if (!st.sleeping && !st.tpActive && !['searching', 'excited', 'error', 'love'].includes(st.expr)) {
            const dx = (Math.random() - .5) * 22, dy = (Math.random() - .5) * 12;
            movePupils(dx, dy, 220);
            setTimeout(() => { movePupils(0, 0, 380); scheduleSacc(); }, 900 + Math.random() * 1300);
        } else scheduleSacc();
    }, 1200 + Math.random() * 2800);
}

/* ── Touch / Pointer ──────────────────────────────────────── */
document.body.addEventListener('pointerdown', e => {
    if (e.target.closest('#panel') || e.target.closest('#tp') || e.target.closest('#radial')) return;
    wakeUp();

    // ripple
    const r = document.createElement('div'); r.className = 'ripple';
    r.style.cssText = `width:80px;height:80px;left:${e.clientX - 40}px;top:${e.clientY - 40}px;`;
    document.body.appendChild(r); setTimeout(() => r.remove(), 700);

    // long press detection
    lpTmr = setTimeout(() => openRadial(), 620);

    if (st.radialOpen) { closeRadial(); return; }
    if (st.panelOpen) {
        const rect = $('panel').getBoundingClientRect();
        if (e.clientY < rect.top) closePanel();
    } else {
        openPanel();
        setExpr('curious', 800);
    }
    resetSleepTimer();
}, { passive: true });

document.body.addEventListener('pointerup', () => clearTimeout(lpTmr), { passive: true });
document.body.addEventListener('pointermove', () => clearTimeout(lpTmr), { passive: true });

/* Swipe */
let ty0 = 0, tx0 = 0;
document.addEventListener('touchstart', e => { ty0 = e.touches[0].clientY; tx0 = e.touches[0].clientX; }, { passive: true });
document.addEventListener('touchend', e => {
    const dy = e.changedTouches[0].clientY - ty0;
    const dx = e.changedTouches[0].clientX - tx0;
    if (Math.abs(dy) > Math.abs(dx)) {
        if (dy > 80 && st.panelOpen) closePanel();
    } else {
        if (Math.abs(dx) > 60 && st.panelOpen) swipeTab(dx > 0 ? -1 : 1);
    }
}, { passive: true });

/* ── Panel ────────────────────────────────────────────────── */
function openPanel() {
    st.panelOpen = true; $('panel').classList.add('open');
    $('scene').style.transform = 'translateY(-10%) scale(0.83)';
    setExpr('focused');
    clearTimeout(panTmr); panTmr = setTimeout(closePanel, 10000);
}
function closePanel() {
    st.panelOpen = false; $('panel').classList.remove('open');
    $('scene').style.transform = '';
    clearTimeout(panTmr);
    if (st.expr === 'focused') setExpr('neutral');
}

const tabs = ['actions', 'explore', 'settings'];
function swipeTab(dir) {
    let idx = tabs.indexOf(st.panelTab) + dir;
    if (idx < 0) idx = tabs.length - 1; if (idx >= tabs.length) idx = 0;
    setTab(tabs[idx]);
}
function setTab(name) {
    st.panelTab = name;
    document.querySelectorAll('.ptab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
    document.querySelectorAll('.ppage').forEach(p => p.classList.toggle('active', p.id === 'page-' + name));
}

/* Panel button eye tracking */
document.querySelectorAll('.btn').forEach(btn => {
    btn.addEventListener('pointerenter', () => {
        const r = btn.getBoundingClientRect();
        movePupils((r.left + r.width / 2) / window.innerWidth * 26 - 13,
            (r.top + r.height / 2) / window.innerHeight * 18 - 9, 180);
        clearTimeout(panTmr); panTmr = setTimeout(closePanel, 10000);
    });
    btn.addEventListener('pointerleave', () => movePupils(0, 0, 300));
});

function onBtn(action) {
    setExpr('happy', 700);
    switch (action) {
        case 'talk':
            setStatusText('Listening…'); setExpr('listening');
            send({ type: 'toggle_mic', data: { muted: false } }); break;
        case 'mode':
            st.mode = st.mode === 'Auto' ? 'Manual' : 'Auto';
            $('sb-mode').textContent = st.mode;
            send({ type: 'switch_mode', data: { mode: st.mode.toLowerCase() } });
            setStatusText(`Mode: ${st.mode}`); break;
        case 'ai':
            st.ai = st.ai === 'Online' ? 'Offline' : 'Online';
            $('sb-ai').textContent = st.ai;
            $('sb-ai').className = `badge${st.ai === 'Offline' ? ' off' : ''}`;
            setStatusText(`AI: ${st.ai}`); break;
        case 'tp': closePanel(); startTP(); break;
        case 'home': send({ type: 'robot_action', data: { action: 'home' } }); setStatusText('Returning home…'); break;
        case 'explore': send({ type: 'robot_action', data: { action: 'explore' } }); setExpr('searching'); setStatusText('Exploring…'); break;
        case 'settings': setStatusText('Settings — coming soon'); break;
        case 'charge': setStatusText('Return to charge…'); break;
        case 'scan': setExpr('searching'); setStatusText('Scanning room…'); send({ type: 'robot_action', data: { action: 'scan' } }); break;
    }
}

/* ── Radial Menu ──────────────────────────────────────────── */
function openRadial() {
    if (st.panelOpen) closePanel();
    st.radialOpen = true; $('radial').classList.add('open');
    setExpr('curious');
}
function closeRadial() {
    st.radialOpen = false; $('radial').classList.remove('open');
    setExpr('neutral');
}
function onRadial(action) {
    closeRadial();
    switch (action) {
        case 'tp': startTP(); break;
        case 'explore': onBtn('explore'); break;
        case 'settings': openPanel(); setTab('settings'); break;
        case 'chat': setExpr('listening'); setStatusText('AI Chat mode…'); onBtn('talk'); break;
    }
}

/* ── Status Text ──────────────────────────────────────────── */
function setStatusText(msg, ms = 4000) {
    $('stxt').textContent = msg; $('stxt').classList.add('on');
    clearTimeout(stxtTmr);
    if (ms > 0) stxtTmr = setTimeout(() => $('stxt').classList.remove('on'), ms);
}

/* ── Notification ─────────────────────────────────────────── */
function notify(msg, type = 'info', ms = 4200) {
    const el = $('notif'); el.innerHTML = msg; el.className = `show ${type}`;
    clearTimeout(notifTmr); notifTmr = setTimeout(() => el.classList.remove('show'), ms);
}

/* ── Alert border ─────────────────────────────────────────── */
function alertBorder(lvl) {
    $('aborder').className = lvl || '';
    if (lvl) setTimeout(() => $('aborder').className = '', 5500);
}

/* ── Sleep ────────────────────────────────────────────────── */
function activateSleep() {
    st.sleeping = true; $('sleep-ovl').classList.add('on');
    setStatusText('Sleeping…', -1); spawnZzz();
}
function deactivateSleep() {
    st.sleeping = false; $('sleep-ovl').classList.remove('on');
    $('stxt').classList.remove('on'); stopZzz();
}
function wakeUp() {
    if (st.sleeping) { setExpr('excited', 600); setTimeout(() => setStatusText('Hello!', 1500), 300); }
}
function spawnZzz() {
    stopZzz(); zzzIv = setInterval(() => {
        if (!st.sleeping) { stopZzz(); return; }
        const z = document.createElement('div'); z.className = 'zzz';
        z.textContent = 'z'; z.style.left = (window.innerWidth / 2 + 55 + Math.random() * 35) + 'px';
        z.style.top = (window.innerHeight / 2 - 25) + 'px'; z.style.fontSize = (0.7 + Math.random() * 0.7) + 'rem';
        document.body.appendChild(z); setTimeout(() => z.remove(), 3200);
    }, 1600);
}
function stopZzz() { clearInterval(zzzIv); }
function resetSleepTimer() {
    clearTimeout(sleepTmr);
    sleepTmr = setTimeout(() => {
        setExpr('searching'); setStatusText('Looking for someone…', 5000);
        setTimeout(() => { if (!st.sleeping && !st.tpActive) setExpr('sleepy'); }, 6000);
    }, 50000);
}

/* ── Telepresence ─────────────────────────────────────────── */
const tpSt = { mic: true, cam: true, spk: true };

function startTP() {
    st.tpActive = true;
    const tp = $('tp');
    $('tp-loading').style.display = 'flex';
    $('tp-video').style.opacity = '0';
    tp.classList.add('active');
    $('eyes').style.opacity = '0'; $('eyes').style.transform = 'scale(0.5)';
    $('scene').querySelector('#stxt').classList.remove('on');
    tp.addEventListener('pointerdown', toggleTPBar, { passive: true });
    tpSecs = 0; if (tpTimerIv) clearInterval(tpTimerIv);
    setTimeout(() => {
        $('tp-loading').style.display = 'none';
        $('tp-video').style.opacity = '1';
        $('tp-cname').textContent = 'Connected';
        tpTimerIv = setInterval(() => {
            tpSecs++; const m = String(Math.floor(tpSecs / 60)).padStart(2, '0');
            const s = String(tpSecs % 60).padStart(2, '0');
            $('tp-dur').textContent = `${m}:${s}`;
        }, 1000);
    }, 2000);
    $('tp').dataset.hide = setTimeout(() => $('tp-bar').classList.add('hide'), 4500);
}
function toggleTPBar() {
    const bar = $('tp-bar'); bar.classList.remove('hide');
    clearTimeout($('tp').dataset.hide);
    $('tp').dataset.hide = setTimeout(() => bar.classList.add('hide'), 5000);
}
function stopTP() {
    st.tpActive = false; clearInterval(tpTimerIv);
    $('tp').classList.remove('active');
    $('eyes').style.opacity = '1'; $('eyes').style.transform = '';
    $('tp-bar').classList.remove('hide');
    $('tp-video').style.opacity = '0'; $('tp-loading').style.display = 'flex';
    $('tp-dur').textContent = '00:00';
    setExpr('neutral'); setStatusText('Back to robot mode', 2000);
}
function onTP(action) {
    switch (action) {
        case 'mic': tpSt.mic = !tpSt.mic;
            $('tp-ico-mic').className = `tp-ico ${tpSt.mic ? 'on' : 'off'}`;
            $('tp-ico-mic').textContent = tpSt.mic ? '🎤' : '🔇'; break;
        case 'cam': tpSt.cam = !tpSt.cam;
            $('tp-ico-cam').className = `tp-ico ${tpSt.cam ? 'on' : 'off'}`;
            $('tp-ico-cam').textContent = tpSt.cam ? '📷' : '🚫'; break;
        case 'spk': tpSt.spk = !tpSt.spk;
            $('tp-ico-spk').className = `tp-ico ${tpSt.spk ? 'on' : 'off'}`;
            $('tp-ico-spk').textContent = tpSt.spk ? '🔊' : '🔇'; break;
        case 'end': case 'robot': stopTP(); break;
    }
}

/* ── WebSocket ────────────────────────────────────────────── */
function connectWS() {
    const host = location.hostname || '127.0.0.1';
    ws = new WebSocket(`ws://${host}:8765`);
    ws.onopen = () => notify('🔗 Server connected', 'info');
    ws.onmessage = e => handleWS(JSON.parse(e.data));
    ws.onerror = () => { };
    ws.onclose = () => setTimeout(connectWS, 3000);
}
function send(o) { if (ws?.readyState === 1) ws.send(JSON.stringify(o)); }

function handleWS(m) {
    switch (m.type) {
        case 'telemetry':
            if (m.data.battery != null) {
                const b = m.data.battery; $('sb-bat').textContent = b + '%';
                if (b < 15) { alertBorder('danger'); notify('🔋 Low battery!', 'danger'); setExpr('sleepy'); }
                else if (b < 30) notify('🔋 Battery at ' + b + '%', 'warn');
            }
            if (m.data.wifi_rssi != null) {
                const sig = m.data.wifi_rssi > -60 ? 'Strong' : m.data.wifi_rssi > -80 ? 'Weak' : 'Lost';
                $('sb-wifi').textContent = sig;
                if (sig === 'Lost') { notify('📶 WiFi lost', 'warn'); setExpr('error', 2000); }
            }
            if (m.data.mode != null) { st.mode = m.data.mode; $('sb-mode').textContent = m.data.mode.charAt(0).toUpperCase() + m.data.mode.slice(1); }
            if (m.data.direction != null) { const d = m.data.direction; movePupils(d === 'left' ? -22 : d === 'right' ? 22 : 0, 0, 200); }
            if (m.data.obstacle != null && m.data.obstacle) { setExpr('alert', 2000); setStatusText('Obstacle detected!', 2000); notify('🚧 Obstacle!', 'warn'); }
            break;
        case 'emotion': setExpr(emotionMap(m.data.expression || m.data.emotion)); break;
        case 'status_text': setStatusText(m.data.text); break;
        case 'speaking_start': setExpr('speaking'); st.speaking = true; break;
        case 'speaking_stop': st.speaking = false; setExpr('neutral'); break;
        case 'eye_config': Object.assign(eyeCfg, m.data); applyEyeStyle(); break;
        case 'mode_change':
            if (m.data.mode === 'telepresence') startTP();
            else if (st.tpActive) stopTP();
            $('sb-mode').textContent = (m.data.mode || '').charAt(0).toUpperCase() + (m.data.mode || '').slice(1);
            break;
        case 'ai_response':
            if (m.data.emotion) setExpr(emotionMap(m.data.emotion));
            break;
        case 'call_incoming': notify('📞 Incoming call…', 'info', 6000); break;
        case 'wake_word': setExpr('listening'); setStatusText('Listening…', 3000); break;
    }
}

const EMAP = {
    excited: 'excited', happy: 'happy', joy: 'happy',
    sad: 'sad', anger: 'alert', stern: 'alert',
    curious: 'curious', interest: 'curious',
    love: 'happy', surprise: 'excited',
    fear: 'alert', disgust: 'error',
    thinking: 'thinking', speaking: 'speaking',
    neutral: 'neutral'
};
function emotionMap(e) { return EMAP[e] || 'neutral'; }

/* ── Autonomous mode status updates ──────────────────────── */
let lastExploreText = '';
function updateExploreText(txt) {
    if (txt !== lastExploreText) { lastExploreText = txt; setStatusText(txt, 5000); }
}

/* ── Boot ─────────────────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', () => {
    applyEyeStyle();
    setTab('actions');
    boot();
});
