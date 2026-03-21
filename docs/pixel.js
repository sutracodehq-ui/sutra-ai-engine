(function() {
    console.log("⚡ SutraAI Click Shield Active");

    // 1. Configuration
    const script = document.currentScript;
    const params = new URLSearchParams(script.src.split('?')[1]);
    const tenantId = params.get('tid');
    const adId = params.get('ad');

    if (!tenantId || !adId) {
        console.error("ClickShield: Missing tid or ad parameter");
        return;
    }

    // 2. Signal Tracking
    let signals = {
        mouse_moves: 0,
        scroll_depth: 0,
        touch_events: 0,
        start_time: Date.now()
    };

    window.addEventListener('mousemove', () => signals.mouse_moves++, { once: false });
    window.addEventListener('touchstart', () => signals.touch_events++, { once: false });
    window.addEventListener('scroll', () => {
        const depth = (window.scrollY + window.innerHeight) / document.documentElement.scrollHeight;
        signals.scroll_depth = Math.max(signals.scroll_depth, depth);
    }, { once: false });

    // 3. Fingerprinting (Basic)
    function getFingerprint() {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        ctx.textBaseline = "top";
        ctx.font = "14px 'Arial'";
        ctx.textBaseline = "alphabetic";
        ctx.fillStyle = "#f60";
        ctx.fillRect(125,1,62,20);
        ctx.fillStyle = "#069";
        ctx.fillText("SutraShield", 2, 15);
        ctx.fillStyle = "rgba(102, 204, 0, 0.7)";
        ctx.fillText("SutraShield", 4, 17);
        return btoa(canvas.toDataURL());
    }

    // 4. Reporting
    async function track() {
        const payload = {
            ad_id: adId,
            tenant_id: parseInt(tenantId),
            client_data: {
                ua: navigator.userAgent,
                resolution: `${window.screen.width}x${window.screen.height}`,
                timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
                language: navigator.language,
                fingerprint: getFingerprint().substring(0, 32),
                signals: {
                    mouse_moves: signals.mouse_moves,
                    scroll_depth: parseFloat(signals.scroll_depth.toFixed(2)),
                    touch_events: signals.touch_events,
                    time_on_page_ms: Date.now() - signals.start_time,
                    is_trusted_viewer: true
                }
            },
            referrer: document.referrer
        };

        try {
            const response = await fetch('/v1/clicks/track', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await response.json();
            
            if (result.is_blocked) {
                console.warn("🛡️ ClickShield: Blocked suspicious interaction");
                // Optional: Redirect or hide content
                // document.body.innerHTML = "<h1>Access Denied</h1>";
            }
        } catch (e) {
            // Silently fail to not break user experience
        }
    }

    // Track on unload or after a few seconds
    window.addEventListener('beforeunload', track);
    setTimeout(track, 3000); // Also track after 3 seconds for bot detection

})();
