/* Desktop conventions for the GliaAnalysis window (after NED-Net's app_shell.js).
 *
 * - Starts in dark mode when the system is set to dark (and follows changes).
 * - Ctrl+1..9 (Cmd on macOS) go to the 1st..9th page in the sidebar;
 *   Ctrl/Cmd+O opens a project folder; Ctrl/Cmd+, opens Settings.
 * - Elements with data-go="<page id>" (status-bar items) open that page.
 * - The logo in the sidebar goes to Setup; the start panel's
 *   "Open project…" presses the toolbar button.
 * - window.glia.* is what the native menu (desktop.py) calls.
 */
(function () {
    "use strict";

    function byId(id) { return document.getElementById(id); }

    // Run fn once the element exists (Dash renders after page load).
    function whenPresent(id, fn, timeoutMs) {
        var waited = 0, step = 100, limit = timeoutMs || 15000;
        (function poll() {
            var el = byId(id);
            if (el) { fn(el); return; }
            waited += step;
            if (waited < limit) { setTimeout(poll, step); }
        })();
    }

    // Sidebar page buttons have pattern-matching ids: {"id":"setup","type":"tab"}
    function navItems() {
        return Array.prototype.slice.call(document.querySelectorAll("#sidebar-nav .nav-item"));
    }

    function pageIdOf(el) {
        try { return JSON.parse(el.id).id; } catch (e) { return null; }
    }

    function go(pageId) {
        // Pages in the list plus Settings in the sidebar footer
        var items = Array.prototype.slice.call(document.querySelectorAll("#sidebar .nav-item"));
        for (var i = 0; i < items.length; i++) {
            if (pageIdOf(items[i]) === pageId) { items[i].click(); return; }
        }
    }

    function goIndex(i) {
        var items = navItems();
        if (items[i]) { items[i].click(); }
    }

    function openProject() {
        var btn = byId("toolbar-open-project");
        if (btn) { btn.click(); }
    }

    // Cell type: microglia | astrocyte
    function setMode(mode) {
        var inputs = document.querySelectorAll("#sidebar-mode input");
        for (var i = 0; i < inputs.length; i++) {
            var inp = inputs[i];
            if (inp.value === mode || (inp.id || "").slice(-mode.length - 1) === "_" + mode) {
                if (!inp.checked) { inp.click(); }
                return;
            }
        }
    }

    function isDark() {
        var t = byId("theme-toggle");
        return !!(t && t.checked);
    }

    function setDark(wantDark) {
        var t = byId("theme-toggle");
        if (t && t.checked !== wantDark) { t.click(); }
    }

    function toggleTheme() { setDark(!isDark()); }

    document.addEventListener("click", function (e) {
        if (!e.target.closest) { return; }
        if (e.target.closest("#brand-home")) { go("setup"); }
        if (e.target.closest("#start-open-project")) { openProject(); }
        var link = e.target.closest("[data-go]");
        if (link) { go(link.getAttribute("data-go")); }
    });

    window.glia = { go: go, goIndex: goIndex, openProject: openProject,
                    setMode: setMode, toggleTheme: toggleTheme };

    // Follow the system appearance
    var media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
    if (media) {
        whenPresent("theme-toggle", function () {
            if (media.matches) { setDark(true); }
        });
        var onChange = function (e) { setDark(e.matches); };
        if (media.addEventListener) { media.addEventListener("change", onChange); }
        else if (media.addListener) { media.addListener(onChange); }
    }

    // Keyboard shortcuts
    var isMac = /Mac/.test(navigator.platform || navigator.userAgent);
    document.addEventListener("keydown", function (e) {
        var mod = isMac ? e.metaKey : e.ctrlKey;
        if (!mod || e.altKey || e.shiftKey) { return; }
        var key = (e.key || "").toLowerCase();
        if (key >= "1" && key <= "9") {
            e.preventDefault();
            goIndex(parseInt(key, 10) - 1);
        } else if (key === "o") {
            e.preventDefault();
            openProject();
        } else if (key === ",") {
            e.preventDefault();
            go("settings");
        }
    });
})();
