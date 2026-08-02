/* Mermaid theme follower for sphinx-book-theme.
 *
 * sphinxcontrib-mermaid injects mermaid.min.js and an init call. We replace
 * the static theme with a getter against document.documentElement.dataset.theme
 * ("light" | "dark" | "auto"; "auto" falls back to light).
 *
 * On theme toggle, the previously-rendered diagrams are reset (source
 * restored, data-processed cleared) and mermaid.run() is called again.
 */
(function () {
  "use strict";

  function currentTheme() {
    var t = document.documentElement.dataset.theme || "light";
    return t === "dark" ? "dark" : "default";
  }

  function cacheSources() {
    document.querySelectorAll("pre.mermaid, div.mermaid").forEach(function (el) {
      if (!el.dataset.source) {
        el.dataset.source = el.textContent.trim();
      }
    });
  }

  function resetRenderedDiagrams() {
    document.querySelectorAll("pre.mermaid, div.mermaid").forEach(function (el) {
      if (el.dataset.source) {
        el.removeAttribute("data-processed");
        el.innerHTML = el.dataset.source;
      }
    });
  }

  function applyTheme() {
    if (typeof mermaid === "undefined") {
      return;
    }
    mermaid.initialize({
      startOnLoad: false,
      theme: currentTheme(),
      securityLevel: "loose",
    });
    resetRenderedDiagrams();
    try {
      mermaid.run();
    } catch (e) {
      // mermaid.run throws if there are no nodes left to process — safe to ignore.
    }
  }

  function init() {
    if (typeof mermaid === "undefined") {
      // sphinxcontrib-mermaid loads mermaid.min.js after DOMContentLoaded;
      // retry shortly.
      window.setTimeout(init, 60);
      return;
    }
    cacheSources();
    applyTheme();

    new MutationObserver(function (mutations) {
      mutations.forEach(function (m) {
        if (m.attributeName === "data-theme") {
          applyTheme();
        }
      });
    }).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
