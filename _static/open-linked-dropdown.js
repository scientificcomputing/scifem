// Open a collapsed `:class: dropdown` admonition when the page is navigated to it.
//
// sphinx-togglebutton collapses with a `toggle-hidden` class rather than a native
// `<details>`, so the browser cannot expand it for us and a link to a target inside one
// would scroll to a box that is still shut.
(function () {
  function openTarget() {
    if (!window.location.hash) {
      return;
    }
    var target = document.getElementById(window.location.hash.slice(1));
    if (!target) {
      return;
    }
    var box = target.closest(".admonition.toggle");
    if (box && box.classList.contains("toggle-hidden")) {
      box.classList.remove("toggle-hidden");
      // Keep the button's label and aria-expanded in step; togglebutton exposes this.
      if (typeof window.syncAllToggleHints === "function") {
        window.syncAllToggleHints();
      }
      box.scrollIntoView({ block: "center" });
    }
  }

  // togglebutton adds `toggle-hidden` on DOMContentLoaded, so run after it.
  window.addEventListener("load", openTarget);
  window.addEventListener("hashchange", openTarget);
})();
