// Sidebar starts collapsed except the active page's own chain.
// The theme renders every top-level section as <details open>; closing the
// non-active ones keeps the hierarchy readable. The native <details>/<summary>
// chevron toggles keep working for the user to open sections manually.
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".bd-sidebar-primary details").forEach(function (details) {
    if (!details.querySelector("a.current")) {
      details.removeAttribute("open");
    }
  });
});
