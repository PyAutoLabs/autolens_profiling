/* Catalogue-only browser. No temporal selection, cross-setup joins or scientific judgement. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const root = $("browser"),
    status = $("load-status"),
    nav = $("navigation"),
    results = $("results");
  const repo = "https://github.com/PyAutoLabs/autolens_profiling";
  let catalogue,
    state = {},
    generation = 0;
  const cache = new Map();
  const names = {
    imaging: "Imaging",
    interferometer: "Interferometer",
    datacube: "Datacube",
    point_source_image: "Point source · image plane",
    point_source_source: "Point source · source plane",
    multi_dataset: "Multiple datasets",
    cluster: "Cluster",
    lens: "Lens components",
    experiments: "Other experiments",
  };
  const models = {
    delaunay: "Delaunay",
    rectangular: "Rectangular",
    mge: "MGE",
    delaunay_nn: "Delaunay natural neighbour",
  };
  const label = (value) =>
    models[value] || names[value] || String(value).replaceAll("_", " ");
  function node(tag, text, cls) {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (cls) el.className = cls;
    return el;
  }
  function append(parent, tag, text, cls) {
    const el = node(tag, text, cls);
    parent.append(el);
    return el;
  }
  function source(path) {
    if (
      typeof path !== "string" ||
      path.split("/").some((p) => !p || p === "." || p === "..") ||
      /[\\:\x00-\x1f]/.test(path)
    )
      throw Error("Unsafe evidence path");
    return (
      repo +
      "/blob/" +
      encodeURIComponent(catalogue.producer_revision || "main") +
      "/" +
      path.split("/").map(encodeURIComponent).join("/")
    );
  }
  function link(parent, path, text) {
    const a = append(parent, "a", text);
    a.href = source(path);
    return a;
  }
  function disclosure(parent, title) {
    const d = append(parent, "details");
    append(d, "summary", title);
    return d;
  }
  const instrument = (s) => s.instrument || "unspecified";
  function configurationLabel(s) {
    if (s.role === "planned_baseline") return "Baseline · not measured";
    const info = catalogue.evidence_shards.find(
      (item) => item.setup_id === s.id,
    );
    const axes = {
      runtime: "Runtime",
      breakdown: "Breakdown",
      compile: "Compile",
      memory: "Memory",
    };
    const pixels = s.configuration.source_pixels?.value;
    return [
      s.role === "reference_candidate" ? "Reference candidate" : "Archive",
      info?.axes.map((a) => axes[a] || a).join(" + ") || "Unsuccessful run",
      info?.devices.map((d) => d.toUpperCase()).join(" / "),
      info?.precisions.join(" / "),
      pixels == null ? null : pixels + " source pixels",
      s.configuration_id.slice(0, 6),
    ]
      .filter(Boolean)
      .join(" · ");
  }

  async function verified(path, expected) {
    const response = await fetch(path);
    if (!response.ok)
      throw Error("Could not load evidence (HTTP " + response.status + ").");
    const bytes = await response.arrayBuffer();
    if (!crypto.subtle)
      throw Error(
        "Evidence verification needs HTTPS or localhost. Original evidence links remain available.",
      );
    const actual = Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
      (x) => x.toString(16).padStart(2, "0"),
    ).join("");
    if (actual !== expected)
      throw Error(
        "Evidence changed since this page was published. Reload the page to get a matching catalogue.",
      );
    return JSON.parse(new TextDecoder().decode(bytes));
  }
  function route(next, push = true) {
    state = next;
    const hash = new URLSearchParams(Object.entries(next).filter(([, v]) => v));
    if (push && location.hash.slice(1) !== hash.toString())
      history.pushState(null, "", "#" + hash);
    render();
  }
  function fromURL() {
    route(
      Object.fromEntries(new URLSearchParams(location.hash.slice(1))),
      false,
    );
  }
  function navigation() {
    nav.replaceChildren();
    const project = disclosure(nav, "AutoLens");
    project.id = "project-picker";
    project.open = true;
    const families = [
      ...new Set([
        ...catalogue.setups.map((s) => s.dataset),
        ...catalogue.navigation
          .filter((s) => s.category === "scientific_entrypoint")
          .map((s) => s.dataset),
      ]),
    ];
    families.sort(
      (a, b) => Object.keys(names).indexOf(a) - Object.keys(names).indexOf(b),
    );
    for (const dataset of families) {
      const family = disclosure(project, label(dataset));
      family.dataset.family = dataset;
      const choices = [
        ...new Set([
          ...catalogue.setups
            .filter((s) => s.dataset === dataset)
            .map((s) => s.model),
          ...catalogue.navigation
            .filter(
              (s) =>
                s.category === "scientific_entrypoint" && s.dataset === dataset,
            )
            .map((s) => s.model),
        ]),
      ].sort();
      for (const model of choices) {
        const button = append(family, "button", label(model), "model-choice");
        button.type = "button";
        button.dataset.dataset = dataset;
        button.dataset.model = model;
        button.onclick = () => {
          route({ dataset, model });
          $("results").scrollIntoView({ behavior: "smooth", block: "start" });
        };
      }
    }
    const shared = disclosure(project, "Shared measurement tools");
    const tools = [
      ...new Set(
        catalogue.navigation
          .filter((s) => s.category === "shared_measurement_tools")
          .map((s) => s.path.split("/")[2]),
      ),
    ];
    for (const tool of tools) {
      const details = disclosure(shared, label(tool));
      const ul = append(details, "ul", "", "evidence-list");
      catalogue.navigation
        .filter(
          (s) =>
            s.category === "shared_measurement_tools" &&
            s.path.split("/")[2] === tool,
        )
        .forEach((s) =>
          link(append(ul, "li"), s.path, s.path.split("/").at(-1)),
        );
    }
  }
  function selector(parent, id, title, options, value, change) {
    const l = append(parent, "label", title);
    l.htmlFor = id;
    const select = append(l, "select");
    select.id = id;
    options.forEach(([v, t]) => {
      const o = append(select, "option", t);
      o.value = v;
    });
    select.value = value;
    select.onchange = () => change(select.value);
    return select;
  }
  function metadata(setup) {
    const details = disclosure(results, "Configuration details");
    const dl = append(details, "dl", "", "config");
    const entries = Object.entries(setup.configuration || {});
    const priority = [
      "source_pixels",
      "psf_shape",
      "image_shape",
      "image_pixels_masked",
      "n_vis",
      "regularization",
      "transformer",
      "oversampling",
    ];
    entries.sort(
      ([a], [b]) =>
        (priority.includes(a) ? priority.indexOf(a) : 99) -
        (priority.includes(b) ? priority.indexOf(b) : 99),
    );
    for (const [key, entry] of entries) {
      append(dl, "dt", label(key));
      const dd = append(dl, "dd");
      if (entry.value === null) {
        append(dd, "span", "Not recorded", "muted");
        dd.title = entry.reason || "Unknown";
      } else
        append(
          dd,
          typeof entry.value === "object" ? "code" : "span",
          typeof entry.value === "object"
            ? JSON.stringify(entry.value)
            : String(entry.value),
        );
    }
  }
  function evidence(setup, records) {
    const details = disclosure(results, "Profiling evidence and scripts");
    const ul = append(details, "ul", "", "evidence-list");
    const anchors = new Map();
    if (setup.evidence)
      anchors.set(JSON.stringify(setup.evidence), setup.evidence);
    records.forEach((r) => anchors.set(JSON.stringify(r.evidence), r.evidence));
    for (const ev of anchors.values()) {
      const li = append(ul, "li");
      link(li, ev.path, ev.path);
      if (ev.fragment) append(li, "code", " · " + ev.fragment);
    }
    const scripts = catalogue.navigation.filter(
      (s) => s.dataset === setup.dataset && s.model === setup.model,
    );
    scripts.forEach((s) => link(append(ul, "li"), s.path, s.path));
    if (!anchors.size && !scripts.length)
      append(details, "p", "No evidence has been recorded for this setup.");
  }
  function qualification(records) {
    const d = disclosure(results, "Qualification and measurement method");
    for (const r of records) {
      const p = append(d, "p");
      append(p, "strong", r.metric + ": ");
      append(
        p,
        "span",
        "Exact recorded value: " +
          r.measurement[r.metric] +
          " " +
          r.unit +
          ". ",
      );
      append(p, "span", r.validation.status + " — " + r.validation.reason);
      append(
        d,
        "p",
        "Host: " +
          (r.provenance.host || "not recorded") +
          " · precision: " +
          (r.identity.precision || "not recorded") +
          " · software: " +
          JSON.stringify(r.identity.software),
        "metric-meta",
      );
      append(d, "p", "Method: " + JSON.stringify(r.method), "metric-meta");
    }
  }
  function panels(records) {
    const axes = {
      runtime: [
        "Likelihood runtime",
        "Full likelihood observations. Single-JIT blocks can include first-call effects; batch timings are not single-call latency.",
      ],
      breakdown: [
        "Likelihood breakdown",
        "Instrumented component costs are not the full compiled likelihood time. Bars compare values only within this setup and unit.",
      ],
      compile: [
        "Compilation and setup",
        "Trace, compile, first call and setup wall time are distinct measurements; do not add them together.",
      ],
      memory: [
        "Memory",
        "Observed host RSS is not GPU VRAM. Static compiler estimates are shown separately.",
      ],
    };
    for (const [axis, [title, note]] of Object.entries(axes)) {
      const section = disclosure(results, title);
      section.className = "metric-panel";
      section.open = axis === "runtime";
      append(section, "p", note, "axis-note");
      const rows = records.filter((r) => r.axis === axis);
      if (!rows.length) {
        append(section, "p", "Not measured for this configuration.", "empty");
        continue;
      }
      const groups = new Map();
      for (const r of rows) {
        const scope =
          r.metric === "vmap.batch_time"
            ? "Batch wall time"
            : r.metric.includes("vmap")
              ? "Per-replica batch cost"
              : axis === "runtime"
                ? "Single-call observations"
                : "Component observations";
        const key = [
          scope,
          r.identity.device || "device unknown",
          r.identity.backend || "backend unknown",
          r.identity.precision || "precision unknown",
          r.provenance.host || "host unknown",
          r.unit,
        ].join(" · ");
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(r);
      }
      for (const [key, group] of groups) {
        append(section, "p", key, "metric-meta");
        const list = append(section, "ul", "", "metric-list");
        const max = Math.max(...group.map((r) => r.measurement[r.metric]));
        for (const r of group) {
          const value = r.measurement[r.metric];
          const li = append(list, "li", "", "metric-row");
          const line = append(li, "div", "", "metric-title");
          append(line, "span", label(r.metric));
          const displayUnit =
            r.unit === "s" && value > 0 && value < 1 ? "ms" : r.unit;
          const displayValue = displayUnit === "ms" ? value * 1000 : value;
          const measured = append(
            line,
            "strong",
            Number(displayValue.toPrecision(4)) + " " + displayUnit,
            "metric-value",
          );
          measured.dataset.value = String(value);
          measured.dataset.unit = r.unit;
          measured.title = "Exact recorded value: " + value + " " + r.unit;
          const track = append(li, "div", "", "bar-track");
          track.setAttribute("aria-hidden", "true");
          const bar = append(track, "div", "", "bar");
          bar.style.width = (max > 0 ? (100 * value) / max : 0) + "%";
          append(
            li,
            "p",
            (r.method.statistic || "Statistic not recorded") +
              " · " +
              (r.method.repetitions === null
                ? "repetitions not recorded"
                : r.method.repetitions + " repetitions"),
            "metric-meta",
          );
        }
      }
    }
  }
  function advice(setup) {
    const items = [
      ...(catalogue.hazards || []),
      ...(catalogue.recommendations || []),
    ].filter((x) => x.applies_to.setup_ids.includes(setup.id));
    const d = disclosure(results, "Hazards and setup guidance");
    if (!items.length)
      append(
        d,
        "p",
        "No version-qualified hazard or recommendation is bound to this configuration. This is not evidence that it is hazard-free.",
      );
    for (const item of items) {
      append(d, "h4", item.title);
      append(d, "p", item.description);
      append(d, "p", item.applies_to.limitations);
      item.evidence.forEach((ev) => link(append(d, "p"), ev.path, ev.path));
    }
    const unbound = (catalogue.unbound_findings || []).filter((f) =>
      f.evidence.path.includes("/" + setup.dataset + "/"),
    );
    if (unbound.length) {
      const other = disclosure(
        d,
        "Related findings · applicability unverified",
      );
      unbound.forEach((f) => {
        append(other, "p", f.title + " — " + f.reason);
        link(append(other, "p"), f.evidence.path, f.evidence.path);
      });
    }
    const estimates = (catalogue.static_memory_estimates || []).filter(
      (e) => setup.evidence && e.evidence.path === setup.evidence.path,
    );
    if (estimates.length) {
      const other = disclosure(d, "Static compiler memory estimates");
      estimates.forEach((e) =>
        append(
          other,
          "p",
          e.per_replica.value + " " + e.per_replica.unit + " · " + e.reason,
        ),
      );
    }
  }
  async function render() {
    const ticket = ++generation;
    results.replaceChildren();
    results.removeAttribute("aria-busy");
    status.textContent = "";
    status.className = "";
    const choices = catalogue.setups.filter(
      (s) => s.dataset === state.dataset && s.model === state.model,
    );
    nav
      .querySelectorAll(".model-choice")
      .forEach((b) =>
        b.setAttribute(
          "aria-current",
          String(
            b.dataset.dataset === state.dataset &&
              b.dataset.model === state.model,
          ),
        ),
      );
    nav.querySelectorAll("[data-family]").forEach((d) => {
      if (d.dataset.family === state.dataset) d.open = true;
    });
    if (!state.dataset || !state.model) {
      results.hidden = true;
      return;
    }
    results.hidden = false;
    const picker = $("project-picker");
    picker.open = false;
    picker.querySelector("summary").textContent =
      "AutoLens / " +
      label(state.dataset) +
      " / " +
      label(state.model) +
      " · change setup";
    append(results, "h2", label(state.dataset) + " / " + label(state.model));
    if (!choices.length) {
      append(
        results,
        "p",
        "No catalogue measurements for this model yet.",
        "empty",
      );
      evidence({ dataset: state.dataset, model: state.model }, []);
      return;
    }
    const instruments = [...new Set(choices.map(instrument))].sort();
    if (
      (state.setup && !choices.some((s) => s.id === state.setup)) ||
      (state.instrument && !instruments.includes(state.instrument))
    ) {
      append(
        results,
        "p",
        "This linked configuration is not in the published catalogue. No substitute measurements are shown.",
        "empty",
      );
      const choose = append(
        results,
        "button",
        "Choose an available configuration",
        "retry",
      );
      choose.type = "button";
      choose.onclick = () =>
        route({ dataset: state.dataset, model: state.model });
      return;
    }
    const requested = choices.find((s) => s.id === state.setup);
    const selectedInstrument = requested
      ? instrument(requested)
      : instruments.includes(state.instrument)
        ? state.instrument
        : instrument(
            choices.find((s) => s.role === "reference_candidate") || choices[0],
          );
    const rank = (s) =>
      s.role === "reference_candidate"
        ? catalogue.records.some(
            (r) => r.setup_id === s.id && r.axis === "runtime",
          )
          ? 0
          : 1
        : s.role === "planned_baseline"
          ? 3
          : 2;
    const variants = choices
      .filter((s) => instrument(s) === selectedInstrument)
      .sort(
        (a, b) =>
          rank(a) - rank(b) ||
          configurationLabel(a).localeCompare(configurationLabel(b)),
      );
    // Baseline is a coverage placeholder, not a fallback claim that archive evidence is accepted.
    const setup = requested || variants[0];
    state = {
      dataset: state.dataset,
      model: state.model,
      instrument: selectedInstrument,
      setup: setup.id,
    };
    history.replaceState(null, "", "#" + new URLSearchParams(state));
    const fields = append(results, "div", "", "selectors");
    selector(
      fields,
      "instrument",
      "Instrument",
      instruments.map((i) => [
        i,
        i === "unspecified" ? "Not specified" : i.toUpperCase(),
      ]),
      selectedInstrument,
      (value) =>
        route({
          dataset: state.dataset,
          model: state.model,
          instrument: value,
        }),
    );
    selector(
      fields,
      "configuration",
      "Configuration / evidence run",
      variants.map((s) => [s.id, configurationLabel(s)]),
      setup.id,
      (value) => route({ ...state, setup: value }),
    );
    const context = append(results, "div", "", "context");
    append(
      context,
      "p",
      setup.role === "planned_baseline"
        ? "Baseline pending · no accepted measurement"
        : "Unreviewed archive evidence",
      "qualification",
    );
    append(
      context,
      "p",
      "Other devices or runs may have different settings. This view keeps each recorded configuration separate.",
      "muted",
    );
    const keySettings = [
      "source_pixels",
      "psf_shape",
      "image_pixels_masked",
      "n_vis",
    ];
    const facts = append(context, "p", "", "metric-meta");
    facts.textContent = keySettings
      .map(
        (k) =>
          label(k) +
          ": " +
          (setup.configuration[k]?.value == null
            ? "not recorded"
            : JSON.stringify(setup.configuration[k].value)),
      )
      .join(" · ");
    if (setup.role === "planned_baseline") {
      const slots = catalogue.planned_cells.filter(
        (c) => c.setup_id === setup.id,
      );
      append(
        context,
        "p",
        [...new Set(slots.map((s) => s.device.toUpperCase()))].join(" / ") +
          " · " +
          slots.length +
          " measurement slots not measured.",
      );
      panels([]);
      metadata(setup);
      advice(setup);
      evidence(setup, []);
      return;
    }
    const manifest = catalogue.evidence_shards.find(
      (s) => s.setup_id === setup.id,
    );
    if (!manifest) {
      const failed = catalogue.selections.filter(
        (s) => s.setup_id === setup.id,
      );
      append(
        context,
        "p",
        failed.map((s) => s.status + ": " + s.reason).join("; ") ||
          "No measured evidence is available.",
      );
      panels([]);
      metadata(setup);
      advice(setup);
      evidence(setup, []);
      return;
    }
    status.textContent = "Loading selected setup…";
    results.setAttribute("aria-busy", "true");
    try {
      if (!/^catalogue\/shards\/[a-f0-9]{20}\.json$/.test(manifest.path))
        throw Error("Invalid evidence shard path.");
      let shard = cache.get(manifest.sha256);
      if (!shard) {
        shard = await verified(manifest.path, manifest.sha256);
        cache.set(manifest.sha256, shard);
      }
      if (ticket !== generation) return;
      if (
        shard.version !== 2 ||
        shard.schema !== "profiling-summary" ||
        shard.setups.length !== 1 ||
        shard.setups[0].id !== setup.id ||
        JSON.stringify(shard.setups[0]) !== JSON.stringify(setup) ||
        shard.records.length !== manifest.records ||
        new Set(shard.records.map((r) => r.id)).size !== shard.records.length ||
        shard.records.some(
          (r) =>
            r.setup_id !== setup.id ||
            !Number.isFinite(r.measurement[r.metric]) ||
            r.measurement[r.metric] < 0,
        )
      )
        throw Error("Evidence does not match the selected setup.");
      panels(shard.records);
      metadata(setup);
      qualification(shard.records);
      advice(setup);
      evidence(setup, shard.records);
      status.textContent = "";
    } catch (error) {
      if (ticket !== generation) return;
      status.textContent = "Evidence unavailable. " + error.message;
      status.className = "status-error";
      const retry = append(results, "button", "Retry evidence", "retry");
      retry.type = "button";
      retry.onclick = () => {
        cache.delete(manifest.sha256);
        render();
      };
      metadata(setup);
      evidence(setup, []);
    } finally {
      if (ticket === generation) results.removeAttribute("aria-busy");
    }
  }
  async function boot() {
    try {
      catalogue = await verified("catalogue.json", root.dataset.catalogueSha);
      if (
        catalogue.schema !== "profiling-summary" ||
        catalogue.version !== 2 ||
        !Array.isArray(catalogue.evidence_shards)
      )
        throw Error("Unsupported setup catalogue.");
      navigation();
      window.addEventListener("hashchange", fromURL);
      fromURL();
    } catch (error) {
      status.textContent = "Catalogue unavailable. " + error.message;
      status.className = "status-error";
      const a = append(status, "a", " Open original evidence.");
      a.href = repo + "/tree/main/results";
    }
  }
  boot();
})();
