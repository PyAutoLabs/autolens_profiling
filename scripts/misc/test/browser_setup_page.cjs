/* Real Chromium interaction tests. Run with Playwright on NODE_PATH. */
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const http = require("node:http");
const fs = require("node:fs/promises");
const path = require("node:path");
const root = path.resolve(__dirname, "../../../dashboard");
(async () => {
  const server = http.createServer(async (req, res) => {
    try {
      const name = decodeURIComponent(
        new URL(req.url, "http://localhost").pathname,
      );
      const target = path.resolve(
        root,
        "." + (name === "/" ? "/index.html" : name),
      );
      if (!target.startsWith(root + path.sep)) throw Error("bad path");
      res.setHeader(
        "Content-Type",
        target.endsWith(".json")
          ? "application/json"
          : "text/html; charset=utf-8",
      );
      res.end(await fs.readFile(target));
    } catch {
      res.statusCode = 404;
      res.end("Not found");
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = "http://127.0.0.1:" + server.address().port;
  let browser;
  try {
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(base);
    await page.locator(".model-choice").first().waitFor({ state: "attached" });
    assert.equal(await page.locator(".hero svg").count(), 1);
    assert.equal(await page.locator("#results").isVisible(), false);
    const selectModel = async (dataset, model) => {
      await page.locator('[data-family="' + dataset + '"] > summary').click();
      await page
        .locator(
          '.model-choice[data-dataset="' +
            dataset +
            '"][data-model="' +
            model +
            '"]',
        )
        .click();
      await page.waitForFunction(
        () => !document.getElementById("results").hasAttribute("aria-busy"),
      );
    };
    await selectModel("imaging", "delaunay");
    await page.locator("#instrument").selectOption("hst");
    await page.waitForFunction(
      () => !document.getElementById("results").hasAttribute("aria-busy"),
    );
    assert.match(
      await page.locator("#results h2").innerText(),
      /Imaging \/ Delaunay/,
    );
    assert.match(await page.locator(".context").innerText(), /Unreviewed/);
    assert.ok((await page.locator(".metric-value").count()) > 0);
    assert.ok(
      (await page.locator(".metric-panel[open] .metric-value").count()) > 0,
    );
    assert.match(await page.locator(".context").innerText(), /1500/);
    const original = page.url();
    await page.locator("#instrument").selectOption("euclid");
    await page
      .locator("#configuration")
      .selectOption({ label: "Baseline · not measured" });
    assert.match(
      await page.locator(".context").innerText(),
      /Baseline pending/,
    );
    assert.equal(await page.locator(".metric-value").count(), 0);
    await page.goBack();
    await page.waitForFunction(
      () => document.querySelector("#instrument")?.value === "euclid",
    );
    await page.goBack();
    await page.waitForFunction(
      () =>
        document.querySelector("#instrument")?.value === "hst" &&
        !document.getElementById("results").hasAttribute("aria-busy"),
    );
    assert.equal(page.url(), original);
    await page.goForward();
    await page.waitForFunction(
      () => document.querySelector("#instrument")?.value === "euclid",
    );
    await page.goto(original);
    await page.waitForFunction(() => document.querySelector(".metric-value"));
    assert.equal(await page.locator("#instrument").inputValue(), "hst");
    // Numeric values and evidence belong to exactly the selected source row.
    const doc = JSON.parse(
      await fs.readFile(path.join(root, "catalogue.json"), "utf8"),
    );
    const sid = await page.locator("#configuration").inputValue();
    const manifest = doc.evidence_shards.find((s) => s.setup_id === sid);
    const shard = JSON.parse(
      await fs.readFile(path.join(root, manifest.path), "utf8"),
    );
    const shown = await page
      .locator(".metric-value")
      .evaluateAll((nodes) =>
        nodes.map((n) => n.dataset.value + " " + n.dataset.unit),
      );
    assert.deepEqual(
      shown.sort(),
      shard.records
        .map((r) => String(r.measurement[r.metric]) + " " + r.unit)
        .sort(),
    );
    assert.equal(
      await page
        .locator("#results details")
        .filter({
          has: page.locator("summary", {
            hasText: "Profiling evidence and scripts",
          }),
        })
        .getAttribute("open"),
      null,
    );
    for (const width of [320, 390, 768, 1280]) {
      await page.setViewportSize({ width, height: 900 });
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth + 1,
        ),
        "page overflow at " + width,
      );
      if (process.env.BROWSER_ARTIFACT_DIR) {
        await fs.mkdir(process.env.BROWSER_ARTIFACT_DIR, { recursive: true });
        await page.screenshot({
          path: path.join(
            process.env.BROWSER_ARTIFACT_DIR,
            "setup-" + width + ".png",
          ),
          fullPage: true,
        });
      }
    }
    await page.emulateMedia({ colorScheme: "dark" });
    assert.ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    );
    if (process.env.BROWSER_ARTIFACT_DIR)
      await page.screenshot({
        path: path.join(process.env.BROWSER_ARTIFACT_DIR, "setup-dark.png"),
        fullPage: true,
      });
    // Keyboard focus and selection remain native.
    await page.locator("#instrument").focus();
    assert.equal(
      await page.evaluate(() => document.activeElement.id),
      "instrument",
    );
    await page.keyboard.press("Home");
    await page.keyboard.press("Enter");
    // A corrupt shard must not leave measurements from another setup visible.
    const broken = await browser.newPage();
    await broken.route("**/catalogue/shards/*.json", (r) =>
      r.fulfill({ body: "{}", contentType: "application/json" }),
    );
    await broken.goto(original);
    await broken.getByText("Retry evidence", { exact: true }).waitFor();
    assert.match(
      await broken.locator("#load-status").innerText(),
      /Evidence changed/,
    );
    assert.equal(await broken.locator(".metric-value").count(), 0);
    await broken.unroute("**/catalogue/shards/*.json");
    await broken.getByText("Retry evidence", { exact: true }).click();
    await broken.waitForFunction(() =>
      document.querySelector(".metric-value"),
    );
    // Late responses cannot overwrite a newer instrument selection.
    const delayed = await browser.newPage();
    await delayed.route("**/catalogue/shards/*.json", async (r) => {
      await new Promise((ok) => setTimeout(ok, 350));
      await r.continue();
    });
    await delayed.goto(original);
    await delayed.locator("#instrument").waitFor();
    await delayed.locator("#instrument").selectOption("euclid");
    await delayed
      .locator("#configuration")
      .selectOption({ label: "Baseline · not measured" });
    await delayed.waitForTimeout(700);
    assert.match(
      await delayed.locator(".context").innerText(),
      /Baseline pending/,
    );
    assert.equal(await delayed.locator(".metric-value").count(), 0);
    // Changing dataset routes updates the same complete result surface.
    await page.locator("#project-picker > summary").click();
    await selectModel("interferometer", "delaunay");
    assert.match(
      await page.locator("#results h2").innerText(),
      /Interferometer/,
    );
    // A stale deep link must not silently substitute a different measurement.
    const unknown = await browser.newPage();
    await unknown.goto(base + "#dataset=imaging&model=delaunay&setup=missing");
    await unknown
      .getByText("Choose an available configuration", { exact: true })
      .waitFor();
    assert.equal(await unknown.locator(".metric-value").count(), 0);
    await unknown
      .getByText("Choose an available configuration", { exact: true })
      .click();
    await unknown.waitForFunction(() =>
      document.querySelector(".metric-value"),
    );
    // Missing root catalogue and JavaScript-disabled fallback are actionable.
    const unavailable = await browser.newPage();
    await unavailable.route("**/catalogue.json", (r) =>
      r.fulfill({ status: 503, body: "Unavailable" }),
    );
    await unavailable.goto(base);
    await unavailable
      .getByText("Open original evidence.", { exact: true })
      .waitFor();
    const nojs = await browser.newContext({ javaScriptEnabled: false });
    const fallback = await nojs.newPage();
    await fallback.goto(base);
    assert.ok((await fallback.locator("noscript a").count()) > 0);
    // Regression: no reference candidate, but runtime evidence exists. Never
    // greet users with an empty runtime panel and collapsed measured breakdown.
    const journey = await browser.newPage();
    journey.on("pageerror", (e) => errors.push(e.message));
    const ready = async () =>
      journey.waitForFunction(
        () =>
          document.querySelector("#configuration") &&
          !document.querySelector("#results").hasAttribute("aria-busy"),
      );
    for (const model of ["mge", "rectangular"]) {
      await journey.goto(
        base + "#dataset=imaging&model=" + model + "&instrument=hst",
      );
      await ready();
      assert.ok(
        await journey
          .locator('.metric-panel[data-axis="runtime"][open] .metric-value')
          .count(),
      );
      const active = await journey.locator("#configuration").inputValue();
      assert.ok(
        doc.evidence_shards
          .find((s) => s.setup_id === active)
          .axes.includes("runtime"),
      );
      assert.equal(
        await journey.locator(".measurement-choice button").count(),
        4,
      );
      for (const axis of ["breakdown", "compile", "memory", "runtime"]) {
        await journey
          .locator('.measurement-choice button[data-axis="' + axis + '"]')
          .click();
        await ready();
        assert.equal(
          new URL(journey.url()).hash.includes("axis=" + axis),
          true,
        );
        const refs = doc.evidence_shards.filter(
          (s) =>
            s.setup_id.startsWith("imaging/" + model + "/hst/") &&
            s.axes.includes(axis),
        );
        if (refs.length) {
          assert.ok(
            await journey
              .locator(
                '.metric-panel[data-axis="' + axis + '"][open] .metric-value',
              )
              .count(),
            model + "/" + axis,
          );
          const options = await journey
            .locator("#device option")
            .evaluateAll((els) => els.map((e) => e.value).filter(Boolean));
          if (options.length) {
            await journey.locator("#device").selectOption(options[0]);
            await ready();
            const selected = doc.evidence_shards.find(
              (s) =>
                s.setup_id ===
                new URLSearchParams(new URL(journey.url()).hash.slice(1)).get(
                  "setup",
                ),
            );
            assert.ok(selected.axis_devices[axis].includes(options[0]));
            const runOptions = await journey
              .locator("#configuration option")
              .evaluateAll((els) => els.map((e) => e.value));
            const another = runOptions.find(
              (id) =>
                id !== selected.setup_id &&
                doc.evidence_shards.some((s) => s.setup_id === id),
            );
            if (another) {
              await journey.locator("#configuration").selectOption(another);
              await ready();
              assert.equal(
                await journey.locator("#device").inputValue(),
                options[0],
              );
              assert.equal(
                new URLSearchParams(new URL(journey.url()).hash.slice(1)).get(
                  "device",
                ),
                options[0],
              );
            }
          }
        } else {
          assert.equal(await journey.locator(".metric-value").count(), 0);
          assert.match(
            await journey.locator(".measurement-overview").innerText(),
            /No recorded measurements/,
          );
        }
      }
      // Use keyboard activation for a named measurement, then check history.
      const beforeAxis = journey.url();
      await journey
        .locator('.measurement-choice button[data-axis="breakdown"]')
        .focus();
      await journey.keyboard.press("Enter");
      await ready();
      await journey.goBack();
      await ready();
      assert.equal(journey.url(), beforeAxis);
      assert.equal(
        await journey.locator(".related-hazards").count(),
        model === "rectangular" ? 1 : 0,
      );
      await journey
        .getByText("Hazards and setup guidance", { exact: true })
        .click();
      await journey.locator(".shared-hazards > summary").click();
      assert.match(
        await journey.locator(".shared-hazards").innerText(),
        /does not establish/,
      );
      if (model === "rectangular") {
        await journey.locator(".related-hazards > summary").click();
        assert.match(
          await journey.locator(".related-hazards").innerText(),
          /Absolute inversion floors/,
        );
      }
      for (const width of [320, 390, 768, 1280]) {
        await journey.setViewportSize({ width, height: 900 });
        assert.ok(
          await journey.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth + 1,
          ),
        );
      }
    }
    // Exact breakdown-only links still open their data rather than switching run.
    const breakdown = doc.evidence_shards.find(
      (s) =>
        s.setup_id.startsWith("imaging/mge/hst/") &&
        s.axes.length === 1 &&
        s.axes[0] === "breakdown",
    );
    await journey.goto(
      base +
        "#" +
        new URLSearchParams({
          dataset: "imaging",
          model: "mge",
          instrument: "hst",
          setup: breakdown.setup_id,
        }),
    );
    await ready();
    assert.equal(
      await journey.locator("#configuration").inputValue(),
      breakdown.setup_id,
    );
    assert.ok(
      await journey
        .locator('.metric-panel[data-axis="breakdown"][open] .metric-value')
        .count(),
    );
    await journey
      .locator('.metric-panel[data-axis="runtime"] > summary')
      .click();
    assert.match(
      await journey.locator('.metric-panel[data-axis="runtime"]').innerText(),
      /other runs contain/,
    );
    await journey
      .getByRole("button", { name: "Browse runtime runs", exact: true })
      .click();
    await ready();
    assert.ok(
      await journey
        .locator('.metric-panel[data-axis="runtime"][open] .metric-value')
        .count(),
    );
    const cpuRun = doc.evidence_shards.find(
      (s) =>
        s.setup_id.startsWith("imaging/delaunay/hst/") &&
        s.axis_devices.runtime?.includes("cpu") &&
        !s.axis_devices.runtime.includes("a100"),
    );
    await journey.goto(
      base +
        "#" +
        new URLSearchParams({
          dataset: "imaging",
          model: "delaunay",
          instrument: "hst",
          axis: "runtime",
          device: "a100",
          setup: cpuRun.setup_id,
        }),
    );
    await journey
      .getByText("Open the exact run without the device filter", {
        exact: true,
      })
      .waitFor();
    assert.equal(await journey.locator(".metric-value").count(), 0);
    await journey
      .getByText("Open the exact run without the device filter", {
        exact: true,
      })
      .click();
    await ready();
    assert.equal(
      await journey.locator("#configuration").inputValue(),
      cpuRun.setup_id,
    );
    assert.deepEqual(errors, []);
    console.log(
      "Browser checks passed: navigation, values, history, deep links, keyboard, 4 widths, dark mode, corrupt/retry, delayed response, unavailable catalogue and no-JS fallback.",
    );
  } finally {
    if (browser) await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
