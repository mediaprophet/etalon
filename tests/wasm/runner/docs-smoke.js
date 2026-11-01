import { chromium } from "playwright";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, value, index, all) => {
    if (index % 2 === 0) pairs.push([value, all[index + 1]]);
    return pairs;
  }, []),
);
const mode = args["--mode"];
const featureLevel = args["--feature-level"] || "core";
if (!["default-availability", "software-webgpu"].includes(mode))
  throw new Error("Unknown browser qualification mode");
if (!["core", "compatibility"].includes(featureLevel))
  throw new Error("Unknown feature level");
for (const name of ["--chrome", "--profile", "--url"])
  if (!args[name]) throw new Error(`Missing ${name}`);

const browserArgs = [
  "--disable-background-networking",
  "--disable-component-update",
  "--disable-extensions",
  "--disable-sync",
  "--disk-cache-size=1048576",
  "--media-cache-size=1048576",
];
if (mode === "software-webgpu")
  browserArgs.push(
    "--enable-unsafe-webgpu",
    "--use-webgpu-adapter=swiftshader",
    "--use-gpu-in-tests",
  );
const context = await chromium.launchPersistentContext(args["--profile"], {
  executablePath: args["--chrome"],
  headless: true,
  viewport: { width: 1280, height: 900 },
  args: browserArgs,
});
const messages = [];
try {
  const page = context.pages()[0] || (await context.newPage());
  page.on("pageerror", (error) => {
    if (messages.length < 8) messages.push(String(error).slice(0, 1000));
  });
  await page.goto(args["--url"], { waitUntil: "load", timeout: 30000 });
  const selector = page.locator("#feature-level");
  if ((await selector.count()) > 0) await selector.selectOption(featureLevel);
  else if (featureLevel !== "core") throw new Error("Feature selector missing");
  await page.locator("#run-tests").click();
  await page.waitForFunction(
    () => document.body.dataset.etalonState === "complete",
    null,
    { timeout: 60000 },
  );
  const environment = await page.locator("#environment").innerText();
  process.stderr.write(environment.slice(0, 16000) + "\n");
  const document = await page.content();
  if (Buffer.byteLength(document) > 2 * 1024 * 1024)
    throw new Error("Browser DOM budget exceeded");
  process.stdout.write(document);
} catch (error) {
  throw new Error(`${error}; browser errors: ${messages.join("; ")}`);
} finally {
  await context.close();
}
