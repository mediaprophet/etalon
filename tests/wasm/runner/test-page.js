const MAX_DIAGNOSTICS = 32;
const MAX_MESSAGE_LENGTH = 2048;

export async function runTestPage(page, url) {
  const diagnostics = [];
  const record = (message) => {
    diagnostics.push(String(message).slice(0, MAX_MESSAGE_LENGTH));
    if (diagnostics.length > MAX_DIAGNOSTICS) diagnostics.shift();
  };
  page.on("console", (message) =>
    record(`${message.type()}: ${message.text()}`),
  );
  page.on("pageerror", (error) => record(`pageerror: ${error.message}`));

  let response;
  try {
    await page.goto(url);
    const handle = await page.waitForFunction(() => {
      const failure = window.sessionStorage.test_failure;
      if (failure) return { passed: false, message: failure };
      if (window.sessionStorage.test_success) return { passed: true };
      return false;
    });
    const result = await handle.jsonValue();
    await handle.dispose();
    response = result.passed
      ? { status: 200, body: "OK" }
      : {
          status: 500,
          body: [
            String(result.message).slice(0, MAX_MESSAGE_LENGTH),
            ...diagnostics,
          ].join("\n"),
        };
  } catch (error) {
    response = {
      status: 500,
      body: [
        `WASM browser test failed: ${String(error.message).slice(0, MAX_MESSAGE_LENGTH)}`,
        ...diagnostics,
      ].join("\n"),
    };
  } finally {
    try {
      await page.close();
    } catch (error) {
      response = {
        status: 500,
        body: `${response.body}\nFailed to close browser page: ${String(error.message).slice(0, MAX_MESSAGE_LENGTH)}`,
      };
    }
  }
  return response;
}
