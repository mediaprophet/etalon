let loading;
export function loadRuntime() {
  if (!loading) {
    const revision = document.querySelector(
      'meta[name="etalon-revision"]',
    ).content;
    loading = (async () => {
      const module = await import(
        "./browser/etalon_browser_demo.js?rev=" + revision
      );
      await module.default({
        module_or_path: new URL(
          "./browser/etalon_browser_demo_bg.wasm?rev=" + revision,
          import.meta.url,
        ),
      });
      return module;
    })().catch((error) => {
      loading = undefined;
      throw error;
    });
  }
  return loading;
}
