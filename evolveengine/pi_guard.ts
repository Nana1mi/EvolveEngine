import { existsSync, realpathSync, statSync, writeFileSync } from "node:fs";
import { isAbsolute, relative, resolve, sep } from "node:path";

export default function (pi) {
  const cfg = JSON.parse(process.env.EVOLVE_PI_GUARD ?? "{}");
  const root = realpathSync(cfg.root);
  const within = (path, scope) => scope.some((item) => path === item || path.startsWith(item + "/"));
  function checkedPath(input, scope) {
    if (typeof input !== "string" || !input || input.startsWith("@")) return false;
    try {
      const path = resolve(root, input);
      if (!existsSync(path) || !statSync(path).isFile()) return false;
      const real = realpathSync(path);
      const name = relative(root, real).split(sep).join("/");
      return !isAbsolute(name) && within(name, scope);
    } catch {
      return false;
    }
  }
  pi.on("session_start", () => writeFileSync(cfg.ready, "ready\n"));
  pi.on("tool_call", (event) => {
    const { toolName, input } = event;
    if (toolName === "read" && checkedPath(input.path, cfg.read)) return;
    if (toolName === "edit" && checkedPath(input.path, cfg.edit)) return;
    if (toolName === "bash" && cfg.compile && input.command === cfg.compile) {
      input.timeout = 15;
      return;
    }
    return { block: true, reason: "This tool action is outside the selected optimization contract" };
  });
}
