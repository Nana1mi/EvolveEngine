import { appendFileSync, realpathSync, statSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

export default function (pi) {
  const c = JSON.parse(process.env.EVOLVE_PI_GUARD ?? "{}");
  if (c.edit.length !== 1 || c.read.length !== 1 || c.edit[0] !== c.read[0] || c.compile !== null) {
    throw new Error("Expected an isolated one-file read/edit contract");
  }
  const root = realpathSync(c.root);
  const expected = resolve(root, c.edit[0]);
  let attempts = 0;
  pi.on("session_start", () => writeFileSync(c.ready, "ready\n"));
  pi.on("tool_call", (event) => {
    const { toolName, input } = event;
    attempts += 1;
    const linesOK = toolName !== "read" || (Number.isSafeInteger(input?.offset) && input.offset >= 1 &&
       Number.isSafeInteger(input?.limit) && input.limit >= 1 && input.limit <= c.max_lines);
    let ok = false;
    if (attempts <= c.max_tools && (toolName === "read" || toolName === "edit") && linesOK && input?.path === c.edit[0]) {
      try { ok = realpathSync(expected) === expected && statSync(expected).isFile(); } catch { /* fail closed */ }
    }
    appendFileSync(c.receipt, JSON.stringify({tool: toolName, accepted: ok, ordinal: attempts}) + "\n");
    if (!ok) return { block: true, reason: "Calibration tool scope or signed lease exceeded" };
  });
}
