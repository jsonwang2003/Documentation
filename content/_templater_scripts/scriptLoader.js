/*
Obsidian University Workflow
Copyright (c) 2026 Jason Klein
MIT License

This script provides a cross-environment module loader (`requireScript`)
that works in both Obsidian's Templater environment and a Node.js test
environment. It allows other scripts to load their dependencies using a
single, consistent function call.
*/

function requireScript(scriptName) {
  const obsidianPath = `_templater_scripts/${scriptName}`;
  const nodePath = `./${scriptName}`;

  try {
    // Try Obsidian's require first
    if (typeof require !== 'undefined') {
      const script = require(obsidianPath);
      if (script) return script;
    }
    // Fallback to Node.js require for testing
    return require(nodePath);
  } catch (e) {
    console.error(`Templater: requireScript - Failed to load '${scriptName}'`);
    console.error(e);
    return null;
  }
}

module.exports = { requireScript };