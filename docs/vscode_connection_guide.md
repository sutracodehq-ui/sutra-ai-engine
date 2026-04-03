# Connecting Sutra-Coder to VS Code

This guide explains how to connect your self-hosted **Sutra-Coder** model to your local VS Code environment using popular AI extensions.

## Core Connection Settings

Your model is served by **Ollama**. There are two ways to connect depending on your environment:

1. **Native Host (Mac App):** Use `http://localhost:11434`
2. **Docker Network:** Use `http://sutra-ai-ollama:11434` (If your VS Code is running inside a Dev Container).

---

## 1. Connecting via [Continue](https://continue.dev/) (Recommended)

Continue is the standard for open-source local LLM integration in VS Code.

### Step 1: Install Continue
Install the **Continue** extension from the VS Code Marketplace.

### Step 2: Configure `config.json`
1. Click the **Continue** icon in the sidebar.
2. Click the gear icon (Settings) to open `config.json`.
3. Add the following entry to the `models` array:

```json
{
  "title": "Sutra-Coder (Local)",
  "model": "qwen2.5-coder-7b-sutra-tuned",
  "provider": "ollama",
  "baseUrl": "http://localhost:11434"
}
```

### Step 3: Use for Autocomplete
To use it for autocomplete, set it in the `tabAutocompleteModel` section:

```json
"tabAutocompleteModel": {
  "title": "Sutra-Coder (Autocomplete)",
  "model": "qwen2.5-coder-7b-sutra-tuned",
  "provider": "ollama"
}
```

---

## 2. Connecting via [Cline](https://cline.bot/) / Roo Code

Cline and Roo Code are excellent for autonomous coding tasks.

### Step 1: Select Provider
In the Cline sidebar, open the settings (gear icon).

### Step 2: Configure Ollama
1. **API Provider:** Select `Ollama`.
2. **Base URL:** `http://localhost:11434` (or `http://127.0.0.1:11434`).
3. **Model ID:** Enter `qwen2.5-coder-7b-sutra-tuned`.

---

## Troubleshooting

### Connection Refused?
- **Host Binding:** Open your Ollama settings and ensure `OLLAMA_HOST` is set to `0.0.0.0` if you are trying to reach it across different network bridges.
- **Model not found:** Run `ollama list` in your terminal to verify `qwen2.5-coder-7b-sutra-tuned` exists.

### Performance Lag?
If your Mac starts lagging, try the **1.5b** or **3b** variants of Qwen2.5-Coder for autocomplete and keep the **7b** version for Chat/Edit only.
