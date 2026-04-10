
# Sophia NLU Engine - Home Assistant Integration

Advanced local Natural Language Understanding for Home Assistant.

This integration works together with the **Sophia NLU App** (the backend engine).

## Requirements

- Home Assistant 2025.1 or newer
- The **Sophia NLU App** installed and running (from the custom App repository)
- A valid Sophia NLU license

## Installation

### 1. Install the Sophia NLU App (Backend)

1. Go to **Settings → Apps → App Store**
2. Click the three dots (⋮) in the top right → **Repositories**
3. Add this repository:
   ```
   https://ha-downloader:YOUR_TOKEN_HERE@git.nlu.to/aquila/sophia-ha-app.git
   ```

4. Click **Install** on "Sophia NLU" and start the app.

### 2. Install the Sophia NLU Integration

1. Make sure [HACS](https://hacs.xyz) is installed.
2. Go to **HACS → Integrations**
3. Click the three dots (⋮) → **Custom repositories**
4. Add this repository:
   ```
   https://git.nlu.to/aquila/sophia-ha-integration.git
   ```
   - Category: **Integration**
5. Search for **"Sophia NLU Engine"** and click **Install**.

### 3. Configure the Integration

1. Go to **Settings → Devices & Services**
2. Click **Add Integration** and search for **"Sophia NLU Engine"**
3. Enter your license key when prompted.
4. The integration will connect to the running Sophia NLU App.


## Usage

Once both the App and Integration are installed and configured, you can use Sophia NLU with:
- Assist (voice) pipelines
- Conversation agent
- Custom automations and scripts via the `sophia_nlu.process_text` service

Example service call:
```yaml
service: sophia_nlu.process_text
data:
  text: "Turn on the living room lights"
```

---

## Support & Licensing

- This is a **commercial product**. A valid license key is required.
- License is tied to your Home Assistant instance (`core.uuid`).
- For support, questions, or license issues, please contact us at support@cicero.sh

---

## Links

- [App Repository](https://git.nlu.to/aquila/sophia-ha-app)
- [Issue Tracker](https://git.nlu.to/aquila/sophia-ha-integration/issues)

---

Made by Aquila Labs


