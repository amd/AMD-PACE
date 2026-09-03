# Browser setup (WebVoyager agent)

The `webvoyager` example drives a real Chrome via Selenium. You need two things: the `selenium`
Python package (shipped as a runtime dependency) and a Chrome-for-Testing / Chromium binary plus
a matching `chromedriver`.

Verify at any time with `pace-agent check` — it reports the `selenium` and `chrome`
preflight lines.

## 1. Selenium

Installed with the package. If it is missing:

```bash
pip install selenium
```

## 2. Chrome + chromedriver

PACE Agent resolves a browser binary in this order (first existing match wins):

1. `$PACE_AGENT_CHROME_BIN`
2. `~/.pace_agent_sel/chrome-linux64/chrome` (full Chrome for Testing — preferred)
3. `~/.pace_agent_sel/chrome-headless-shell-*/chrome-headless-shell`
4. `chromium`, `chromium-browser`, or `google-chrome` on `PATH`

and a driver in this order:

1. `$PACE_AGENT_CHROMEDRIVER`
2. `~/.pace_agent_sel/chromedriver-*/chromedriver`
3. `chromedriver` on `PATH`

### Option A — Google Chrome (.deb, pulls all system deps)

On recent Ubuntu the `chromium` apt package is snap-only (no apt candidate), so the
simplest rootful option is Google Chrome — apt installs it together with its shared
libraries (libnss3, libatk, ...) and puts `google-chrome` on `PATH`:

```bash
wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt-get install -y ./google-chrome-stable_current_amd64.deb
```

`chromedriver` is then fetched automatically by Selenium Manager (bundled with
selenium >= 4.6) — no separate driver install needed.

### Option B — Chrome for Testing (pinned, no root)

Download a **matching** Chrome + chromedriver pair for `linux64` and unpack them
under `~/.pace_agent_sel/`:

```bash
mkdir -p ~/.pace_agent_sel && cd ~/.pace_agent_sel
json=https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json
for kind in chrome chromedriver; do
  url=$(curl -s "$json" | python3 -c "import sys,json;d=json.load(sys.stdin);print(next(x['url'] for x in d['channels']['Stable']['downloads']['$kind'] if x['platform']=='linux64'))")
  curl -sSL "$url" -o "$kind.zip" && unzip -q -o "$kind.zip" && rm "$kind.zip"
done
chmod +x chrome-linux64/chrome chromedriver-linux64/chromedriver
```

Resulting layout (exactly what PACE Agent looks for):

```
~/.pace_agent_sel/chrome-linux64/chrome
~/.pace_agent_sel/chromedriver-linux64/chromedriver
```

Chrome for Testing does **not** pull system libraries — on a bare server also install
its runtime deps (e.g. `sudo apt-get install -y libnss3 libatk-bridge2.0-0 libgtk-3-0
libasound2 libgbm1`). Override the locations with `PACE_AGENT_CHROME_BIN` /
`PACE_AGENT_CHROMEDRIVER` if you keep them elsewhere.

## 3. Verify

```bash
pace-agent check            # expect [ok] selenium and [ok] chrome
pace-agent profile pace_agent.examples.webvoyager --config packages/pace_agent/configs/pace_agent_wv_vision.json --out run/
```
