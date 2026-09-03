# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

import glob
import os
import shutil
import sys

# pace observe: ONE execute_script that also STAMPS each element with a stable data-pace-agent-idx.
# Actions later resolve elements by that id (not by re-querying position), so a DOM mutation
# during the LLM call can't make us act on the wrong element (the Set-of-Marks staleness bug).
_OBSERVE_JS = """
let els=[...document.querySelectorAll('a,button,input,select,textarea,[role],[onclick]')];
let out=[];
els.forEach((e,i)=>{
  e.setAttribute('data-pace-agent-idx', i);
  let name=(e.innerText||e.value||e.getAttribute('aria-label')||'').trim();
  out.push({index:i, role:e.getAttribute('role')||e.tagName.toLowerCase(), name:name});
});
return out.filter(n=>n.name);
"""

_STAMP_JS = """
[...document.querySelectorAll('a,button,input,select,textarea,[role],[onclick]')]
  .forEach((e,i)=>e.setAttribute('data-pace-agent-idx', i));
"""


class BrowserUnavailable(RuntimeError):
    pass


def _chrome_binary():
    for c in (
        os.environ.get("PACE_AGENT_CHROME_BIN"),
        # prefer FULL Chrome (supports --user-data-dir + full rendering) over headless-shell
        os.path.expanduser("~/.pace_agent_sel/chrome-linux64/chrome"),
        *glob.glob(
            os.path.expanduser(
                "~/.pace_agent_sel/chrome-headless-shell-*/chrome-headless-shell"
            )
        ),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("google-chrome"),
    ):
        if c and os.path.exists(c):
            return c
    return None


def _is_headless_shell(binary):
    return "headless-shell" in (binary or "")


def _chromedriver():
    for d in (
        os.environ.get("PACE_AGENT_CHROMEDRIVER"),
        *glob.glob(os.path.expanduser("~/.pace_agent_sel/chromedriver-*/chromedriver")),
        shutil.which("chromedriver"),
    ):
        if d and os.path.exists(d):
            return d
    return None


def _browser_flags(headless):
    flags = [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-blink-features=AutomationControlled",
        "--force-device-scale-factor=1",
        "--window-size=1280,900",
        "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36",
    ]
    if headless:
        flags.insert(0, "--headless=new")
    return flags


class SeleniumBrowser:
    def __init__(self, headless=True):
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
        except ImportError as e:
            raise BrowserUnavailable(
                f"selenium not installed: {e} (see docs/browser-setup.md)"
            )
        binary = _chrome_binary()
        if not binary:
            raise BrowserUnavailable(
                "no chrome/chromium binary found (see docs/browser-setup.md)"
            )
        opts = Options()
        # Original WebVoyager driver_config, faithfully: headless + anti-detection flags. The
        # --user-data-dir persistent profile needs FULL Chrome (headless-shell rejects it).
        flags = _browser_flags(headless)
        # full Chrome only: a UNIQUE profile per instance (a shared --user-data-dir is single-locked,
        # so concurrent browsers would collide). Fresh profile still defeats detection with these flags.
        self._profile = None
        if not _is_headless_shell(binary):
            import tempfile

            self._profile = tempfile.mkdtemp(prefix="pace_agent_chrome_")
            flags.append("--user-data-dir=" + self._profile)
        for f in flags:
            opts.add_argument(f)
        opts.binary_location = binary
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = (
            os.path.join(sys.prefix, "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        )
        drv = _chromedriver()
        service = Service(executable_path=drv, env=env) if drv else Service(env=env)
        try:
            self.driver = webdriver.Chrome(options=opts, service=service)
        except Exception as e:
            raise BrowserUnavailable(
                f"could not start chrome: {str(e)[:160]} (see docs/browser-setup.md)"
            )
        from selenium.webdriver.common.by import By

        self._By = By

    def navigate(self, url):
        self.driver.get(url)
        self.await_ready()

    def await_ready(self, timeout=15):
        import time

        end = time.time() + timeout
        while time.time() < end:
            try:
                if (
                    self.driver.execute_script("return document.readyState")
                    == "complete"
                ):
                    return
            except Exception:
                pass
            time.sleep(0.1)

    def observe(self, impl):
        return self.observe_pace() if impl == "pace" else self.observe_naive()

    def screenshot_b64(self):
        try:
            return self.driver.get_screenshot_as_base64()
        except Exception:
            return ""

    # label by the SAME data-pace-agent-idx the model is given in the text list and that actions resolve by
    _SOM_JS = """
    document.querySelectorAll('[data-pace-agent-idx]').forEach(e=>{
      let r=e.getBoundingClientRect(); if(r.width&&r.height){
        let m=document.createElement('div'); m.textContent=e.getAttribute('data-pace-agent-idx');
        m.style.cssText='position:absolute;z-index:99999;background:red;color:#fff;font:10px sans-serif;'
          +'padding:0 2px;left:'+(r.left+scrollX)+'px;top:'+(r.top+scrollY)+'px';
        m.className='pace-agent-som'; document.body.appendChild(m);}});
    """

    def som_screenshot(self):
        """Set-of-Marks screenshot: overlay numbered labels (index = same order as observe/act),
        capture, remove overlays. Returns base64 PNG (or '' on failure). Vision-mode only.
        """
        try:
            self.driver.execute_script(self._SOM_JS)
            shot = self.screenshot_b64()
            self.driver.execute_script(
                "document.querySelectorAll('.pace-agent-som').forEach(e=>e.remove());"
            )
            return shot
        except Exception:
            return ""

    # Set-of-Marks from CALLER-supplied boxes ([[label,[x,y,w,h]],...] in viewport CSS px,
    # device-scale=1) so the drawn numbers match EXACTLY the [labels] in the accessibility tree
    # the model also reads.
    _SOM_BOX_JS = """
    const items = arguments[0];
    for (const it of items) {
      const lbl = it[0], b = it[1];
      const box = document.createElement('div'); box.className='pace-agent-som';
      box.style.cssText='position:fixed;z-index:2147483646;border:2px solid #ff0000;'
        +'pointer-events:none;left:'+b[0]+'px;top:'+b[1]+'px;width:'+b[2]+'px;height:'+b[3]+'px';
      document.body.appendChild(box);
      const tag = document.createElement('div'); tag.className='pace-agent-som'; tag.textContent=lbl;
      tag.style.cssText='position:fixed;z-index:2147483647;background:#ff0000;color:#fff;'
        +'font:bold 11px/1.1 sans-serif;padding:0 2px;pointer-events:none;'
        +'left:'+b[0]+'px;top:'+Math.max(0,b[1]-12)+'px';
      document.body.appendChild(tag);
    }
    """

    def som_from_boxes(self, boxes):
        """Draw numbered Set-of-Marks labels from `boxes` (dict {label:[x,y,w,h]} or list of
        [label, box]), screenshot, remove overlays. Returns base64 PNG (or '' on failure).
        """
        try:
            items = boxes.items() if isinstance(boxes, dict) else boxes
            payload = [
                [str(lbl), [b[0], b[1], b[2], b[3]]]
                for lbl, b in items
                if b and len(b) == 4 and b[2] and b[3]
            ]
            self.driver.execute_script(self._SOM_BOX_JS, payload)
            shot = self.screenshot_b64()
            self.driver.execute_script(
                "document.querySelectorAll('.pace-agent-som').forEach(e=>e.remove());"
            )
            return shot
        except Exception:
            return ""

    def observe_vision(self):
        """Set-of-Marks overlay + screenshot + element list (index-aligned with observe/act)."""
        return {
            "elements": self.observe_pace(),
            "screenshot_b64": self.som_screenshot(),
        }

    def _ele(self, index):
        """Resolve the element the model chose by its STABLE data-pace-agent-idx (set during observe),
        so DOM drift between observe and act can't point us at the wrong element."""
        try:
            return self.driver.find_element(
                self._By.CSS_SELECTOR, f'[data-pace-agent-idx="{index}"]'
            )
        except Exception:
            return None

    def click(self, index):
        el = self._ele(index)
        if el is None:
            return False
        try:
            # keep navigation in the same tab (faithful to WebVoyager's exec_action_click)
            self.driver.execute_script(
                "arguments[0].setAttribute('target','_self');", el
            )
            try:
                el.click()
            except Exception:
                self.driver.execute_script("arguments[0].click();", el)
            self.await_ready()
            return True
        except Exception:
            return False

    def type(self, index, text, submit=False):
        el = self._ele(index)
        if el is None:
            return False
        try:
            el.clear()
            el.send_keys(text)
            if submit:
                from selenium.webdriver.common.keys import Keys

                el.send_keys(Keys.RETURN)
                self.await_ready()
            return True
        except Exception:
            return False

    def type_warn(self, index, text):
        """Type + Enter; returns a warning string if the target isn't a real textbox (else '')."""
        el = self._ele(index)
        if el is None:
            return f"note: element [{index}] does not exist."
        try:
            tag = (el.tag_name or "").lower()
            typ = (el.get_attribute("type") or "").lower()
        except Exception:
            tag, typ = "", ""
        warn = ""
        if tag not in ("input", "textarea") or (
            tag == "input"
            and typ not in ("", "text", "search", "email", "tel", "url", "password")
        ):
            warn = f"note: element [{index}] may not be a textbox (tag <{tag}> type '{typ}') — try the search box, or click a search button first."
        self.type(index, text, submit=True)
        return warn

    def scroll(self, target, direction):
        h = self.driver.execute_script("return window.innerHeight") or 800
        dy = (2 * h // 3) * (1 if direction == "down" else -1)
        if str(target).upper() == "WINDOW":
            self.driver.execute_script(f"window.scrollBy(0,{dy});")
        else:
            el = self._ele(int(target))
            if el is not None:
                self.driver.execute_script(
                    "arguments[0].scrollIntoView();arguments[0].scrollBy(0,arguments[1]);",
                    el,
                    dy,
                )
        import time

        time.sleep(0.3)

    def go_back(self):
        self.driver.back()
        self.await_ready()

    def read_text(self, limit=4000):
        return (self.driver.execute_script("return document.body.innerText") or "")[
            :limit
        ]

    def observe_naive(self):
        self.driver.execute_script(_STAMP_JS)
        els = self.driver.find_elements(
            self._By.CSS_SELECTOR, "a,button,input,select,textarea,[role],[onclick]"
        )
        out = []
        for i, e in enumerate(els):
            try:
                role = e.get_attribute("role") or e.tag_name
                name = (
                    e.text
                    or e.get_attribute("value")
                    or e.get_attribute("aria-label")
                    or ""
                ).strip()
            except Exception:
                continue  # element went stale on a live page — per-element reads are fragile
            if name:
                out.append({"index": i, "role": role, "name": name})
        return out

    def observe_pace(self):
        return self.driver.execute_script(_OBSERVE_JS)

    # ---- WebVoyager text-only pipeline: full WebArena accessibility tree (CDP) + ----
    # ---- coordinate-resolved actions (verbatim port of exec_action_click/type/scroll) ----
    def snapshot_mhtml(self):
        """Self-contained snapshot of the fully-rendered page (DOM + CSS + images inlined) via
        CDP Page.captureSnapshot. Reopening it offline reproduces the page exactly — so the
        replay A/B keeps the FULL page (representative) yet stays deterministic (no network).
        """
        try:
            return self.driver.execute_cdp_cmd(
                "Page.captureSnapshot", {"format": "mhtml"}
            ).get("data", "")
        except Exception:
            return ""

    def observe_ax(self, impl="naive"):
        """WebVoyager text-only observation: full WebArena accessibility tree via CDP.
        naive = per-node bounding-rect round-trips (faithful port); pace = one batched
        DOMSnapshot box call (same output). Returns (tree_text, obs_info)."""
        from pace_agent.bench.workloads.webarena_ax import (
            get_webarena_accessibility_tree,
        )

        return get_webarena_accessibility_tree(self.driver, impl=impl)

    def _ele_from_box(self, ub):
        try:
            cx, cy = ub[0] + ub[2] // 2, ub[1] + ub[3] // 2
            return self.driver.execute_script(
                "return document.elementFromPoint(arguments[0], arguments[1]);", cx, cy
            )
        except Exception:
            return None

    def ax_click(self, ub):
        import time

        el = self._ele_from_box(ub)
        if el is None:
            return False
        try:
            self.driver.execute_script(
                "arguments[0].setAttribute('target', '_self')", el
            )
            el.click()
            time.sleep(3)
            self.await_ready()
            return True
        except Exception:
            return False

    def ax_type(self, ub, content):
        import time
        from selenium.webdriver.common.action_chains import ActionChains
        from selenium.webdriver.common.keys import Keys

        el = self._ele_from_box(ub)
        if el is None:
            return "note: could not resolve the element to type into."
        warn = ""
        try:
            tag = (el.tag_name or "").lower()
            typ = (el.get_attribute("type") or "").lower()
            if tag not in ("input", "textarea") or (
                tag == "input"
                and typ not in ("text", "search", "password", "email", "tel")
            ):
                warn = f"note: element may not be a textbox (tag <{tag}> type {typ})."
        except Exception:
            pass
        try:
            el.clear()
            el.send_keys(Keys.CONTROL + "a")
            el.send_keys(" ")
            el.send_keys(Keys.BACKSPACE)
        except Exception:
            pass
        actions = ActionChains(self.driver)
        actions.click(el).perform()
        actions.pause(1)
        actions.send_keys(content)
        actions.pause(2)
        actions.send_keys(Keys.ENTER)
        actions.perform()
        time.sleep(4)
        self.await_ready()
        return warn

    def ax_scroll(self, target, direction, ub=None):
        import time
        from selenium.webdriver.common.action_chains import ActionChains
        from selenium.webdriver.common.keys import Keys

        h = self.driver.execute_script("return window.innerHeight") or 900
        if str(target).upper() == "WINDOW":
            dy = (2 * h // 3) * (1 if direction == "down" else -1)
            self.driver.execute_script(f"window.scrollBy(0,{dy});")
        elif ub is not None:
            el = self._ele_from_box(ub)
            if el is not None:
                self.driver.execute_script("arguments[0].focus();", el)
                key = Keys.ARROW_DOWN if direction == "down" else Keys.ARROW_UP
                ActionChains(self.driver).key_down(Keys.ALT).send_keys(key).key_up(
                    Keys.ALT
                ).perform()
        time.sleep(1)

    def quit(self):
        try:
            self.driver.quit()
        except Exception:
            pass
        if getattr(self, "_profile", None):
            import shutil

            shutil.rmtree(self._profile, ignore_errors=True)
