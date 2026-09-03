# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from html.parser import HTMLParser

from pace_agent.registry import PaceAgentTool, SkipTool, register_tool

_ROLES = {
    "a": "link",
    "button": "button",
    "input": "textbox",
    "span": "text",
    "div": "group",
}


class _Tree(HTMLParser):
    def __init__(self):
        super().__init__()
        self.nodes = []
        self._stack = []

    def handle_starttag(self, tag, attrs):
        self._stack.append([_ROLES.get(tag, tag), ""])

    def handle_data(self, d):
        d = d.strip()
        if d and self._stack:
            self._stack[-1][1] += d

    def handle_endtag(self, tag):
        if self._stack:
            role, name = self._stack.pop()
            if name:
                self.nodes.append({"role": role, "name": name})


@register_tool("dom", "tool_io", io_pattern=True)
class PaceAgentDom(PaceAgentTool):
    def naive(self, html):
        p = _Tree()
        p.feed(html)
        return p.nodes


@register_tool("file_read", "tool_io", io_pattern=True)
class PaceAgentFileRead(PaceAgentTool):
    def naive(self, path):
        import os

        if not os.path.exists(path):
            raise SkipTool(f"missing {path}")
        with open(path, "rb") as f:
            return f.read()
