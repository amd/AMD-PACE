# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

"""WebArena accessibility-tree extraction (CDP), vendored verbatim from
WebVoyager (utils_webarena.py + get_webarena_accessibility_tree). This is the
EXACT observation the original WebVoyager uses in text-only mode."""

from typing import Any, TypedDict
import re


class AccessibilityTreeNode(TypedDict):
    nodeId: str
    ignored: bool
    role: dict[str, Any]
    chromeRole: dict[str, Any]
    name: dict[str, Any]
    properties: list[dict[str, Any]]
    childIds: list[str]
    parentId: str
    backendDOMNodeId: str
    frameId: str
    bound: list[float] | None
    union_bound: list[float] | None
    offsetrect_bound: list[float] | None


class BrowserConfig(TypedDict):
    win_top_bound: float
    win_left_bound: float
    win_width: float
    win_height: float
    win_right_bound: float
    win_lower_bound: float
    device_pixel_ratio: float


class BrowserInfo(TypedDict):
    DOMTree: dict[str, Any]
    config: BrowserConfig


IGNORED_ACTREE_PROPERTIES = (
    "focusable",
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
)

AccessibilityTree = list[AccessibilityTreeNode]

IN_VIEWPORT_RATIO_THRESHOLD = 0.6


def fetch_browser_info(
    # page: Page,
    browser,
) -> BrowserInfo:
    # extract domtree
    tree = browser.execute_cdp_cmd(
        "DOMSnapshot.captureSnapshot",
        {
            "computedStyles": [],
            "includeDOMRects": True,
            "includePaintOrder": True,
        },
    )

    # calibrate the bounds, in some cases, the bounds are scaled somehow
    bounds = tree["documents"][0]["layout"]["bounds"]
    b = bounds[0]
    n = b[2] / browser.get_window_size()["width"]
    bounds = [[x / n for x in bound] for bound in bounds]
    tree["documents"][0]["layout"]["bounds"] = bounds

    # extract browser info
    # win_top_bound = page.evaluate("window.pageYOffset")
    # win_left_bound = page.evaluate("window.pageXOffset")
    # win_width = page.evaluate("window.screen.width")
    # win_height = page.evaluate("window.screen.height")

    win_top_bound = browser.execute_script("return window.pageYOffset;")
    win_left_bound = browser.execute_script("return window.pageXOffset;")
    win_width = browser.execute_script("return window.screen.width;")
    win_height = browser.execute_script("return window.screen.height;")
    win_right_bound = win_left_bound + win_width
    win_lower_bound = win_top_bound + win_height
    device_pixel_ratio = browser.execute_script("return window.devicePixelRatio;")
    assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

    config: BrowserConfig = {
        "win_top_bound": win_top_bound,
        "win_left_bound": win_left_bound,
        "win_width": win_width,
        "win_height": win_height,
        "win_right_bound": win_right_bound,
        "win_lower_bound": win_lower_bound,
        "device_pixel_ratio": device_pixel_ratio,
    }

    # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
    info: BrowserInfo = {"DOMTree": tree, "config": config}

    return info


def get_element_in_viewport_ratio(
    elem_left_bound: float,
    elem_top_bound: float,
    width: float,
    height: float,
    config: BrowserConfig,
) -> float:
    elem_right_bound = elem_left_bound + width
    elem_lower_bound = elem_top_bound + height

    win_left_bound = 0
    win_right_bound = config["win_width"]
    win_top_bound = 0
    win_lower_bound = config["win_height"]

    # Compute the overlap in x and y axes
    overlap_width = max(
        0,
        min(elem_right_bound, win_right_bound) - max(elem_left_bound, win_left_bound),
    )
    overlap_height = max(
        0,
        min(elem_lower_bound, win_lower_bound) - max(elem_top_bound, win_top_bound),
    )

    # fraction of the element's area that is visible in the viewport (0..1)
    denom = width * height
    ratio = (overlap_width * overlap_height) / denom if denom else 0.0
    return ratio


def get_bounding_client_rect(browser, backend_node_id: str) -> dict[str, Any]:
    try:
        remote_object = browser.execute_cdp_cmd(
            "DOM.resolveNode", {"backendNodeId": int(backend_node_id)}
        )
        remote_object_id = remote_object["object"]["objectId"]
        response = browser.execute_cdp_cmd(
            "Runtime.callFunctionOn",
            {
                "objectId": remote_object_id,
                "functionDeclaration": """
                    function() {
                        if (this.nodeType == 3) {
                            var range = document.createRange();
                            range.selectNode(this);
                            var rect = range.getBoundingClientRect().toJSON();
                            range.detach();
                            return rect;
                        } else {
                            return this.getBoundingClientRect().toJSON();
                        }
                    }
                """,
                "returnByValue": True,
            },
        )
        return response
    except Exception:
        return {"result": {"subtype": "error"}}


def _boxes_batched(browser):
    """All boxes in ONE CDP call (DOMSnapshot.captureSnapshot), keyed by backendNodeId — the pace
    path. Replaces the naive path's 2 round-trips PER NODE with a single batched call, and matches
    the naive boxes EXACTLY (incl. text nodes, whose box is the union of their text-line boxes,
    mirroring naive's Range.getBoundingClientRect). Viewport-relative CSS px (device-scale = 1).
    """
    snap = browser.execute_cdp_cmd(
        "DOMSnapshot.captureSnapshot", {"computedStyles": []}
    )
    out = {}
    for doc in snap.get("documents", []):
        nodes = doc.get("nodes", {})
        backend = nodes.get("backendNodeId", [])
        ntype = nodes.get("nodeType", [])
        layout = doc.get("layout", {})
        nidx, bounds = layout.get("nodeIndex", []), layout.get("bounds", [])
        sx = doc.get("scrollOffsetX", 0) or 0
        sy = doc.get("scrollOffsetY", 0) or 0
        # union each text node's line boxes -> one box (matches naive's Range on text nodes)
        tb = doc.get("textBoxes", {})
        tb_layout, tb_bounds = tb.get("layoutIndex", []), tb.get("bounds", [])
        text_union = {}
        for j, li in enumerate(tb_layout):
            x, y, w, h = tb_bounds[j]
            if li in text_union:
                x0, y0, x1, y1 = text_union[li]
                text_union[li] = [
                    min(x0, x),
                    min(y0, y),
                    max(x1, x + w),
                    max(y1, y + h),
                ]
            else:
                text_union[li] = [x, y, x + w, y + h]
        for i, ni in enumerate(nidx):
            if ni >= len(backend):
                continue
            is_text = ni < len(ntype) and ntype[ni] == 3
            if is_text and i in text_union:
                x0, y0, x1, y1 = text_union[i]
                box = [x0, y0, x1 - x0, y1 - y0]
            elif i < len(bounds):
                box = list(bounds[i])
            else:
                continue
            out[str(backend[ni])] = [box[0] - sx, box[1] - sy, box[2], box[3]]
    return out


def fetch_page_accessibility_tree(
    info: BrowserInfo,
    browser,
    # client: CDPSession,
    current_viewport_only: bool,
    impl: str = "naive",
) -> AccessibilityTree:
    accessibility_tree: AccessibilityTree = browser.execute_cdp_cmd(
        "Accessibility.getFullAXTree", {}
    )["nodes"]

    # a few nodes are repeated in the accessibility tree
    seen_ids = set()
    _accessibility_tree = []
    for node in accessibility_tree:
        if node["nodeId"] not in seen_ids:
            _accessibility_tree.append(node)
            seen_ids.add(node["nodeId"])
    accessibility_tree = _accessibility_tree

    # pace: one batched box call up front (same boxes, no per-node round-trips)
    batched = _boxes_batched(browser) if impl == "pace" else None

    nodeid_to_cursor = {}
    for cursor, node in enumerate(accessibility_tree):
        nodeid_to_cursor[node["nodeId"]] = cursor
        # usually because the node is not visible etc
        if "backendDOMNodeId" not in node:
            node["union_bound"] = None
            continue
        backend_node_id = str(node["backendDOMNodeId"])
        if node["role"]["value"] == "RootWebArea":
            # always inside the viewport
            node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
        elif batched is not None:
            node["union_bound"] = batched.get(backend_node_id)  # pace: batched lookup
        else:
            response = get_bounding_client_rect(browser, backend_node_id)
            if response.get("result", {}).get("subtype", "") == "error":
                node["union_bound"] = None
            else:
                x = response["result"]["value"]["x"]
                y = response["result"]["value"]["y"]
                width = response["result"]["value"]["width"]
                height = response["result"]["value"]["height"]
                node["union_bound"] = [x, y, width, height]

    # filter nodes that are not in the current viewport
    if current_viewport_only:

        def remove_node_in_graph(node: AccessibilityTreeNode) -> None:
            # update the node information in the accessibility tree
            nodeid = node["nodeId"]
            node_cursor = nodeid_to_cursor[nodeid]
            parent_nodeid = node["parentId"]
            children_nodeids = node["childIds"]
            parent_cursor = nodeid_to_cursor[parent_nodeid]
            # update the children of the parent node
            assert accessibility_tree[parent_cursor].get("parentId", "Root") is not None
            # remove the nodeid from parent's childIds
            index = accessibility_tree[parent_cursor]["childIds"].index(nodeid)
            accessibility_tree[parent_cursor]["childIds"].pop(index)
            # Insert children_nodeids in the same location
            for child_nodeid in children_nodeids:
                accessibility_tree[parent_cursor]["childIds"].insert(
                    index, child_nodeid
                )
                index += 1
            # update children node's parent
            for child_nodeid in children_nodeids:
                child_cursor = nodeid_to_cursor[child_nodeid]
                accessibility_tree[child_cursor]["parentId"] = parent_nodeid
            # mark as removed
            accessibility_tree[node_cursor]["parentId"] = "[REMOVED]"

        config = info["config"]
        for node in accessibility_tree:
            if not node["union_bound"]:
                remove_node_in_graph(node)
                continue

            [x, y, width, height] = node["union_bound"]

            # invisible node
            if width == 0 or height == 0:
                remove_node_in_graph(node)
                continue

            in_viewport_ratio = get_element_in_viewport_ratio(
                elem_left_bound=float(x),
                elem_top_bound=float(y),
                width=float(width),
                height=float(height),
                config=config,
            )

            if in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD:
                remove_node_in_graph(node)

        accessibility_tree = [
            node
            for node in accessibility_tree
            if node.get("parentId", "Root") != "[REMOVED]"
        ]

    return accessibility_tree


def parse_accessibility_tree(
    accessibility_tree: AccessibilityTree,
) -> tuple[str, dict[str, Any]]:
    """Parse the accessibility tree into a string text"""
    node_id_to_idx = {}
    for idx, node in enumerate(accessibility_tree):
        node_id_to_idx[node["nodeId"]] = idx

    obs_nodes_info = {}

    def dfs(idx: int, obs_node_id: str, depth: int) -> str:
        tree_str = ""
        node = accessibility_tree[idx]
        indent = "\t" * depth
        valid_node = True
        try:
            role = node["role"]["value"]
            name = node["name"]["value"]
            node_str = f"[{obs_node_id}] {role} {repr(name)}"
            properties = []
            for property in node.get("properties", []):
                try:
                    if property["name"] in IGNORED_ACTREE_PROPERTIES:
                        continue
                    properties.append(
                        f'{property["name"]}: {property["value"]["value"]}'
                    )
                except KeyError:
                    pass

            if properties:
                node_str += " " + " ".join(properties)

            # check valid
            if not node_str.strip():
                valid_node = False

            # empty generic node
            if not name.strip():
                if not properties:
                    if role in [
                        "generic",
                        "img",
                        "list",
                        "strong",
                        "paragraph",
                        "banner",
                        "navigation",
                        "Section",
                        "LabelText",
                        "Legend",
                        "listitem",
                    ]:
                        valid_node = False
                elif role in ["listitem"]:
                    valid_node = False

            if valid_node:
                tree_str += f"{indent}{node_str}"
                obs_nodes_info[obs_node_id] = {
                    "backend_id": node["backendDOMNodeId"],
                    "union_bound": node["union_bound"],
                    "text": node_str,
                }

        except Exception:
            valid_node = False

        for _, child_node_id in enumerate(node["childIds"]):
            if child_node_id not in node_id_to_idx:
                continue
            # mark this to save some tokens
            child_depth = depth + 1 if valid_node else depth
            child_str = dfs(node_id_to_idx[child_node_id], child_node_id, child_depth)
            if child_str.strip():
                if tree_str.strip():
                    tree_str += "\n"
                tree_str += child_str

        return tree_str

    tree_str = dfs(0, accessibility_tree[0]["nodeId"], 0)
    return tree_str, obs_nodes_info


def clean_accesibility_tree(tree_str: str) -> str:
    """further clean accesibility tree"""
    clean_lines: list[str] = []
    for line in tree_str.split("\n"):
        if "statictext" in line.lower():
            prev_lines = clean_lines[-3:]
            pattern = r"\[\d+\] StaticText '([^']+)'"

            match = re.search(pattern, line)
            if match:
                static_text = match.group(1)
                if all(static_text not in prev_line for prev_line in prev_lines):
                    clean_lines.append(line)
        else:
            clean_lines.append(line)

    return "\n".join(clean_lines)


# ---------------------------------------------------------------------------
# Wrapper (from WebVoyager utils.py) — the exact observation WebVoyager uses in
# text-only mode: full WebArena accessibility tree via Chrome DevTools Protocol.
# Vendored from PhaniAgentCPU/AgentCPU/WebVoyager (WebVoyager/WebArena, Apache-2.0).
# ---------------------------------------------------------------------------
def get_webarena_accessibility_tree(browser, save_file=None, impl="naive"):
    browser_info = fetch_browser_info(browser)
    accessibility_tree = fetch_page_accessibility_tree(
        browser_info, browser, current_viewport_only=True, impl=impl
    )
    content, obs_nodes_info = parse_accessibility_tree(accessibility_tree)
    content = clean_accesibility_tree(content)
    return content, obs_nodes_info
