"""Image overlay helpers.

Adds a timestamp banner to a frame WITHOUT covering any of the original video
content. The default behaviour extends the canvas with a small black strip
below (or above) the frame and writes the timestamp inside that strip, so
every pixel of the original video is preserved.

Modes:
    - "extend-bottom" (default): add a strip below the image
    - "extend-top": add a strip above the image
    - "overlay-topleft" etc.: legacy in-frame badge (may occlude content)
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in (
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "arial.ttf",
        "Arial.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _measure(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int, int]:
    """Return (text_width, text_height, baseline_y_offset)."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1], -bbox[1]
    except Exception:
        w, h = draw.textsize(text, font=font)  # type: ignore[attr-defined]
        return w, h, 0


def _extend_canvas(
    img: Image.Image, text: str, *, position: str, padding: int, font_scale: float
) -> Image.Image:
    """Add a strip to the top or bottom of the image and render text in it.
    Original pixels are never modified."""
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    font_size = max(10, int(round(max(w, h) * font_scale)))
    font = _load_font(font_size)

    tmp = Image.new("RGB", (10, 10))
    draw_tmp = ImageDraw.Draw(tmp)
    tw, th, y_offset = _measure(draw_tmp, text, font)

    strip_h = th + 2 * padding
    new_w = w
    new_h = h + strip_h

    canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    if position == "extend-top":
        canvas.paste(img, (0, strip_h))
        strip_y0 = 0
    else:  # extend-bottom
        canvas.paste(img, (0, 0))
        strip_y0 = h

    draw = ImageDraw.Draw(canvas)
    x = padding
    y = strip_y0 + padding + y_offset
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return canvas


def _overlay_corner(
    img: Image.Image, text: str, *, position: str, padding: int, font_scale: float
) -> Image.Image:
    """Legacy behavior: draw a badge inside the image (may occlude content)."""
    out = img.copy()
    if out.mode != "RGB":
        out = out.convert("RGB")

    w, h = out.size
    font_size = max(10, int(round(max(w, h) * font_scale)))
    font = _load_font(font_size)
    draw = ImageDraw.Draw(out)
    tw, th, y_offset = _measure(draw, text, font)

    box_w = tw + 2 * padding
    box_h = th + 2 * padding
    if position == "overlay-topright":
        x0, y0 = w - box_w, 0
    elif position == "overlay-bottomleft":
        x0, y0 = 0, h - box_h
    elif position == "overlay-bottomright":
        x0, y0 = w - box_w, h - box_h
    else:
        x0, y0 = 0, 0

    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(0, 0, 0))
    draw.text((x0 + padding, y0 + padding + y_offset), text, fill=(255, 255, 255), font=font)
    return out


def annotate_frame(
    img: Image.Image,
    text: str,
    *,
    position: str = "extend-bottom",
    padding: int = 6,
    font_scale: float = 0.028,
) -> Image.Image:
    """Return an annotated copy of `img`.

    Default (`extend-bottom`) never occludes original pixels — it adds a small
    black strip below the frame and writes the timestamp text into that strip.

    Set `position` to one of:
      - "extend-bottom" (default, non-occluding)
      - "extend-top"    (non-occluding)
      - "overlay-topleft", "overlay-topright",
        "overlay-bottomleft", "overlay-bottomright"  (in-frame badge)
    """
    if not text:
        return img

    if position.startswith("extend-"):
        return _extend_canvas(img, text, position=position, padding=padding, font_scale=font_scale)
    return _overlay_corner(img, text, position=position, padding=padding, font_scale=font_scale)
