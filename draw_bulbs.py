#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成灯泡亮/灭两张 GIF：images/pic_bulboff.gif、images/pic_bulbon.gif"""
import os
import numpy as np
from PIL import Image, ImageDraw

W, H = 400, 720
CX, CY, R = 200, 215, 140                                  # 玻璃球圆心/半径
NECK = [(150, 340), (250, 340), (248, 452), (152, 452)]    # 玻璃颈
BASE_BOX = [146, 420, 254, 660]                            # 螺口灯头
TIP_BOX = [170, 654, 230, 702]                             # 底部触点


def np_to_img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


def vertical_grad(y0, y1, c_top, c_bottom):
    t = np.clip((np.arange(H, dtype=np.float32) - y0) / float(y1 - y0), 0.0, 1.0)
    c0 = np.array(c_top, dtype=np.float32)
    c1 = np.array(c_bottom, dtype=np.float32)
    col = c0[None, :] + (c1 - c0)[None, :] * t[:, None]     # (H, 3)
    return np.repeat(col[:, None, :], W, axis=1)            # (H, W, 3)


def radial_grad(center, radius, stops):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.clip(np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / radius, 0.0, 1.0)
    out = np.zeros((H, W, 3), np.float32)
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        m = (d >= t0) & (d <= t1)
        tt = np.clip((d - t0) / max(t1 - t0, 1e-6), 0.0, 1.0)
        seg = (np.array(c0, np.float32)[None, None, :]
               + (np.array(c1, np.float32) - np.array(c0, np.float32))[None, None, :] * tt[..., None])
        out = np.where(m[..., None], seg, out)
    return out


def add_glow(img, center, radius, color, max_alpha, power=2.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / radius
    a = np.clip(1.0 - d, 0.0, 1.0) ** power * max_alpha
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
    rgba[..., 3] = (a * 255).astype(np.uint8)
    img.alpha_composite(Image.fromarray(rgba, "RGBA"))


def glass_mask():
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    d.polygon(NECK, fill=255)
    d.ellipse([CX - R, CY - R, CX + R, CY + R], fill=255)
    return m


def build(on):
    img = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    # 光晕（仅点亮状态）
    if on:
        add_glow(img, (CX, CY), 305, (255, 205, 40), 0.55, 2.0)
        add_glow(img, (CX, CY), 195, (255, 232, 110), 0.85, 1.8)

    # 玻璃渐变
    if on:
        grad = radial_grad((CX, CY), R, [
            (0.00, (255, 250, 196)),
            (0.55, (255, 235, 120)),
            (1.00, (243, 165, 28)),
        ])
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        hl = np.clip(np.sqrt((xx - 150) ** 2 + (yy - 150) ** 2) / 190, 0.0, 1.0)
        grad = grad + (255.0 - grad) * (((1.0 - hl) ** 2 * 0.35)[..., None])
    else:
        grad = vertical_grad(70, 350, (255, 255, 255), (214, 214, 214))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        shade = (np.clip((xx - (CX + 30)) / 130.0, 0.0, 1.0) ** 1.5 * 0.10)
        grad = grad * (1.0 - shade[..., None])

    img.paste(np_to_img(grad), (0, 0), glass_mask())

    # 玻璃轮廓
    d = ImageDraw.Draw(img)
    outline = (196, 110, 0) if on else (140, 140, 140)
    d.ellipse([CX - R, CY - R, CX + R, CY + R], outline=outline, width=4)
    d.line([(151, 360), (153, 452)], fill=outline, width=4)
    d.line([(249, 360), (247, 452)], fill=outline, width=4)

    # 灯丝
    if on:
        add_glow(img, (CX, 300), 72, (255, 255, 220), 0.55, 2.0)
        fil, coil = (255, 249, 196), (255, 255, 255)
    else:
        fil, coil = (168, 168, 168), (148, 148, 148)
    d = ImageDraw.Draw(img)
    d.rectangle([194, 330, 206, 445], fill=fil)
    d.ellipse([186, 292, 214, 330], fill=coil)

    # 灯头（金属）
    if on:
        base_grad = vertical_grad(420, 660, (233, 196, 158), (176, 116, 58))
        base_out, groove, tip_fill = (122, 74, 26), (146, 88, 32), (110, 66, 24)
    else:
        base_grad = vertical_grad(420, 660, (232, 232, 232), (152, 152, 152))
        base_out, groove, tip_fill = (110, 110, 110), (128, 128, 128), (92, 92, 92)

    bm = Image.new("L", (W, H), 0)
    bd = ImageDraw.Draw(bm)
    bd.rounded_rectangle(BASE_BOX, radius=16, fill=255)
    bd.rounded_rectangle(TIP_BOX, radius=10, fill=255)
    img.paste(np_to_img(base_grad), (0, 0), bm)

    d = ImageDraw.Draw(img)
    d.rounded_rectangle(BASE_BOX, radius=16, outline=base_out, width=4)
    for y in (452, 486, 520, 554, 588, 622):
        d.line([(148, y), (252, y)], fill=groove, width=4)
    d.rounded_rectangle(TIP_BOX, radius=10, fill=tip_fill)

    return img.convert("RGB")


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    os.makedirs(out_dir, exist_ok=True)
    build(False).convert("P", palette=Image.ADAPTIVE, colors=256).save(
        os.path.join(out_dir, "pic_bulboff.gif"))
    build(True).convert("P", palette=Image.ADAPTIVE, colors=256).save(
        os.path.join(out_dir, "pic_bulbon.gif"))
    print("saved:", os.listdir(out_dir))
