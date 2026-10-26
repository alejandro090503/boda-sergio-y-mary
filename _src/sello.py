"""Cambia L&N -> S&M en el sello del sobre de boda-nao-leo (imagen y video)."""
import cv2, numpy as np, sys, os, glob
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT = r'C:/Users/aleja/xv-giselle/_src/GreatVibes.ttf'
AMP = r'C:/Windows/Fonts/georgiai.ttf'
INK = (44, 62, 34)  # verde oscuro del grabado (RGB)

def glyph(R):
    """Monograma diagonal S & M como el original (L arriba-izq, N abajo-der)."""
    S = int(R * 2.2)
    can = Image.new('L', (S, S), 0); d = ImageDraw.Draw(can)
    big = ImageFont.truetype(FONT, int(R * 1.05)); amp = ImageFont.truetype(AMP, int(R * 0.5))
    def put(t, f, cx, cy):
        b = d.textbbox((0, 0), t, font=f); d.text((cx - (b[0]+b[2])/2, cy - (b[1]+b[3])/2), t, font=f, fill=255)
    put('S', big, S*0.28, S*0.36); put('&', amp, S*0.50, S*0.50); put('M', big, S*0.74, S*0.66)
    m = can.crop(can.getbbox())
    w = int(R * 1.08); h = int(m.height * w / m.width)
    if h > R * 1.08: h = int(R * 1.08); w = int(m.width * h / m.height)
    return m.resize((w, h), Image.LANCZOS).filter(ImageFilter.MaxFilter(3))

def stamp(img_bgr, cx, cy, R, alpha=1.0):
    """Borra las letras viejas e imprime las nuevas. img_bgr se modifica."""
    cx, cy, R = float(cx), float(cy), float(R)
    H, W = img_bgr.shape[:2]
    x0, y0 = int(cx - R), int(cy - R); x1, y1 = int(cx + R), int(cy + R)
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H or R < 20:
        # recorte parcial: trabajar en la parte visible
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
        if x1 - x0 < 20 or y1 - y0 < 20: return img_bgr
    crop = img_bgr[y0:y1, x0:x1].copy()
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - cx, yy - cy) / R
    inner = rr < 0.66
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(float)
    blur = cv2.GaussianBlur(gray, (0, 0), R * 0.2)
    dark = (gray < blur - 12) | (gray < 150)
    mask = (inner & dark).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), 1)
    mask[~inner] = 0
    clean = cv2.inpaint(crop, mask, max(3, int(R * 0.06)), cv2.INPAINT_TELEA)
    # letras nuevas
    g = glyph(R)
    pil = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)).convert('RGBA')
    gx = int(cx - x0 - g.width / 2); gy = int(cy - y0 - g.height / 2)
    a = np.array(g).astype(float) / 255 * alpha
    shadow = Image.fromarray((a * 110).astype('uint8')).filter(ImageFilter.GaussianBlur(R * 0.012 + .5))
    light = Image.fromarray((a * 90).astype('uint8'))
    lay = Image.new('RGBA', pil.size, (0, 0, 0, 0))
    lay.paste(Image.new('RGBA', g.size, (255, 248, 220, 255)), (gx - 1, gy - 1), light)
    lay.paste(Image.new('RGBA', g.size, (20, 26, 12, 255)), (gx + 2, gy + 2), shadow)
    lay.paste(Image.new('RGBA', g.size, INK + (255,)), (gx, gy), Image.fromarray((a * 255).astype('uint8')))
    pil = Image.alpha_composite(pil, lay)
    out = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)
    img_bgr[y0:y1, x0:x1] = out
    return img_bgr

def find(gray, prev=None):
    g = cv2.medianBlur(gray, 5)
    if prev is None:
        c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1.2, 200, param1=120, param2=60, minRadius=90, maxRadius=200)
        return tuple(c[0][0]) if c is not None else None
    px, py, pr = prev
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1.2, 50, param1=120, param2=40, minRadius=int(pr * .85), maxRadius=int(pr * 1.2))
    if c is None: return None
    best = min(c[0], key=lambda k: (k[0] - px) ** 2 + (k[1] - py) ** 2 + 4 * (k[2] - pr) ** 2)
    if np.hypot(best[0] - px, best[1] - py) > pr * .6: return None
    return tuple(best)

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'img':
        im = cv2.imread('orig-sobre.png')
        cx, cy, R = find(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        stamp(im, cx, cy, R)
        cv2.imwrite('sobre-sm.png', im)
        c = im[int(cy - R - 20):int(cy + R + 20), int(cx - R - 20):int(cx + R + 20)]
        cv2.imwrite('crop-sm.png', c)
        print(cx, cy, R)
    else:
        frames = sorted(glob.glob('fr_in/*.png'))
        track = []; prev = None
        for f in frames:
            im = cv2.imread(f)
            r = find(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), prev)
            track.append(r); prev = r if r else prev
        np.save('track.npy', np.array([t if t else (np.nan,) * 3 for t in track], float))
        print('detected', sum(1 for t in track if t), 'of', len(track))

def run_video():
    frames = sorted(glob.glob('fr_in/*.png'))
    det = []
    prev = None
    for f in frames:
        g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        r = find(g, prev) if prev else None
        if r is None:
            gb = cv2.medianBlur(g, 5)
            c = cv2.HoughCircles(gb, cv2.HOUGH_GRADIENT, 1.2, 300, param1=120, param2=70, minRadius=80, maxRadius=140)
            r = tuple(c[0][0]) if c is not None else None
        det.append(r); prev = r
    t = np.array([d if d else (np.nan,) * 3 for d in det], float)
    # mediana móvil sobre los válidos
    sm = t.copy()
    for i in range(len(t)):
        w = t[max(0, i - 3):i + 4]; w = w[~np.isnan(w[:, 0])]
        if len(w) >= 3 and not np.isnan(t[i, 0]): sm[i] = np.median(w, 0)
    ref = None
    for i, f in enumerate(frames):
        im = cv2.imread(f)
        cx, cy, R = sm[i]
        if not np.isnan(cx):
            yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
            disc = (np.hypot(xx - cx, yy - cy) < R * .55)
            sd = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[disc].std()
            if ref is None: ref = sd
            if sd > 4:
                stamp(im, cx, cy, R, alpha=min(1.0, sd / ref))
        cv2.imwrite(f.replace('fr_in', 'fr_out'), im)
    np.save('track.npy', sm)
    print('stamped', int((~np.isnan(sm[:, 0])).sum()), 'of', len(frames))

if __name__ == '__main__' and sys.argv[1] == 'vid2':
    run_video()
