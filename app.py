"""
Hang It! Poster Generator - API pentru Shopify
Backend Flask optimizat pentru Railway deployment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from PIL import Image, ImageDraw, ImageFont
from colorthief import ColorThief
import numpy as np
import colorsys
import requests
import re
import os
import io
import base64
import tempfile
from pathlib import Path

app = Flask(__name__)
CORS(app, origins=["*"])

# Config
CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "62068285c95d49e5a2c394c485e17bd1")
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "5d6813b8396d479e88ce5dbbf18c4442")
FONTS_FOLDER = Path(__file__).parent / 'fonts'

def remove_diacritics(text):
    for d, r in {'ă':'a','Ă':'A','â':'a','Â':'A','î':'i','Î':'I','ș':'s','Ș':'S','ț':'t','Ț':'T'}.items():
        text = text.replace(d, r)
    return text

def is_black_and_white(img):
    arr = np.array(img.resize((100, 100))) / 255.0
    hsv = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            hsv[i, j] = colorsys.rgb_to_hsv(*arr[i, j])
    if np.sum(hsv[:, :, 1] > 0.5) > 20:
        return False
    return bool(hsv[:, :, 1].mean() < 0.08)

def get_bw_background(img):
    arr = np.array(img.resize((100, 100)))
    lum = int(arr.mean())
    if lum >= 180: return (255, 255, 255)
    if lum <= 100: return (0, 0, 0)
    return (max(110, min(170, lum)),) * 3

def get_accent_color(img):
    arr = np.array(img.resize((200, 200))).reshape(-1, 3)
    max_sat, best = -1, (0, 0, 0)
    for r, g, b in arr:
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        if v >= 0.15 and s >= 0.20 and s > max_sat:
            max_sat, best = s, (r, g, b)
    return best if max_sat != -1 else img.resize((1, 1)).getpixel((0, 0))

def rgb_to_hex(rgb): return "#{:02x}{:02x}{:02x}".format(*[int(c) for c in rgb])
def validate_rgb_color(c): return tuple(int(max(0, min(255, x))) for x in c)
def choose_code_color(bg): return "white" if (bg[0]*299 + bg[1]*587 + bg[2]*114) / 1000 < 125 else "black"
def get_color_distance(c1, c2): return sum((a-b)**2 for a, b in zip(c1, c2)) ** 0.5

def extract_margins_image(cover_img):
    w, h = cover_img.size
    margins = [
        cover_img.crop((0, 0, int(w * 0.20), h)).rotate(90, expand=True),
        cover_img.crop((int(w * 0.80), 0, w, h)).rotate(90, expand=True),
        cover_img.crop((0, 0, w, int(h * 0.20))),
        cover_img.crop((0, int(h * 0.80), w, h))
    ]
    combined = Image.new("RGB", (max(m.width for m in margins), sum(m.height for m in margins)))
    y = 0
    for m in margins:
        combined.paste(m, (0, y))
        y += m.height
    return combined

def interpolate_color(c1, c2, f):
    return tuple(int(max(0, min(255, c1[i] + (c2[i] - c1[i]) * f))) for i in range(3))

def get_color_palette(img_source, bg_color, num=5, min_dist=60, is_bw=False):
    if is_bw:
        br = (bg_color[0]*299 + bg_color[1]*587 + bg_color[2]*114) / 1000
        grays = [(60,60,60),(100,100,100),(140,140,140),(180,180,180),(220,220,220)] if br < 128 else [(30,30,30),(70,70,70),(110,110,110),(150,150,150),(190,190,190)]
        return grays
    
    palette = ColorThief(img_source).get_palette(color_count=20, quality=1)
    valid = [validate_rgb_color(c) for c in palette if get_color_distance(c, bg_color) >= min_dist][:num]
    if len(valid) < num:
        valid = [validate_rgb_color(c) for c in palette if get_color_distance(c, bg_color) >= min_dist * 0.7][:num]
    
    if len(valid) < num and len(valid) >= 2:
        valid.sort(key=lambda c: (c[0]*299 + c[1]*587 + c[2]*114) / 1000)
        while len(valid) < num:
            new = []
            for i in range(len(valid) - 1):
                new.append(valid[i])
                if len(valid) + len(new) < num:
                    new.append(validate_rgb_color(interpolate_color(valid[i], valid[i+1], 0.5)))
            new.append(valid[-1])
            valid = new[:num]
    
    valid = valid[:num]
    valid.sort(key=lambda c: (c[0]*299 + c[1]*587 + c[2]*114) / 1000)
    return valid

def calc_track_settings(n): 
    if n < 10: return 58, 1470, 76
    if n < 15: return 52, 1610, 70
    if n < 20: return 48, 1730, 66
    return 44, 1810, 62

def calc_title_fonts(artist, album, sep_y, sep_h, cover_y):
    total, al, abl = len(artist) + len(album), len(artist), len(album)
    if total > 60 or al > 35 or abl > 35: fs, vs = 90, 20
    elif total > 45 or al > 25 or abl > 25: fs, vs = 105, 25
    elif total > 35 or al > 20 or abl > 20: fs, vs = 115, 30
    else: fs, vs = 125, 35
    sep_bot = sep_y + sep_h
    center = sep_bot + (cover_y - sep_bot) // 2
    artist_y = center - (fs + vs + fs) // 2
    return fs, fs, artist_y, artist_y + fs + vs

def draw_tracklist(draw, data, cx, y_start, y_max, font_reg_path, font_bold_path, init_fs, color, max_w, lh):
    max_h = y_max - y_start
    fs = init_fs
    
    while fs >= 28:
        try:
            fr = ImageFont.truetype(str(font_reg_path), fs)
            fb = ImageFont.truetype(str(font_bold_path), fs)
        except:
            fr = fb = ImageFont.load_default()
        
        cur_lh = int(fs * 1.3)
        cur_max_w = max_w + (init_fs - fs) * 8
        lines, line, w = [], [], 0
        
        for i, (title, dur) in enumerate(data):
            tt = title + " "
            dt = dur + ("/ " if i < len(data) - 1 else "")
            tw = draw.textbbox((0, 0), tt, font=fr)[2]
            dw = draw.textbbox((0, 0), dt, font=fb)[2]
            if w + tw + dw > cur_max_w and line:
                lines.append(line)
                line, w = [], 0
            line.append((tt, dt, tw, dw))
            w += tw + dw
        if line: lines.append(line)
        
        if len(lines) * cur_lh <= max_h:
            y = y_start
            for ln in lines:
                lw = sum(it[2] + it[3] for it in ln)
                x = cx - lw // 2
                for tt, dt, tw, dw in ln:
                    draw.text((x, y), tt, fill=color, font=fr)
                    draw.text((x + tw, y), dt, fill=color, font=fb)
                    x += tw + dw
                y += cur_lh
            return
        fs -= 2

def gen_spotify_code(uri, bg, bar="white"):
    if "open.spotify.com" in uri:
        m = re.search(r'spotify\.com/(album|track|playlist)/([a-zA-Z0-9]+)', uri)
        if m: uri = f"spotify:{m.group(1)}:{m.group(2)}"
    try:
        r = requests.get(f"https://scannables.scdn.co/uri/plain/png/{bg.lstrip('#')}/{bar}/640/{uri}", timeout=10)
        return Image.open(io.BytesIO(r.content)) if r.status_code == 200 else None
    except: return None

def create_poster(data, cover, dominant, accent, palette, bg):
    W, H = 2480, 3500
    tc = "white" if (bg[0]*299 + bg[1]*587 + bg[2]*114) / 1000 < 125 else "black"
    
    cx, cy, cs = 335, 523, 1810
    sep_y, sep_h, sep_w = 170, 4, 1820
    sep_x = (W - sep_w) // 2
    sep_bot_y = H - 345 - sep_h
    
    afs, alfs, ay, aly = calc_title_fonts(data['artist_name'], data['album_title'], sep_y, sep_h, cy)
    
    # Fonts
    fb, fbo = FONTS_FOLDER / 'Book.otf', FONTS_FOLDER / 'Bold.otf'
    fo_r, fo_b = FONTS_FOLDER / 'Outfit-Regular.ttf', FONTS_FOLDER / 'Outfit-Bold.ttf'
    
    if fb.exists() and fbo.exists():
        af, alf, inf, bf = [ImageFont.truetype(str(fb), s) for s in [afs, alfs, 65, 55]]
        alf = ImageFont.truetype(str(fbo), alfs)
        tfr, tfb = fb, fbo
    elif fo_r.exists() and fo_b.exists():
        af, inf, bf = [ImageFont.truetype(str(fo_r), s) for s in [afs, 65, 55]]
        alf = ImageFont.truetype(str(fo_b), alfs)
        tfr, tfb = fo_r, fo_b
    else:
        af = alf = inf = bf = ImageFont.load_default()
        tfr = tfb = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    
    tfs, tmw, tlh = calc_track_settings(len(data['tracklist']))
    
    poster = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(poster)
    
    draw.rectangle([(sep_x, sep_y), (sep_x + sep_w, sep_y + sep_h)], fill=tc)
    draw.rectangle([(sep_x, sep_bot_y), (sep_x + sep_w, sep_bot_y + sep_h)], fill=tc)
    draw.text((cx + 31, ay), data['artist_name'], fill=tc, font=af)
    draw.text((cx + 31, aly), data['album_title'], fill=tc, font=alf)
    poster.paste(cover.resize((cs, cs)), (cx, cy))
    
    code = gen_spotify_code(data['spotify_url'], rgb_to_hex(bg), choose_code_color(bg))
    cw, ch = 580, 145
    code_x, code_y = W - 315 - cw, sep_bot_y - 85 - ch
    
    draw_tracklist(draw, data['tracklist'], W // 2, cy + cs + 50, code_y - 30, tfr, tfb, tfs, tc, tmw, tlh)
    
    draw.text((sep_x, sep_bot_y - 150), f"Label: {data['label']}\nGenre: {data['genre']}", fill=tc, font=inf, spacing=15)
    draw.text((sep_x, sep_bot_y + 40), data['release_year'], fill=tc, font=bf)
    st = f"{data['total_songs']} SONGS"
    draw.text(((W - draw.textbbox((0,0), st, font=bf)[2]) // 2, sep_bot_y + 40), st, fill=tc, font=bf)
    draw.text((sep_x + sep_w - draw.textbbox((0,0), data['duration'], font=bf)[2], sep_bot_y + 40), data['duration'], fill=tc, font=bf)
    
    if code: poster.paste(code.resize((cw, ch)), (code_x, code_y))
    
    pw, ph = 110, 30
    px = code_x + (cw - pw * 5) // 2
    for i, c in enumerate(palette[:5]):
        draw.rectangle([(px + i * pw, code_y + ch + 2), (px + (i+1) * pw, code_y + ch + 2 + ph)], fill=c)
    
    return poster

# Routes
@app.route('/')
def index():
    return jsonify({'status': 'ok', 'service': 'Hang It! API'})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/api/search', methods=['POST', 'OPTIONS'])
def search():
    if request.method == 'OPTIONS': return '', 204
    try:
        q = request.json.get('query', '')
        if not q: return jsonify({'error': 'Query required'}), 400
        
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))
        res = sp.search(q=q, type="album", limit=5)
        
        if not res['albums']['items']:
            return jsonify({'results': []})
        
        return jsonify({'success': True, 'results': [{
            'id': a['id'], 'name': a['name'],
            'artist': " · ".join([x["name"] for x in a["artists"]]),
            'cover': a['images'][0]['url'] if a['images'] else None,
            'release_year': a['release_date'].split('-')[0] if a.get('release_date') else 'N/A'
        } for a in res['albums']['items']]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-preview', methods=['POST', 'OPTIONS'])
def generate():
    if request.method == 'OPTIONS': return '', 204
    try:
        d = request.json
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))
        
        if d.get('album_id'):
            album = sp.album(d['album_id'])
        else:
            q = f"{d.get('artist', '')} {d.get('album', '')}".strip()
            if not q: return jsonify({'error': 'Artist/album required'}), 400
            res = sp.search(q=q, type="album", limit=1)
            if not res['albums']['items']: return jsonify({'error': 'Not found'}), 404
            album = sp.album(res['albums']['items'][0]['id'])
        
        artist_data = sp.artist(album['artists'][0]['id'])
        
        # Cover
        r = requests.get(album['images'][0]['url'], timeout=15)
        cover = Image.open(io.BytesIO(r.content)).convert('RGB')
        
        # Colors from margins
        margins = extract_margins_image(cover)
        mbuf = io.BytesIO()
        margins.save(mbuf, format='JPEG', quality=95)
        mbuf.seek(0)
        mdata = mbuf.getvalue()
        
        is_bw = is_black_and_white(margins)
        if is_bw:
            dominant = accent = get_bw_background(margins)
        else:
            dominant = validate_rgb_color(ColorThief(io.BytesIO(mdata)).get_color(quality=1))
            accent = validate_rgb_color(get_accent_color(cover))
        palette = get_color_palette(io.BytesIO(mdata), dominant, 5, 60, is_black_and_white(cover))
        
        # Tracks
        tracks = sp.album_tracks(album['id'])['items']
        tlist, tms = [], 0
        for t in tracks:
            title = remove_diacritics(t['name'])
            for s in ["(feat", "(Feat", "feat.", "feat ", "(with", "(With", "with "]:
                if s in title: title = title.split(s)[0].strip()
            tlist.append((title, f"{t['duration_ms']//60000}:{(t['duration_ms']//1000)%60:02d}"))
            tms += t['duration_ms']
        
        tsec = tms // 1000
        dur = f"{tsec//3600} HR {(tsec%3600)//60:02d} MIN" if tsec >= 3600 else f"{tsec//60} MIN {tsec%60:02d} SEC"
        
        # Genre
        genre = d.get('genre', '')
        if not genre:
            genres = [g.lower() for g in artist_data.get('genres', [])]
            genre = genres[0].title() if genres else ""
        
        # Label
        label = album.get("label", "Unknown")
        if "/" in label: label = label.split("/")[0].strip()
        
        data = {
            'artist_name': remove_diacritics(" · ".join([a["name"] for a in album["artists"]])),
            'album_title': remove_diacritics(album['name']),
            'tracklist': tlist,
            'total_songs': len(tracks),
            'duration': dur,
            'release_year': album['release_date'].split('-')[0],
            'label': label,
            'genre': genre,
            'spotify_url': album['external_urls']['spotify']
        }
        
        # Generate posters
        posters = []
        for bg, var in [(dominant, 'primary'), (accent, 'accent')]:
            poster = create_poster(data, cover, dominant, accent, palette, bg)
            preview = poster.copy()
            preview.thumbnail((600, 850), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            preview.save(buf, format='JPEG', quality=85)
            posters.append({
                'variant': var,
                'preview': f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}",
                'background_color': rgb_to_hex(bg)
            })
        
        # Cover thumb
        ct = cover.copy()
        ct.thumbnail((300, 300), Image.Resampling.LANCZOS)
        cbuf = io.BytesIO()
        ct.save(cbuf, format='JPEG', quality=85)
        
        return jsonify({
            'success': True,
            'album_info': {
                'id': album['id'],
                'name': data['album_title'],
                'artist': data['artist_name'],
                'cover': f"data:image/jpeg;base64,{base64.b64encode(cbuf.getvalue()).decode()}",
                'release_year': data['release_year'],
                'total_songs': len(tracks),
                'duration': dur,
                'genre': genre,
                'genre_found': bool(genre and not d.get('genre'))
            },
            'posters': posters
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
