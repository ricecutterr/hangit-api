"""
Hang It! Poster Generator - Web Application
Backend Flask pentru generarea posterelor de album
"""

from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS
import os
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

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, origins=["*"])

# ==========================================
# CONFIGURARE
# ==========================================
CLIENT_ID = "62068285c95d49e5a2c394c485e17bd1"
CLIENT_SECRET = "5d6813b8396d479e88ce5dbbf18c4442"

UPLOAD_FOLDER = tempfile.gettempdir()
FONTS_FOLDER = Path(__file__).parent / 'fonts'

# ==========================================
# FUNCȚII HELPER
# ==========================================

def remove_diacritics(text):
    romanian_diacritics = {
        'ă': 'a', 'Ă': 'A', 'â': 'a', 'Â': 'A',
        'î': 'i', 'Î': 'I', 'ș': 's', 'Ș': 'S', 'ț': 't', 'Ț': 'T'
    }
    for d, r in romanian_diacritics.items():
        text = text.replace(d, r)
    return text

def clean_filename(text):
    text = text.strip().replace(" ", "_")
    return re.sub(r'[\\/:*?"<>|]', '', text)

def is_black_and_white(img):
    img_small = img.resize((100, 100))
    arr = np.array(img_small) / 255.0
    hsv = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            r, g, b = arr[i, j]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            hsv[i, j] = (h, s, v)
    saturation_avg = hsv[:, :, 1].mean()
    high_saturation_pixels = np.sum(hsv[:, :, 1] > 0.5)
    if high_saturation_pixels > 20:
        return False
    return bool(saturation_avg < 0.08)

def get_bw_background(img):
    img_small = img.resize((100, 100))
    arr = np.array(img_small)
    lum = int(arr.mean())
    luminosity = arr.mean(axis=2)
    very_dark = np.sum(luminosity < 50)
    very_light = np.sum(luminosity > 205)
    extreme_ratio = (very_dark + very_light) / luminosity.size
    if extreme_ratio < 0.60:
        lum = max(110, min(170, lum if 100 <= lum <= 180 else (110 if lum < 100 else 160)))
        return (lum, lum, lum)
    if lum >= 180: return (255, 255, 255)
    if lum <= 100: return (0, 0, 0)
    return (max(110, min(170, lum)),) * 3

def get_accent_color(img):
    small = img.resize((200, 200))
    arr = np.array(small).reshape(-1, 3)
    max_saturation, best_color = -1, (0, 0, 0)
    for r, g, b in arr:
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        if v >= 0.15 and s >= 0.20 and s > max_saturation:
            max_saturation, best_color = s, (r, g, b)
    return best_color if max_saturation != -1 else img.resize((1, 1)).getpixel((0, 0))

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def validate_rgb_color(color):
    return tuple(int(max(0, min(255, c))) for c in color)

def choose_code_color(bg_rgb):
    if isinstance(bg_rgb, str):
        bg_rgb = hex_to_rgb(bg_rgb)
    r, g, b = bg_rgb
    brightness = (r*299 + g*587 + b*114) / 1000
    return "white" if brightness < 125 else "black"

def extract_margins_image(cover_img):
    """Extrage marginile imaginii pentru analiza culorilor (ca în aplicația originală)"""
    w, h = cover_img.size
    
    # Extragem 20% din fiecare margine
    left_margin = cover_img.crop((0, 0, int(w * 0.20), h))
    right_margin = cover_img.crop((int(w * 0.80), 0, w, h))
    top_margin = cover_img.crop((0, 0, w, int(h * 0.20)))
    bottom_margin = cover_img.crop((0, int(h * 0.80), w, h))
    
    # Rotim marginile laterale pentru a le combina
    left_rotated = left_margin.rotate(90, expand=True)
    right_rotated = right_margin.rotate(90, expand=True)
    
    horizontal_margins = [left_rotated, right_rotated, top_margin, bottom_margin]
    
    # Combinăm toate marginile într-o singură imagine
    combined_width = max(m.width for m in horizontal_margins)
    combined_height = sum(m.height for m in horizontal_margins)
    
    combined = Image.new("RGB", (combined_width, combined_height))
    
    y_offset = 0
    for m in horizontal_margins:
        combined.paste(m, (0, y_offset))
        y_offset += m.height
    
    return combined

def get_color_distance(c1, c2):
    return sum((a-b)**2 for a, b in zip(c1, c2)) ** 0.5

def interpolate_color(color1, color2, factor):
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    r = int(max(0, min(255, r1 + (r2 - r1) * factor)))
    g = int(max(0, min(255, g1 + (g2 - g1) * factor)))
    b = int(max(0, min(255, b1 + (b2 - b1) * factor)))
    return (r, g, b)

def get_color_palette(img_source, background_color, num_colors=5, min_distance=60, is_bw=False):
    if is_bw:
        bg_brightness = (background_color[0]*299 + background_color[1]*587 + background_color[2]*114) / 1000
        if bg_brightness < 128:
            grays = [(60,60,60), (100,100,100), (140,140,140), (180,180,180), (220,220,220)]
        else:
            grays = [(30,30,30), (70,70,70), (110,110,110), (150,150,150), (190,190,190)]
        return [validate_rgb_color(c) for c in grays]
    
    color_thief = ColorThief(img_source)
    palette = color_thief.get_palette(color_count=20, quality=1)
    
    valid_colors = []
    for color in palette:
        distance = get_color_distance(color, background_color)
        if distance >= min_distance:
            valid_colors.append(validate_rgb_color(color))
        if len(valid_colors) >= num_colors:
            break
    
    if len(valid_colors) < num_colors:
        valid_colors = []
        for color in palette:
            distance = get_color_distance(color, background_color)
            if distance >= min_distance * 0.7:
                valid_colors.append(validate_rgb_color(color))
            if len(valid_colors) >= num_colors:
                break
    
    # Dacă tot nu avem suficiente culori, interpolăm între cele existente
    if len(valid_colors) < num_colors and len(valid_colors) >= 2:
        def get_brightness(color):
            r, g, b = color
            return (r*299 + g*587 + b*114) / 1000
        
        valid_colors.sort(key=get_brightness)
        
        while len(valid_colors) < num_colors:
            new_colors = []
            for i in range(len(valid_colors) - 1):
                new_colors.append(valid_colors[i])
                if len(valid_colors) + len(new_colors) < num_colors:
                    mid_color = validate_rgb_color(interpolate_color(valid_colors[i], valid_colors[i+1], 0.5))
                    new_colors.append(mid_color)
            new_colors.append(valid_colors[-1])
            valid_colors = new_colors[:num_colors]
        
        return valid_colors
    
    valid_colors = valid_colors[:num_colors]
    
    def get_brightness(color):
        r, g, b = color
        return (r*299 + g*587 + b*114) / 1000
    
    valid_colors.sort(key=get_brightness)
    
    return [validate_rgb_color(c) for c in valid_colors]

def calculate_dynamic_tracklist_settings(tracklist_data, cover_width):
    n = len(tracklist_data)
    if n < 10: return 58, 1470, 76
    if n < 15: return 52, 1610, 70
    if n < 20: return 48, 1730, 66
    return 44, 1810, 62

def calculate_dynamic_artist_title_fonts(artist_text, album_text, separator_y, separator_height, cover_y):
    total_len = len(artist_text) + len(album_text)
    artist_len, album_len = len(artist_text), len(album_text)
    
    if total_len > 60 or artist_len > 35 or album_len > 35:
        fs, vs = 90, 20
    elif total_len > 45 or artist_len > 25 or album_len > 25:
        fs, vs = 105, 25
    elif total_len > 35 or artist_len > 20 or album_len > 20:
        fs, vs = 115, 30
    else:
        fs, vs = 125, 35
    
    separator_bottom = separator_y + separator_height
    available_height = cover_y - separator_bottom
    group_height = fs + vs + fs
    center_y = separator_bottom + (available_height // 2)
    artist_y = center_y - (group_height // 2)
    album_y = artist_y + fs + vs
    
    return fs, fs, artist_y, album_y

def draw_tracklist_centered(draw, data, center_x, y_start, y_max, font_path_regular, font_path_bold, initial_font_size, color, max_line_width, line_height):
    """Desenează tracklist-ul centrat, micșorând fontul dacă e necesar"""
    max_height = y_max - y_start
    font_size = initial_font_size
    min_font_size = 28  # Nu mergem sub această dimensiune
    
    while font_size >= min_font_size:
        # Încărcăm fonturile la dimensiunea curentă
        try:
            font_regular = ImageFont.truetype(str(font_path_regular), font_size)
            font_bold = ImageFont.truetype(str(font_path_bold), font_size)
        except:
            font_regular = font_bold = ImageFont.load_default()
        
        # Calculăm line_height proporțional cu font_size
        current_line_height = int(font_size * 1.3)
        
        # Calculăm max_line_width proporțional (mai mic fontul = mai multe caractere pe linie)
        current_max_width = max_line_width + (initial_font_size - font_size) * 8
        
        lines, current_line, current_width = [], [], 0
        
        for i, (title, duration) in enumerate(data):
            title_text = title + " "
            duration_text = duration + ("/ " if i < len(data) - 1 else "")
            
            tw = draw.textbbox((0, 0), title_text, font=font_regular)[2]
            dw = draw.textbbox((0, 0), duration_text, font=font_bold)[2]
            
            if current_width + tw + dw > current_max_width and current_line:
                lines.append(current_line)
                current_line, current_width = [], 0
            
            current_line.append((title_text, duration_text, tw, dw))
            current_width += tw + dw
        
        if current_line:
            lines.append(current_line)
        
        # Verificăm dacă încape
        total_height = len(lines) * current_line_height
        if total_height <= max_height:
            # Încape! Desenăm
            current_y = y_start
            for line in lines:
                line_width = sum(item[2] + item[3] for item in line)
                current_x = center_x - (line_width // 2)
                for title_text, duration_text, tw, dw in line:
                    draw.text((current_x, current_y), title_text, fill=color, font=font_regular)
                    current_x += tw
                    draw.text((current_x, current_y), duration_text, fill=color, font=font_bold)
                    current_x += dw
                current_y += current_line_height
            
            if font_size < initial_font_size:
                print(f"📏 Font tracklist micșorat: {initial_font_size} → {font_size}")
            return
        
        # Nu încape, micșorăm fontul
        font_size -= 2
    
    # Dacă ajungem aici, desenăm cu fontul minim (și acceptăm overflow)
    print(f"⚠️ Tracklist prea lung, folosim font minim {min_font_size}")
    font_regular = ImageFont.truetype(str(font_path_regular), min_font_size)
    font_bold = ImageFont.truetype(str(font_path_bold), min_font_size)
    current_line_height = int(min_font_size * 1.3)
    
    lines, current_line, current_width = [], [], 0
    current_max_width = max_line_width + (initial_font_size - min_font_size) * 8
    
    for i, (title, duration) in enumerate(data):
        title_text = title + " "
        duration_text = duration + ("/ " if i < len(data) - 1 else "")
        tw = draw.textbbox((0, 0), title_text, font=font_regular)[2]
        dw = draw.textbbox((0, 0), duration_text, font=font_bold)[2]
        if current_width + tw + dw > current_max_width and current_line:
            lines.append(current_line)
            current_line, current_width = [], 0
        current_line.append((title_text, duration_text, tw, dw))
        current_width += tw + dw
    if current_line:
        lines.append(current_line)
    
    current_y = y_start
    for line in lines:
        if current_y + current_line_height > y_max:
            break
        line_width = sum(item[2] + item[3] for item in line)
        current_x = center_x - (line_width // 2)
        for title_text, duration_text, tw, dw in line:
            draw.text((current_x, current_y), title_text, fill=color, font=font_regular)
            current_x += tw
            draw.text((current_x, current_y), duration_text, fill=color, font=font_bold)
            current_x += dw
        current_y += current_line_height

def generate_spotify_code_api(spotify_uri, bg_color, bar_color="white", size=640):
    if "open.spotify.com" in spotify_uri:
        match = re.search(r'spotify\.com/(album|track|playlist)/([a-zA-Z0-9]+)', spotify_uri)
        if match:
            spotify_uri = f"spotify:{match.group(1)}:{match.group(2)}"
    
    url = f"https://scannables.scdn.co/uri/plain/png/{bg_color.lstrip('#')}/{bar_color}/{size}/{spotify_uri}"
    try:
        response = requests.get(url, timeout=10)
        return Image.open(io.BytesIO(response.content)) if response.status_code == 200 else None
    except:
        return None

def create_poster(album_data, cover_img, dominant_color, accent_color, color_palette, bg_color):
    WIDTH, HEIGHT = 2480, 3500
    R, G, B = bg_color
    text_color = "white" if (R*299 + G*587 + B*114) / 1000 < 125 else "black"
    
    cover_x, cover_y, cover_size = 335, 523, 1810
    separator_top_y, separator_height, separator_width = 170, 4, 1820
    separator_x = (WIDTH - separator_width) // 2
    separator_bottom_y = HEIGHT - 345 - separator_height
    text_x = cover_x + 31
    
    artist_name = album_data['artist_name']
    album_title = album_data['album_title']
    
    artist_fs, album_fs, artist_y, album_y = calculate_dynamic_artist_title_fonts(
        artist_name, album_title, separator_top_y, separator_height, cover_y
    )
    
    font_book = FONTS_FOLDER / 'Book.otf'
    font_bold = FONTS_FOLDER / 'Bold.otf'
    font_outfit_reg = FONTS_FOLDER / 'Outfit-Regular.ttf'
    font_outfit_bold = FONTS_FOLDER / 'Outfit-Bold.ttf'
    
    # Încearcă fonturile în ordine: Book/Bold (original) -> Outfit -> sistem
    if font_book.exists() and font_bold.exists():
        artist_font = ImageFont.truetype(str(font_book), artist_fs)
        album_font = ImageFont.truetype(str(font_bold), album_fs)
        info_font = ImageFont.truetype(str(font_book), 65)
        bottom_font = ImageFont.truetype(str(font_book), 55)
    elif font_outfit_reg.exists() and font_outfit_bold.exists():
        artist_font = ImageFont.truetype(str(font_outfit_reg), artist_fs)
        album_font = ImageFont.truetype(str(font_outfit_bold), album_fs)
        info_font = ImageFont.truetype(str(font_outfit_reg), 65)
        bottom_font = ImageFont.truetype(str(font_outfit_reg), 55)
    else:
        for reg, bld in [("arial.ttf", "arialbd.ttf"), ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]:
            try:
                artist_font = ImageFont.truetype(reg, artist_fs)
                album_font = ImageFont.truetype(bld, album_fs)
                info_font = ImageFont.truetype(reg, 65)
                bottom_font = ImageFont.truetype(reg, 55)
                break
            except: continue
        else:
            artist_font = album_font = info_font = bottom_font = ImageFont.load_default()
    
    track_fs, track_max_w, track_lh = calculate_dynamic_tracklist_settings(album_data['tracklist'], cover_size)
    
    # Determinăm căile fonturilor pentru tracklist
    if font_book.exists() and font_bold.exists():
        track_font_path_regular = font_book
        track_font_path_bold = font_bold
    elif font_outfit_reg.exists() and font_outfit_bold.exists():
        track_font_path_regular = font_outfit_reg
        track_font_path_bold = font_outfit_bold
    else:
        # Fallback la fonturi sistem
        track_font_path_regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        track_font_path_bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if not track_font_path_regular.exists():
            track_font_path_regular = Path("arial.ttf")
            track_font_path_bold = Path("arialbd.ttf")
    
    poster = Image.new("RGB", (WIDTH, HEIGHT), bg_color)
    draw = ImageDraw.Draw(poster)
    
    # Separators
    draw.rectangle([(separator_x, separator_top_y), (separator_x + separator_width, separator_top_y + separator_height)], fill=text_color)
    draw.rectangle([(separator_x, separator_bottom_y), (separator_x + separator_width, separator_bottom_y + separator_height)], fill=text_color)
    
    # Artist & Album
    draw.text((text_x, artist_y), artist_name, fill=text_color, font=artist_font)
    draw.text((text_x, album_y), album_title, fill=text_color, font=album_font)
    
    # Cover
    poster.paste(cover_img.resize((cover_size, cover_size)), (cover_x, cover_y))
    
    # Spotify Code
    code_img = generate_spotify_code_api(album_data['spotify_url'], rgb_to_hex(bg_color), choose_code_color(bg_color))
    code_w, code_h = 580, 145
    code_x = WIDTH - 315 - code_w
    code_y = separator_bottom_y - 85 - code_h
    
    # Label & Genre - poziția lor
    info_y = separator_bottom_y - 150
    
    # Tracklist - y_max trebuie să fie deasupra Spotify Code (cu margine)
    tracklist_y_start = cover_y + cover_size + 50
    tracklist_y_max = code_y - 30  # 30px margine deasupra Spotify Code
    
    draw_tracklist_centered(draw, album_data['tracklist'], WIDTH // 2, tracklist_y_start, tracklist_y_max, track_font_path_regular, track_font_path_bold, track_fs, text_color, track_max_w, track_lh)
    
    # Label & Genre
    extra_info = f"Label: {album_data['label']}\nGenre: {album_data['genre']}"
    draw.text((separator_x, info_y), extra_info, fill=text_color, font=info_font, spacing=15)
    
    # Bottom info
    bottom_y = separator_bottom_y + 40
    draw.text((separator_x, bottom_y), album_data['release_year'], fill=text_color, font=bottom_font)
    
    songs_text = f"{album_data['total_songs']} SONGS"
    songs_w = draw.textbbox((0, 0), songs_text, font=bottom_font)[2]
    draw.text(((WIDTH - songs_w) // 2, bottom_y), songs_text, fill=text_color, font=bottom_font)
    
    duration_w = draw.textbbox((0, 0), album_data['duration'], font=bottom_font)[2]
    draw.text((separator_x + separator_width - duration_w, bottom_y), album_data['duration'], fill=text_color, font=bottom_font)
    
    # Spotify Code (paste)
    if code_img:
        poster.paste(code_img.resize((code_w, code_h)), (code_x, code_y))
    
    # Palette
    pal_w, pal_h = 110, 30
    pal_start_x = code_x + (code_w - pal_w * 5) // 2
    pal_y = code_y + code_h + 2
    for i, color in enumerate(color_palette[:5]):
        draw.rectangle([(pal_start_x + i * pal_w, pal_y), (pal_start_x + (i + 1) * pal_w, pal_y + pal_h)], fill=color)
    
    return poster

# ==========================================
# ROUTES
# ==========================================

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Servește fișierele statice (logo, etc.)"""
    return send_from_directory('static', filename)

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/api/search', methods=['POST'])
def search_album():
    """Caută album - returnează 5 rezultate"""
    try:
        query = request.json.get('query', '')
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))
        results = sp.search(q=query, type="album", limit=5)
        
        if not results['albums']['items']:
            return jsonify({'error': 'Album not found'}), 404
        
        albums = [{
            'id': a['id'],
            'name': a['name'],
            'artist': " · ".join([art["name"] for art in a["artists"]]),
            'cover': a['images'][0]['url'] if a['images'] else None,
            'release_year': a['release_date'].split('-')[0] if a.get('release_date') else 'N/A',
            'total_tracks': a.get('total_tracks', 0)
        } for a in results['albums']['items']]
        
        return jsonify({'success': True, 'results': albums})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/select-album', methods=['POST'])
def select_album():
    """Selectează un album și returnează detalii complete"""
    try:
        album_id = request.json.get('album_id', '')
        if not album_id:
            return jsonify({'error': 'Album ID is required'}), 400
        
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))
        album = sp.album(album_id)
        artist_data = sp.artist(album['artists'][0]['id'])
        
        # Cover - cu error handling
        cover_url = album['images'][0]['url']
        print(f"📥 Downloading cover from: {cover_url}")
        
        cover_response = requests.get(cover_url, timeout=15)
        if cover_response.status_code != 200:
            return jsonify({'error': f'Failed to download cover image (status {cover_response.status_code})'}), 500
        
        cover_data = cover_response.content
        print(f"📦 Cover data size: {len(cover_data)} bytes")
        
        if len(cover_data) < 1000:
            return jsonify({'error': 'Cover image too small or invalid'}), 500
        
        try:
            cover_img = Image.open(io.BytesIO(cover_data)).convert('RGB')
            print(f"✅ Cover loaded: {cover_img.size}")
        except Exception as img_err:
            print(f"❌ Image error: {img_err}")
            return jsonify({'error': f'Cannot open cover image: {str(img_err)}'}), 500
        
        cover_path = os.path.join(UPLOAD_FOLDER, f"cover_{album_id}.jpg")
        cover_img.save(cover_path, 'JPEG', quality=95)
        
        # Extragem imaginea cu marginile pentru analiza culorilor (ca în app-ul original)
        try:
            margins_img = extract_margins_image(cover_img)
            margins_buf = io.BytesIO()
            margins_img.save(margins_buf, format='JPEG', quality=95)
            margins_buf.seek(0)
            margins_data = margins_buf.getvalue()
            print(f"✅ Margins extracted: {margins_img.size}, {len(margins_data)} bytes")
        except Exception as margin_err:
            print(f"❌ Margins error: {margin_err}")
            return jsonify({'error': f'Cannot extract margins: {str(margin_err)}'}), 500
        
        # Colors - folosim marginile pentru extragere
        try:
            is_bw = is_black_and_white(margins_img)
            print(f"📊 Is B&W: {is_bw}")
            if is_bw:
                dominant = accent = get_bw_background(margins_img)
            else:
                dominant = validate_rgb_color(ColorThief(io.BytesIO(margins_data)).get_color(quality=1))
                accent = validate_rgb_color(get_accent_color(cover_img))
            print(f"🎨 Dominant: {dominant}, Accent: {accent}")
        except Exception as color_err:
            print(f"❌ Color extraction error: {color_err}")
            return jsonify({'error': f'Cannot extract colors: {str(color_err)}'}), 500
        
        # Paleta tot din margini
        try:
            palette = get_color_palette(io.BytesIO(margins_data), dominant, 5, 60, is_black_and_white(cover_img))
            print(f"🎨 Palette: {palette}")
        except Exception as palette_err:
            print(f"❌ Palette error: {palette_err}")
            return jsonify({'error': f'Cannot extract palette: {str(palette_err)}'}), 500
        
        # Tracks
        tracks = sp.album_tracks(album_id)['items']
        tracklist, total_ms = [], 0
        for t in tracks:
            title = remove_diacritics(t['name'])
            
            # Ștergem doar feat/with și variante
            for s in ["(feat", "(Feat", "(FEAT", "feat.", "Feat.", "feat ", "(with", "(With", "with "]:
                if s in title:
                    title = title.split(s)[0].strip()
            
            tracklist.append({'title': title, 'duration': f"{t['duration_ms']//60000}:{(t['duration_ms']//1000)%60:02d}"})
            total_ms += t['duration_ms']
        
        total_sec = total_ms // 1000
        duration = f"{total_sec//3600} HR {(total_sec%3600)//60:02d} MIN" if total_sec >= 3600 else f"{total_sec//60} MIN {total_sec%60:02d} SEC"
        
        # Genre
        genres = [g.lower() for g in artist_data.get('genres', [])]
        genres = ["trap" if g == "manele" else g for g in genres]
        
        # Label
        label = album.get("label", "Unknown Label").strip()
        if "/" in label:
            parts = [l.strip() for l in label.split("/")][:2]
            label = "/".join(parts) if len("/".join(parts)) <= 30 else parts[0]
        
        # Cover base64
        buf = io.BytesIO()
        cover_img.save(buf, format="JPEG", quality=85)
        
        return jsonify({
            'success': True,
            'album': {
                'id': album_id,
                'name': remove_diacritics(album['name']),
                'artist': remove_diacritics(" · ".join([a["name"] for a in album["artists"]])),
                'cover': f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}",
                'cover_path': cover_path,
                'release_year': album['release_date'].split('-')[0],
                'label': label,
                'genre': genres[0].title() if genres else "",
                'total_songs': len(tracks),
                'duration': duration,
                'spotify_url': album['external_urls']['spotify'],
                'tracklist': tracklist,
                'colors': {
                    'dominant': rgb_to_hex(dominant), 'dominant_rgb': dominant,
                    'accent': rgb_to_hex(accent), 'accent_rgb': accent,
                    'palette': [rgb_to_hex(c) for c in palette], 'palette_rgb': palette,
                    'is_bw': is_bw
                }
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate_posters():
    """Generează posterele"""
    try:
        data = request.json
        album_info, colors = data['album'], data['colors']
        genre = data.get('genre', album_info.get('genre', 'Unknown'))
        
        # Cover
        cover_path = album_info.get('cover_path')
        if cover_path and os.path.exists(cover_path):
            cover_img = Image.open(cover_path).convert('RGB')
        else:
            cover_img = Image.open(io.BytesIO(base64.b64decode(album_info['cover'].split(',')[1]))).convert('RGB')
        
        dominant = tuple(colors.get('dominant_rgb', hex_to_rgb(colors['dominant'])))
        accent = tuple(colors.get('accent_rgb', hex_to_rgb(colors['accent'])))
        palette = [tuple(c) if isinstance(c, list) else hex_to_rgb(c) for c in colors.get('palette_rgb', colors['palette'])]
        
        album_data = {
            'artist_name': album_info['artist'],
            'album_title': album_info['name'],
            'tracklist': [(t['title'], t['duration']) for t in album_info['tracklist']],
            'total_songs': album_info['total_songs'],
            'duration': album_info['duration'],
            'release_year': album_info['release_year'],
            'label': album_info['label'],
            'genre': genre,
            'spotify_url': album_info['spotify_url']
        }
        
        posters = []
        for bg, suffix, ptype in [(dominant, "", "primary"), (accent, "_accent", "accent")]:
            poster = create_poster(album_data, cover_img, dominant, accent, palette, bg)
            
            buf = io.BytesIO()
            poster.save(buf, format='JPEG', quality=95)
            
            name = f"{clean_filename(album_info['artist'])}_{clean_filename(album_info['name'])}_poster{suffix}.jpg"
            tiff_name = name.replace('.jpg', '.tif')
            
            poster.save(os.path.join(UPLOAD_FOLDER, name), 'JPEG', quality=95)
            poster.save(os.path.join(UPLOAD_FOLDER, tiff_name), 'TIFF', compression='tiff_lzw')
            
            posters.append({
                'type': ptype,
                'name': name,
                'tiff_name': tiff_name,
                'preview': f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}",
                'color': rgb_to_hex(bg)
            })
        
        return jsonify({'success': True, 'posters': posters})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<filename>')
def download_poster(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(path, as_attachment=True, download_name=filename) if os.path.exists(path) else (jsonify({'error': 'File not found'}), 404)

@app.route('/api/pick-color', methods=['POST'])
def pick_color():
    try:
        data = request.json
        x, y = data['x'], data['y']
        
        if data.get('cover_path') and os.path.exists(data['cover_path']):
            img = Image.open(data['cover_path'])
        else:
            img = Image.open(io.BytesIO(base64.b64decode(data['cover'].split(',')[1])))
        
        w, h = img.size
        pw, ph = data.get('preview_width', w), data.get('preview_height', h)
        ax, ay = max(0, min(w-1, int(x * w / pw))), max(0, min(h-1, int(y * h / ph)))
        
        color = img.getpixel((ax, ay))
        return jsonify({'success': True, 'color': rgb_to_hex(color), 'rgb': list(color)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    FONTS_FOLDER.mkdir(exist_ok=True)
    port = int(os.environ.get('PORT', 8080))
    print("=" * 50)
    print("🎨 Hang It! Poster Generator")
    print(f"📍 Running on port: {port}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
