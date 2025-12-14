# Hang It! API - Deploy pe Railway

## Deploy rapid (5 minute):

### 1. Creează cont Railway
- Mergi la https://railway.app
- Sign up cu GitHub

### 2. Creează proiect nou
- Click "New Project"
- Alege "Deploy from GitHub repo" sau "Empty Project"

### 3. Dacă alegi "Empty Project":
- Click pe proiect → "Add Service" → "Empty Service"
- În Settings → "Deploy" → Upload acest folder ca ZIP
- SAU conectează un repo GitHub cu aceste fișiere

### 4. Dacă alegi GitHub:
- Fork/Upload aceste fișiere într-un repo nou
- Railway le detectează automat

### 5. Configurare (opțional):
În Railway dashboard → Variables, poți adăuga:
- `SPOTIFY_CLIENT_ID` - ID-ul tău Spotify (sau folosește cel default)
- `SPOTIFY_CLIENT_SECRET` - Secret-ul tău Spotify

### 6. Obține URL-ul:
- Railway → Settings → Domains
- Click "Generate Domain"
- Vei primi ceva gen: `hangit-api-production.up.railway.app`

### 7. Testează:
```
curl https://DOMENIUL-TAU.up.railway.app/
```

## Fonturi custom (opțional):
Pune `Book.otf` și `Bold.otf` în folderul `fonts/` înainte de deploy.
Dacă nu, se folosesc fonturile de sistem (DejaVu).

## Endpoints API:

### GET /
Status și info

### POST /api/search
```json
{"query": "Travis Scott Astroworld"}
```

### POST /api/generate-preview
```json
{
  "artist": "Travis Scott",
  "album": "Astroworld",
  "genre": "Hip-Hop"  // opțional
}
```
sau
```json
{
  "album_id": "41GuZcammIkupMPKH2OJ6I"
}
```
