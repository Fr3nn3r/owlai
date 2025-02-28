import spotipy
from spotipy.oauth2 import SpotifyOAuth
import logging

logger = logging.getLogger("spotify_logger")

# Authenticate with Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="abad9b5d6b754294afc53ff35e53483f",
    client_secret="767d6ab3afc64a53a049ca5fba8c5f8f",
    redirect_uri="http://localhost:8888/callback",
    scope="user-modify-playback-state,user-read-playback-state"
))



# Function to find device ID by name
def get_device_id(device_name):
    devices = sp.devices()
    for device in devices["devices"]:
        if device["name"].lower() == device_name.lower():
            return device["id"]
    return None

# Function to play a song on a specific device
def play_song_on_device(song_name, device_name):
    # Get device ID
    device_id = get_device_id(device_name)
    if not device_id:
        logger.error(f"Device '{device_name}' not found!")
        return

    # Search for the song
    results = sp.search(q=song_name, type="track", limit=1)
    if results["tracks"]["items"]:
        song_uri = results["tracks"]["items"][0]["uri"]
        sp.start_playback(device_id=device_id, uris=[song_uri])
        print(f"Playing '{song_name}' on '{device_name}'")
    else:
        print("Song not found!")

# Function to transfer playback to a new device
def transfer_playback(device_name):
    device_id = get_device_id(device_name)
    if not device_id:
        logger.error(f"Device '{device_name}' not found!")
        return
    
    # Transfer playback to the new device
    sp.transfer_playback(device_id=device_id, force_play=False)
    logger.info(f"Playback transferred to '{device_name}' without interruption.")

    # Ensure playback resumes
    try:
        sp.start_playback(device_id=device_id)
        logger.info("Playback resumed.")
    except Exception as e:
        logger.error(f"Could not start playback: {e}")


# Function to search for a song and play it on a specific device
def play_song_on_spotify_on_device(song_name, artist_name, device_name):
    # Search for the song
    query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=query, type="track", limit=1)
    
    if results["tracks"]["items"]:
        song_uri = results["tracks"]["items"][0]["uri"]
        device_id = get_device_id(device_name)
        
        if not device_id:
            logger.error(f"Device '{device_name}' not found!")
            return
        
        # Start playback on the chosen device
        sp.start_playback(device_id=device_id, uris=[song_uri])
        logger.info(f"Playing '{song_name}' by {artist_name} on '{device_name}'")
    else:
        logger.error(f"Song '{song_name}' by {artist_name} not found!")

def play_song_on_spotify(song_name, artist_name):
    play_song_on_spotify_on_device(song_name, artist_name, "ATHENA")


# Play "Shoot to Thrill" on ATHENA
# play_song_on_spotify("Fly away", "Lenny Kravitz")