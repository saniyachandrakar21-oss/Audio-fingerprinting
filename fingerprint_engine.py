import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import spectrogram
from skimage.feature import peak_local_max

import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import os

def load_song(song_paths):
    
   songs = []
   for path in song_paths :
       
        audio, f_s = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        songs. append({ "name": os.path.basename(path), "audio" : audio , "fs" :f_s })
      
   return songs

def plot_spectrograms(songs,nperseg=4095,show_plot=True):
    spectrograms = []

    for song in songs:
        name = song["name"]
        audio = song["audio"]
        fs = song["fs"]

        frequency, time, spec_matrix = spectrogram(audio,fs,nperseg=nperseg)
        spectrograms.append({"name": name,"frequency": frequency,"time": time,"spec_matrix": spec_matrix})

        if show_plot:

            plt.figure(figsize=(4,3))
            plt.pcolormesh(time,frequency,10*np.log10(spec_matrix + 1e-10),shading='gouraud')
            plt.xlim(10,40)
            plt.ylim(0,5000)
            plt.title(f"Spectrogram of {name}")
            plt.xlabel("Time (in sec)")
            plt.ylabel("Frequency (Hz)")
            plt.colorbar(label="Power (dB)")

            plt.tight_layout()
            plt.show()

    return spectrograms
def plot_constellation_maps(spectrograms, min_distance = 11 , percentile_threshold = 95 , show_plot= True):
    constellation_data = []
    for spec in spectrograms:
        name = spec["name"]
        frequency = spec["frequency"]
        time = spec["time"]
        spec_matrix = spec["spec_matrix"]
        
        spec_in_db = 10*np.log10(spec_matrix + 1e-10)
        peaks = peak_local_max( spec_in_db , min_distance=min_distance, threshold_abs=np.percentile(spec_in_db,percentile_threshold))

        constellation_data.append({"name": name ,"peaks": peaks, "frequency": frequency , "time": time, "spec_in_db": spec_in_db } )
        if show_plot:
            
            # plotting the maps 
            fig, ax = plt.subplots(1, 2, figsize=(14,5))
            # plotting peaks in spectrogram 
        
            ax[0].pcolormesh(time,frequency,spec_in_db,shading='gouraud')
            ax[0].scatter( time[peaks[:,1]],frequency[peaks[:,0]],c='red',s=2)
            ax[0].set_xlim(10,40)
            ax[0].set_ylim(0,5000)
            ax[0].set_title("Spectrogram with Peaks")
            ax[0].set_xlabel("Time (in sec)")
            ax[0].set_ylabel("Frequency (Hz)") 

        # plottimg constellation map separately 
       
            ax[1].scatter(time[peaks[:,1]],frequency[peaks[:,0]],s=4)
            ax[1].set_xlim(10,40)
            ax[1].set_ylim(0,5000)
            ax[1].set_title(f"Constellation Map : {name}")
            ax[1].set_xlabel("Time (in sec)")
            ax[1].set_ylabel("Frequency (Hz)")

        
            plt.tight_layout()
            plt.show()
    return constellation_data

def hash_database(constellation_data , num_pairs= 5):
    database = {}
    for song in constellation_data:
        name = song["name"]
        peaks = song["peaks"]
        peaks = peaks[np.argsort(peaks[:,1])]
        for i in range(len(peaks)):
            f1 = peaks[i][0]
            t1 = peaks[i][1]
            for j in range( i+1, min(i+num_pairs+1,len(peaks))):
                f2 = peaks[j][0]
                t2 = peaks[j][1]
                dt = t2 - t1
                if dt <= 0:
                     continue
                    
                hash_key = (int(f1), int(f2), int(dt) )

                if hash_key not in database:
                    database[hash_key] = []
                database[hash_key].append((name , int(t1)))
    return database

def build_database( Song_Folder):
    #creating list of audios
    song_paths=get_audio_paths(Song_Folder)
   # Song loading  
    songs = load_song(song_paths) 
    # Step 1: Spectrograms
    spectrograms = plot_spectrograms(songs, show_plot = False )

    # Step 2: Constellation maps
    constellation_data = plot_constellation_maps(spectrograms, show_plot = False )

    # Step 3: Hash database
    songs_database = hash_database(constellation_data)
    return songs_database

def get_audio_paths(path):
    
    audio_paths = []
   # Single audio file
    if os.path.isfile(path):
        audio_paths.append(path)
    #  Folder containing audio files
    elif os.path.isdir(path):
        
        for file in os.listdir(path):
            
            if file.endswith(".mp3"):
                full_path = os.path.join(path, file)
                audio_paths.append(full_path)

    else:
        print("Invalid path")

    return audio_paths

def generate_query_hashes(Query_Folder, num_target_peaks=5):
    
    Query_paths= get_audio_paths(Query_Folder)
    Queries =load_song(Query_paths) # songs as list of dict
    spectrograms = plot_spectrograms(Queries, show_plot = False )
    constellation_data = plot_constellation_maps(spectrograms, show_plot = False )
    query_hashes = []
    for query in constellation_data:
        name = query["name"]
        peaks = query["peaks"]
        peaks = peaks[np.argsort(peaks[:,1])]
        hashes = []
        for i in range(len(peaks)):
            
            f1 = peaks[i][0]
            t1 = peaks[i][1]
            for j in range( i+1,min(i+num_target_peaks+1, len(peaks))):
                f2 = peaks[j][0]
                t2 = peaks[j][1]
                dt = t2 - t1
                if dt <= 0:
                    continue
                    

                hash_key = ( int(f1), int(f2), int(dt))

                hashes.append( (hash_key, int(t1))) # hash and time stamp
                
        query_hashes.append({"name": name,"hashes": hashes}) #complete hash 

    return query_hashes, constellation_data

def song_identifier(query_hashes, songs_database):

    offset_count = {}
    for hash_key, query_time in query_hashes:
        if hash_key not in songs_database:
            continue
        matches = songs_database[hash_key]

        for song_name, db_time in matches:
            offset = db_time - query_time
            match_key = ( song_name, offset )

            if match_key not in offset_count:
                offset_count[match_key] = 0
            offset_count[match_key] += 1
    if len(offset_count) == 0:
        print("No match found")
        return None , None
    best_match = max(offset_count, key=offset_count.get)
    best_count = offset_count[best_match]
    if best_count < 10:  
        print("Unable to identify song")
        return None, offset_count
    matched_song = best_match[0]
    print("Matched Song:", matched_song)

    print( "Match Count:",offset_count[best_match])

    return matched_song , offset_count
def plot_offset_histogram(matched_song, offset_count):

    offsets = []
    counts = []

    for (song_name, offset), count in offset_count.items():

        if song_name == matched_song:

            offsets.append(offset)
            counts.append(count)

    if len(offsets) == 0:
        return None

    if len(offsets) == 0:
        print("No offsets to plot")
        return
    fig, ax = plt.subplots(figsize=(12,6))

    ax.bar(offsets, counts,width=30,color='blue')

    ax.set_title(
        f"Offset Histogram : {matched_song}"
    )

    ax.set_xlabel("Offset")

    ax.set_ylabel("Match Count")

    return fig
def get_library_data(song_folder):

    song_paths = get_audio_paths(song_folder)

    songs = load_song(song_paths)

    spectrograms = plot_spectrograms(
        songs,
        show_plot=False
    )

    constellation_data = plot_constellation_maps(
        spectrograms,
        show_plot=False
    )

    library_data = []

    for song in constellation_data:

        name = song["name"]

        peaks = song["peaks"]

        peaks = peaks[np.argsort(peaks[:,1])]

        hash_count = 0

        num_pairs = 5

        for i in range(len(peaks)):

            for j in range(
                i + 1,
                min(i + num_pairs + 1, len(peaks))
            ):

                dt = peaks[j][1] - peaks[i][1]

                if dt > 0:

                    hash_count += 1

        library_data.append({

            "name": name,

            "hashes": hash_count,

            "peaks": song["peaks"],

            "frequency": song["frequency"],

            "time": song["time"]

        })

    return library_data