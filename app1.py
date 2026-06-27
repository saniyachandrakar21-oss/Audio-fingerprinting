import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp/"
os.environ["NUMBA_DISABLE_JIT"] = "1"
from fingerprint_engine import (
    build_database,
    generate_query_hashes,
    song_identifier,
    plot_offset_histogram,
    get_library_data
)
import streamlit as st
import pandas as pd
import numpy as np

import soundfile as sf
from scipy.signal import spectrogram as scipy_spectrogram

import matplotlib.pyplot as plt
import tempfile
import os
import pickle

@st.cache_resource
def load_database():
    with open("database.pkl", "rb") as f:
        return pickle.load(f)

database = load_database()

@st.cache_data
def load_library_names():
    song_names = set()
    for matches in database.values():
        for song_name, _ in matches:
            song_names.add(song_name)
    return sorted(list(song_names))
st.set_page_config(
    page_title="EE200 Audio Fingerprinting",
    layout="wide"
)
@st.cache_data
def load_library_data():
    with open("library_data.pkl", "rb") as f:
        return pickle.load(f)

library_data = load_library_data()

st.title("🎵 EE200: Audio Fingerprinting")

st.caption(
    "Index a library of songs as spectrogram fingerprints, then identify any short clip against it."
)

tab1, tab2, tab3 = st.tabs(
    ["📚 Library", "🎯 Identify", "📦 Batch"]
)
with tab1:
    st.subheader("Library")
    st.success(f"{len(library_data)} songs indexed in database")

    cols = st.columns(4)
    colors = ["#00FFFF", "#FFD700", "#FF69B4", "#AD8CFF", "#7CFC00"]

    for idx, song in enumerate(library_data):
        with cols[idx % 4]:
            with st.container(border=True):
                song_name = os.path.splitext(song["name"])[0]

                st.markdown(
                    f"""
                    <div style="
                        height:70px;
                        overflow:hidden;
                        text-align:center;
                    ">
                        <b>{song_name}</b><br>
                        {song['hashes']:,} hashes
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                fig, ax = plt.subplots(figsize=(3, 2))
                ax.set_facecolor("black")
                peaks = np.array(song["peaks"])
                freq = np.array(song["frequency"])
                time_vals = np.array(song["time"])
                ax.scatter(
                    time_vals[peaks[:,1]],
                    freq[peaks[:,0]],
                    s=0.1,
                    c=colors[idx % len(colors)]
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")
                for spine in ax.spines.values():
                    spine.set_visible(False)

                st.pyplot(fig)
with tab2:

    st.subheader("Identify a Clip")

    uploaded_file = st.file_uploader(
        "Upload Audio",
        type=["wav","mp3"],
        key="identify"
    )

    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Try"):
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp3"
            ) as tmp:

                tmp.write(uploaded_file.getvalue())

                query_path = tmp.name

            audio, fs = sf.read(query_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
                y = audio
                sr = fs

            query_hashes, constellation_data = generate_query_hashes(
                query_path
            )
            matched_song, offset_count = song_identifier(
                query_hashes[0]["hashes"],
                database
            )

            if matched_song:

                st.markdown(f"""
                ## 🎯 MATCH FOUND

                ### {matched_song}
                """)

                st.markdown(
                     """
                ## STEP 1 : SPECTROGRAM GENERATION

                The uploaded audio clip is converted into a spectrogram, providing a time-frequency representation of the signal.
                """)

                fig, ax = plt.subplots(figsize=(4, 3))

                # ADD THESE LINES:
                from scipy.signal import spectrogram as scipy_spectrogram
                freq_s, time_s, Sxx = scipy_spectrogram(audio, fs, nperseg=512)
                Sxx_db = 10 * np.log10(Sxx + 1e-10)
                ax.pcolormesh(time_s, freq_s, Sxx_db, shading="gouraud")
                ax.set_ylim(0, 5000)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Frequency (Hz)")

                left, center, right = st.columns([1, 2, 1])
                with center:
                    st.pyplot(fig)
                plt.close(fig)

                st.markdown(
                    """
                ## STEP 2 : PEAK DETECTION & FINGERPRINT GENERATION

                Prominent spectral peaks are extracted to form a constellation map, and peak pairs are converted into fingerprint hashes.
                """
                )
                col1, col2 = st.columns(2)
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(6, 4))

                    peaks = np.array(constellation_data[0]["peaks"])
                    frequency = np.array(constellation_data[0]["frequency"])
                    time_vals = np.array(constellation_data[0]["time"])
                    spec_in_db = np.array(constellation_data[0]["spec_in_db"])

                    freq_s, time_s, Sxx = scipy_spectrogram(audio, fs, nperseg=512)
                    Sxx_db = 10 * np.log10(Sxx + 1e-10)

                    ax1.pcolormesh(time_s, freq_s, Sxx_db, shading="gouraud")
                    ax1.scatter(
                        time_vals[peaks[:, 1]],
                        frequency[peaks[:, 0]],
                        c="red",
                        s=4
                    )
                    ax1.set_title("Spectrogram with Peaks")
                    ax1.set_ylim(0, 5000)
                    ax1.set_xlabel("Time (s)")
                    ax1.set_ylabel("Frequency (Hz)")
                    st.pyplot(fig1)
                    plt.close(fig1)

                with col2:
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    ax2.scatter(
                        time_vals[peaks[:, 1]],
                        frequency[peaks[:, 0]],
                        c="red",
                        s=4
                    )
                    ax2.set_title("Constellation Map")
                    ax2.set_xlabel("Time (s)")
                    ax2.set_ylabel("Frequency (Hz)")
                    st.pyplot(fig2)
                    plt.close(fig2)

                st.divider()
                st.metric(
                    "Fingerprint Hashes Generated",
                    len(query_hashes[0]["hashes"])
                )
                st.markdown(
                """
                # STEP 3 : DATABASE MATCHING

                The generated fingerprint hashes are compared with the database, and an offset histogram is used to identify the song with the strongest alignment.
                """)

                hist_fig = plot_offset_histogram(
                    matched_song,
                    offset_count
                )
                if hist_fig:
                    st.pyplot(hist_fig)
                    plt.close(hist_fig)

                st.success(
                    f"Matched Song : {matched_song}"
                )
            
        
with tab3:

    st.subheader("Identify Many Clips")

    files = st.file_uploader(
        "Upload Query Clips",
        type=["wav","mp3"],
        accept_multiple_files=True,
        key="batch"
    )

    if files:

        st.write(
            f"{len(files)} file(s) selected"
        )

        if st.button("Run Batch"):

            if len(files) > 5:

                st.error("Maximum 5 files allowed")

            else:
                results = []
                for uploaded_file in files:

                    st.divider()

                    st.header(uploaded_file.name)

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".mp3"
                    ) as tmp:

                        tmp.write(
                            uploaded_file.getvalue()
                        )

                        query_path = tmp.name

                    audio, fs = sf.read(query_path)
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)

                    query_hashes, constellation_data = generate_query_hashes(
                        query_path
                    )

                    matched_song, offset_count = song_identifier(
                        query_hashes[0]["hashes"],
                        database
                    )

                    if matched_song:

                        st.success(
                            f"Matched Song : {matched_song}"
                        )
                        prediction = os.path.splitext(
                            matched_song
                        )[0]

                        results.append({
                            "filename": uploaded_file.name,
                            "prediction": prediction
                        })

                        # =====================
                        # STEP 1
                        # =====================

                        st.markdown(f"""
                        ## 🎯 MATCH FOUND

                        ### {matched_song}
                        """)

                        st.markdown(
                            """
                        ## STEP 1 : SPECTROGRAM GENERATION

                        The uploaded audio clip is converted into a spectrogram, providing a time-frequency representation of the signal.
                        """)

                        fig, ax = plt.subplots(figsize=(4, 3))

                        # ADD THESE LINES:
                        from scipy.signal import spectrogram as scipy_spectrogram
                        freq_s, time_s, Sxx = scipy_spectrogram(audio, fs, nperseg=512)
                        Sxx_db = 10 * np.log10(Sxx + 1e-10)
                        ax.pcolormesh(time_s, freq_s, Sxx_db, shading="gouraud")
                        ax.set_ylim(0, 5000)
                        ax.set_xlabel("Time (s)")
                        ax.set_ylabel("Frequency (Hz)")

                        left, center, right = st.columns([1, 2, 1])
                        with center:
                            st.pyplot(fig)
                        plt.close(fig)

                        st.markdown(
                            """
                        ## STEP 2 : PEAK DETECTION & FINGERPRINT GENERATION

                        Prominent spectral peaks are extracted to form a constellation map, and peak pairs are converted into fingerprint hashes.
                        """
                        )
                        col1, col2 = st.columns(2)
                        with col1:
                            fig1, ax1 = plt.subplots(figsize=(6, 4))

                            peaks = np.array(constellation_data[0]["peaks"])
                            frequency = np.array(constellation_data[0]["frequency"])
                            time_vals = np.array(constellation_data[0]["time"])
                            spec_in_db = np.array(constellation_data[0]["spec_in_db"])

                            freq_s, time_s, Sxx = scipy_spectrogram(audio, fs, nperseg=512)
                            Sxx_db = 10 * np.log10(Sxx + 1e-10)

                            ax1.pcolormesh(time_s, freq_s, Sxx_db, shading="gouraud")
                            ax1.scatter(
                                time_vals[peaks[:, 1]],
                                frequency[peaks[:, 0]],
                                c="red",
                                s=4
                            )
                            ax1.set_title("Spectrogram with Peaks")
                            ax1.set_ylim(0, 5000)
                            ax1.set_xlabel("Time (s)")
                            ax1.set_ylabel("Frequency (Hz)")
                            st.pyplot(fig1)
                            plt.close(fig1)

                        with col2:
                            fig2, ax2 = plt.subplots(figsize=(6, 4))
                            ax2.scatter(
                                time_vals[peaks[:, 1]],
                                frequency[peaks[:, 0]],
                                c="red",
                                s=4
                            )
                            ax2.set_title("Constellation Map")
                            ax2.set_xlabel("Time (s)")
                            ax2.set_ylabel("Frequency (Hz)")
                            st.pyplot(fig2)
                            plt.close(fig2)

                        st.divider()
                        st.metric(
                            "Fingerprint Hashes Generated",
                            len(query_hashes[0]["hashes"])
                        )
                        st.markdown(
                        """
                        # STEP 3 : DATABASE MATCHING

                        The generated fingerprint hashes are compared with the database, and an offset histogram is used to identify the song with the strongest alignment.
                        """)

                        hist_fig = plot_offset_histogram(
                            matched_song,
                            offset_count
                        )
                        if hist_fig:
                            st.pyplot(hist_fig)
                            plt.close(hist_fig)

                        st.success(
                            f"Final Match : {matched_song}"
                        )

                    else:

                        st.error(
                            f"No Match Found for {uploaded_file.name}"
                        )

                        results.append({
                            "filename": uploaded_file.name,
                            "prediction": "No Match"
                        })
                st.divider()

                st.subheader("Batch Results")

                results_df = pd.DataFrame(results)

                st.dataframe(
                    results_df,
                    use_container_width=True
                )

                csv = results_df.to_csv(
                    index=False
                )

                st.download_button(
                    "Download results.csv",
                    csv,
                    file_name="results.csv",
                    mime="text/csv"
                )