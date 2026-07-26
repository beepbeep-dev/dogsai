from __future__ import annotations

import numpy as np
import pytest

from dogsai.audio import (
    F0_MAX,
    NOT_THE_DOG,
    AudioError,
    classify_event,
    detect_events,
    estimate_f0,
    frame_signal,
    has_audio,
    load_audio,
    magnitude_spectrogram,
    read_audio,
    rms_db,
    spectral_centroid,
    spectral_flatness,
    speech_likeness,
)

SR = 16000


def tone(freq: float, seconds: float, sr: int = SR, harmonics: int = 4, amp: float = 0.5):
    """A harmonic tone — stands in for a tonal vocalisation."""
    t = np.arange(int(seconds * sr)) / sr
    signal = np.zeros_like(t)
    for h in range(1, harmonics + 1):
        signal += (amp / h) * np.sin(2 * np.pi * freq * h * t)
    return (signal / max(1e-9, np.abs(signal).max()) * amp).astype(np.float32)


def noise(seconds: float, sr: int = SR, amp: float = 0.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, amp / 3, int(seconds * sr))).astype(np.float32)


def silence(seconds: float, sr: int = SR, amp: float = 1e-4, seed: int = 1):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, amp, int(seconds * sr))).astype(np.float32)


class TestFraming:
    def test_frames_have_the_requested_shape(self):
        frames = frame_signal(np.zeros(16000, dtype=np.float32))
        assert frames.shape[1] == 512
        assert frames.shape[0] > 1

    def test_short_input_is_padded_not_rejected(self):
        frames = frame_signal(np.zeros(100, dtype=np.float32))
        assert frames.shape == (1, 512)

    def test_rms_db_orders_loud_above_quiet(self):
        loud = rms_db(frame_signal(tone(400, 0.2, amp=0.8)))
        quiet = rms_db(frame_signal(tone(400, 0.2, amp=0.01)))
        assert loud.max() > quiet.max() + 20


class TestSpectralFeatures:
    def test_flatness_separates_tone_from_noise(self):
        """The tonality axis the bark literature keys on."""
        tonal = spectral_flatness(magnitude_spectrogram(frame_signal(tone(500, 0.3))))
        noisy = spectral_flatness(magnitude_spectrogram(frame_signal(noise(0.3))))
        assert np.median(tonal) < 0.1
        assert np.median(noisy) > np.median(tonal) * 3

    def test_centroid_tracks_pitch(self):
        low = spectral_centroid(magnitude_spectrogram(frame_signal(tone(200, 0.3, harmonics=1))), SR)
        high = spectral_centroid(magnitude_spectrogram(frame_signal(tone(2000, 0.3, harmonics=1))), SR)
        assert np.median(high) > np.median(low) * 3


class TestF0:
    @pytest.mark.parametrize("freq", [120, 250, 440, 900, 1500])
    def test_recovers_a_known_fundamental(self, freq):
        f0, periodicity = estimate_f0(tone(freq, 0.25), SR)
        assert f0 == pytest.approx(freq, rel=0.06)
        assert periodicity > 0.5

    def test_reports_the_fundamental_not_a_harmonic(self):
        """Naive spectral peak-picking returns 2x here; autocorrelation must not."""
        t = np.arange(int(0.25 * SR)) / SR
        # Second harmonic deliberately louder than the fundamental.
        signal = (0.2 * np.sin(2 * np.pi * 300 * t) + 0.8 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)
        f0, _ = estimate_f0(signal, SR)
        assert f0 == pytest.approx(300, rel=0.08)

    def test_noise_has_low_periodicity(self):
        _, periodicity = estimate_f0(noise(0.25), SR)
        assert periodicity < 0.4

    def test_silence_returns_zero(self):
        f0, periodicity = estimate_f0(np.zeros(4000, dtype=np.float32), SR)
        assert f0 == 0.0 and periodicity == 0.0

    def test_stays_inside_the_search_range(self):
        f0, _ = estimate_f0(tone(6000, 0.2, harmonics=1), SR)
        assert f0 == 0.0 or f0 <= F0_MAX * 1.01


class TestDetectEvents:
    def test_finds_a_burst_between_silences(self):
        audio = np.concatenate([silence(0.5), tone(500, 0.3, amp=0.6), silence(0.5)])
        events = detect_events(audio, SR)
        assert len(events) == 1
        start, end = events[0]
        assert start == pytest.approx(0.5, abs=0.12)
        assert end == pytest.approx(0.8, abs=0.15)

    def test_finds_multiple_separated_bursts(self):
        audio = np.concatenate([
            silence(0.3), tone(500, 0.2, amp=0.6),
            silence(0.5), tone(500, 0.2, amp=0.6), silence(0.3),
        ])
        assert len(detect_events(audio, SR)) == 2

    def test_merges_bursts_split_by_a_tiny_gap(self):
        """Hysteresis and gap-merging stop one bark becoming three."""
        audio = np.concatenate([
            silence(0.3), tone(500, 0.15, amp=0.6),
            silence(0.02), tone(500, 0.15, amp=0.6), silence(0.3),
        ])
        assert len(detect_events(audio, SR)) == 1

    def test_pure_silence_yields_nothing(self):
        assert detect_events(silence(2.0), SR) == []

    def test_steady_noise_yields_nothing(self):
        """No dynamic range means nothing to segment — must not return one big event."""
        assert detect_events(noise(2.0, amp=0.3), SR) == []

    def test_very_short_blips_are_dropped(self):
        audio = np.concatenate([silence(0.4), tone(500, 0.01, amp=0.6), silence(0.4)])
        assert detect_events(audio, SR, min_duration=0.05) == []

    def test_thresholds_adapt_to_recording_gain(self):
        """The same event at 1/10th the gain must still be found."""
        loud = np.concatenate([silence(0.4, amp=1e-3), tone(500, 0.3, amp=0.6), silence(0.4, amp=1e-3)])
        quiet = loud * 0.1
        assert len(detect_events(loud, SR)) == len(detect_events(quiet, SR)) == 1


class TestClassification:
    def _classify(self, audio, start=0.0, end=None, rate=0.0):
        end = end if end is not None else len(audio) / SR
        return classify_event(audio, SR, start, end, bark_rate=rate)

    def test_low_noisy_sustained_reads_as_growl(self):
        rng = np.random.default_rng(0)
        base = tone(180, 0.6, harmonics=6, amp=0.5)
        rough = (base + rng.normal(0, 0.12, len(base))).astype(np.float32)
        assert self._classify(rough).kind == "growl"

    def test_high_tonal_sustained_reads_as_whine(self):
        assert self._classify(tone(900, 0.4, harmonics=3)).kind == "whine"

    def test_very_short_very_high_reads_as_yelp(self):
        event = self._classify(tone(1200, 0.1, harmonics=3))
        assert event.kind == "yelp"
        assert event.concerning

    def test_speech_like_audio_is_not_attributed_to_the_dog(self):
        """People talk in dog videos; speech must not become a howl or growl."""
        t = np.arange(int(1.0 * SR)) / SR
        f0 = 150 + 25 * np.sin(2 * np.pi * 3 * t)          # prosodic wobble
        phase = 2 * np.pi * np.cumsum(f0) / SR
        signal = np.zeros_like(t)
        for h, gain in ((1, 0.3), (3, 0.6), (5, 0.9), (8, 0.7), (12, 0.4)):
            signal += gain * np.sin(h * phase)             # formant-heavy
        signal = (signal / np.abs(signal).max() * 0.5).astype(np.float32)
        event = self._classify(signal)
        assert event.kind == "speech"
        assert event.kind in NOT_THE_DOG

    def test_every_kind_carries_an_interpretation(self):
        for audio in (tone(900, 0.4), tone(1200, 0.1), tone(300, 0.3), noise(0.8)):
            event = self._classify(audio)
            assert event.meaning
            assert -1.0 <= event.valence <= 1.0
            assert 0.0 <= event.arousal <= 1.0
            assert 0.0 <= event.confidence <= 1.0

    def test_rapid_repetition_pushes_a_bark_towards_alarm(self):
        bark = tone(350, 0.15, harmonics=5)
        isolated = self._classify(bark, rate=0.0)
        rapid = self._classify(bark, rate=5.0)
        assert rapid.arousal >= isolated.arousal

    def test_quiet_events_get_lower_confidence(self):
        loud = self._classify(tone(900, 0.4, amp=0.6))
        quiet = self._classify(tone(900, 0.4, amp=0.002))
        assert quiet.confidence < loud.confidence

    def test_event_dict_has_no_translation_claim(self):
        payload = self._classify(tone(900, 0.4)).to_dict()
        assert "likely_context" in payload
        assert "translation" not in payload


class TestSpeechLikeness:
    def test_short_events_are_never_speech(self):
        spectrum = magnitude_spectrogram(frame_signal(tone(150, 0.1)))
        assert speech_likeness(spectrum, SR, 150, 0.6, 0.1, 0.2) == 0.0

    def test_high_frequency_energy_argues_against_speech(self):
        bright = magnitude_spectrogram(frame_signal(tone(150, 1.0, harmonics=40)))
        dull = magnitude_spectrogram(frame_signal(tone(150, 1.0, harmonics=6)))
        assert speech_likeness(bright, SR, 150, 0.6, 1.0, 0.2) <= speech_likeness(
            dull, SR, 150, 0.6, 1.0, 0.2
        )

    def test_out_of_band_pitch_scores_lower(self):
        spectrum = magnitude_spectrogram(frame_signal(tone(150, 1.0)))
        in_band = speech_likeness(spectrum, SR, 150, 0.6, 1.0, 0.2)
        out_of_band = speech_likeness(spectrum, SR, 1500, 0.6, 1.0, 0.2)
        assert out_of_band < in_band


class TestLoadAudio:
    def test_reads_a_real_file(self, session_video):
        video, _ = session_video
        if not has_audio(video):
            pytest.skip("synthetic fixture has no audio track")
        audio, sr = load_audio(video)
        assert sr == 16000 and len(audio) > 0

    def test_silent_video_has_no_audio_stream(self, session_video):
        video, _ = session_video
        assert has_audio(video) is False  # synth writes video only

    def test_missing_file_raises(self):
        with pytest.raises(AudioError):
            load_audio("/nonexistent/clip.mp4")

    def test_read_audio_on_a_silent_video_degrades_gracefully(self, session_video):
        video, _ = session_video
        reading = read_audio(video)
        assert reading.events == []
        assert reading.confidence == 0.0
        assert reading.notes  # explains why there is nothing


class TestAudioReading:
    def test_reports_no_vocalisations_clearly(self, tmp_path):
        from dogsai.audio import AudioReading

        reading = AudioReading(duration=5.0)
        assert "no vocalisations" in reading.render()
        assert reading.to_dict()["n_events"] == 0

    def test_dict_carries_the_disclaimer(self):
        from dogsai.audio import AudioReading, VocalEvent

        reading = AudioReading(
            duration=3.0,
            events=[VocalEvent(0.0, 0.2, "bark", 400, 0.6, 0.1, 900, -20, 0.5, "x")],
        )
        payload = reading.to_dict()
        assert "not language" in payload["disclaimer"]
        assert payload["n_events"] == 1

    def test_dog_events_excludes_speech_and_noise(self):
        from dogsai.audio import AudioReading, VocalEvent

        reading = AudioReading(duration=3.0, events=[
            VocalEvent(0.0, 0.2, "bark", 400, 0.6, 0.1, 900, -20, 0.5, "x"),
            VocalEvent(1.0, 2.0, "speech", 150, 0.6, 0.1, 900, -20, 0.5, "y"),
            VocalEvent(2.0, 2.2, "sound", 0, 0.0, 0.9, 900, -20, 0.2, "z"),
        ])
        assert [e.kind for e in reading.dog_events] == ["bark"]

    def test_concerning_events_are_surfaced(self):
        from dogsai.audio import AudioReading, VocalEvent

        reading = AudioReading(duration=3.0, events=[
            VocalEvent(0.0, 0.1, "yelp", 1200, 0.7, 0.05, 2000, -10, 0.6, "ow", concerning=True),
        ])
        assert reading.concerning
        assert "attention" in reading.render()
