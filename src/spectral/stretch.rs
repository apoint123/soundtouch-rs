use std::f64::consts::LN_2;

use crate::{
    error::{
        Result,
        SoundTouchError,
    },
    fifo::FifoSampleBuffer,
    spectral::{
        stft::StftOperator,
        vocoder::PhaseVocoder,
        windows::{
            ApproximateConfinedGaussianWindow,
            KaiserWindow,
            SpectralWindowShape,
            force_perfect_reconstruction,
        },
    },
};

/// Pre-configured parameter sets for `SpectralStretch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpectralPreset {
    /// Default high-quality preset: ~120ms window, ~30ms hop step.
    #[default]
    Default,

    /// Cheaper performance-focused preset: ~100ms window, ~40ms hop step.
    Cheaper,

    /// Custom manual parameters in sample counts.
    Custom {
        block_samples: usize,
        interval_samples: usize,
    },
}

/// A builder to configure and instantiate a `SpectralStretch` processor.
#[derive(Debug, Clone)]
pub struct SpectralStretchBuilder {
    channels: usize,
    sample_rate: usize,
    tempo: f64,
    pitch: f64,
    formant_factor: f64,
    formant_compensate_pitch: bool,
    preset: SpectralPreset,
    window_shape: SpectralWindowShape,
}

impl SpectralStretchBuilder {
    #[must_use]
    pub const fn new(channels: usize, sample_rate: usize) -> Self {
        Self {
            channels,
            sample_rate,
            tempo: 1.0,
            pitch: 1.0,
            formant_factor: 1.0,
            formant_compensate_pitch: false,
            preset: SpectralPreset::Default,
            window_shape: SpectralWindowShape::Kaiser,
        }
    }

    #[must_use]
    pub const fn preset(mut self, preset: SpectralPreset) -> Self {
        self.preset = preset;
        self
    }

    #[must_use]
    pub const fn window_shape(mut self, window_shape: SpectralWindowShape) -> Self {
        self.window_shape = window_shape;
        self
    }

    #[must_use]
    pub const fn tempo(mut self, tempo: f64) -> Self {
        self.tempo = tempo;
        self
    }

    #[must_use]
    pub const fn pitch(mut self, pitch: f64) -> Self {
        self.pitch = pitch;
        self
    }

    #[must_use]
    pub fn pitch_semi_tones(self, pitch_semi_tones: f64) -> Self {
        let pitch_octaves = pitch_semi_tones / 12.0;
        let pitch_ratio = f64::exp(LN_2 * pitch_octaves);
        self.pitch(pitch_ratio)
    }

    #[must_use]
    pub const fn formant_factor(mut self, factor: f64, compensate_pitch: bool) -> Self {
        self.formant_factor = factor;
        self.formant_compensate_pitch = compensate_pitch;
        self
    }

    /// Builds a new `SpectralStretch` instance.
    pub fn build(self) -> Result<SpectralStretch> {
        SpectralStretch::internal_new(&self)
    }
}

/// A high-performance Phase Vocoder STFT time-stretching and pitch-shifting engine.
pub struct SpectralStretch {
    channels: usize,
    sample_rate: usize,
    block_samples: usize,
    interval_samples: usize,
    tempo: f64,
    pitch: f64,

    stft: StftOperator,
    vocoder: PhaseVocoder,
    input_buffer: FifoSampleBuffer,
    output_buffer: FifoSampleBuffer,
    ola_buffer: Vec<Vec<f32>>,
    analysis_window: Vec<f32>,
    synthesis_window: Vec<f32>,
    spectrum_scratch: Vec<realfft::num_complex::Complex<f32>>,

    /// Input frames advanced by the previous block; `0` means there is no reusable
    /// analysis from a previous block.
    ///
    /// Tells whether `Band::input` already sits exactly where this block needs its
    /// `prev_input`.
    last_input_hop: usize,

    /// Accumulated fractional input position to eliminate tempo drift and provide
    /// global fractional hop scheduling.
    accumulated_input_pos: f64,

    /// Total integer input frames consumed.
    consumed_input_frames: usize,
}

impl SpectralStretch {
    /// Creates a new `SpectralStretchBuilder`.
    #[must_use]
    pub const fn builder(channels: usize, sample_rate: usize) -> SpectralStretchBuilder {
        SpectralStretchBuilder::new(channels, sample_rate)
    }

    fn internal_new(builder: &SpectralStretchBuilder) -> Result<Self> {
        if builder.channels == 0 {
            return Err(SoundTouchError::InvalidChannels {
                provided: builder.channels,
            });
        }
        if builder.sample_rate == 0 {
            return Err(SoundTouchError::InvalidSampleRate {
                provided: builder.sample_rate,
            });
        }

        let (block_samples, interval_samples) = match builder.preset {
            SpectralPreset::Default => (
                (builder.sample_rate as f64 * 0.12) as usize,
                (builder.sample_rate as f64 * 0.03) as usize,
            ),
            SpectralPreset::Cheaper => (
                (builder.sample_rate as f64 * 0.10) as usize,
                (builder.sample_rate as f64 * 0.04) as usize,
            ),
            SpectralPreset::Custom {
                block_samples,
                interval_samples,
            } => (block_samples, interval_samples),
        };

        let block_samples = block_samples.max(64);
        let interval_samples = interval_samples.max(16).min(block_samples / 2);

        let stft = StftOperator::new(block_samples, block_samples);
        let bands = stft.bands();
        let vocoder = PhaseVocoder::new(bands, builder.channels);

        let max_frame_limit = builder.sample_rate * 10;
        let input_buffer = FifoSampleBuffer::new(builder.channels, max_frame_limit)?;
        let output_buffer = FifoSampleBuffer::new(builder.channels, max_frame_limit)?;

        let mut analysis_window = vec![0.0f32; block_samples];
        let mut synthesis_window = vec![0.0f32; block_samples];
        let bandwidth = block_samples as f64 / interval_samples as f64;

        match builder.window_shape {
            SpectralWindowShape::Kaiser => {
                let kaiser = KaiserWindow::with_bandwidth(bandwidth, true);
                kaiser.fill(&mut synthesis_window, 0.0, true);
            }
            SpectralWindowShape::ConfinedGaussian => {
                let acg = ApproximateConfinedGaussianWindow::with_bandwidth(bandwidth);
                acg.fill(&mut synthesis_window, 0.0, true);
            }
        }
        force_perfect_reconstruction(&mut synthesis_window, interval_samples);
        analysis_window.copy_from_slice(&synthesis_window);

        let ola_buffer = vec![vec![0.0f32; block_samples]; builder.channels];
        let spectrum_scratch = vec![realfft::num_complex::Complex::new(0.0, 0.0); bands];

        let mut engine = Self {
            channels: builder.channels,
            sample_rate: builder.sample_rate,
            block_samples,
            interval_samples,
            tempo: builder.tempo,
            pitch: builder.pitch,
            stft,
            vocoder,
            input_buffer,
            output_buffer,
            ola_buffer,
            analysis_window,
            synthesis_window,
            spectrum_scratch,
            last_input_hop: 0,
            accumulated_input_pos: 0.0,
            consumed_input_frames: 0,
        };

        engine.set_tempo(builder.tempo);
        engine.set_pitch(builder.pitch);
        engine.set_formant_factor(builder.formant_factor, builder.formant_compensate_pitch);

        Ok(engine)
    }

    /// Returns the active number of audio channels.
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the base sample rate.
    #[must_use]
    pub const fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Analysis/synthesis window length, in frames per channel.
    #[must_use]
    pub const fn window_frames(&self) -> usize {
        self.block_samples
    }

    /// Number of output frames produced by one analysis step.
    #[must_use]
    pub const fn hop_frames(&self) -> usize {
        self.interval_samples
    }

    /// Additional input needed to run the next analysis step.
    #[must_use]
    pub const fn input_frames_needed(&self) -> usize {
        (self.block_samples + self.interval_samples).saturating_sub(self.input_buffer.frames())
    }

    /// Preallocates both FIFOs without processing or discarding audio.
    ///
    /// Call before entering a realtime thread, with the maximum queued input and
    /// output frame counts the caller will allow.
    pub fn reserve(&mut self, input_frames: usize, output_frames: usize) -> Result<()> {
        self.input_buffer.reserve(input_frames)?;
        self.output_buffer.reserve(output_frames)
    }

    /// Dynamically sets the time-stretching tempo factor.
    pub const fn set_tempo(&mut self, tempo: f64) {
        self.tempo = tempo.clamp(0.1, 10.0);
    }

    /// Dynamically sets the pitch-shifting multiplier ratio.
    pub fn set_pitch(&mut self, pitch: f64) {
        self.pitch = pitch.clamp(0.1, 10.0);
        self.vocoder.set_transpose_factor(self.pitch as f32);
    }

    /// Sets the pitch-shifting factor in semitones (e.g. `12.0` = +1 octave).
    pub fn set_pitch_semi_tones(&mut self, semitones: f64) {
        let pitch_ratio = f64::exp(LN_2 * (semitones / 12.0));
        self.set_pitch(pitch_ratio);
    }

    /// Dynamically sets the formant multiplier factor and pitch compensation.
    pub const fn set_formant_factor(&mut self, factor: f64, compensate_pitch: bool) {
        self.vocoder
            .set_formant_factor(factor as f32, compensate_pitch);
    }

    /// Pushes new planar audio samples into the streaming pipeline.
    pub fn put_samples(&mut self, channels_data: &[impl AsRef<[f32]>]) -> Result<()> {
        self.input_buffer.put_samples(channels_data)?;
        self.process_pipeline()
    }

    /// Pulls processed planar audio samples from the streaming pipeline.
    pub fn receive_samples(&mut self, output: &mut [impl AsMut<[f32]>]) -> Result<usize> {
        self.output_buffer.receive_samples(output)
    }

    /// Returns the number of frames available in the output buffer.
    #[must_use]
    pub const fn num_samples(&self) -> usize {
        self.output_buffer.frames()
    }

    /// Flushes all internal buffers and resets DSP states.
    pub fn clear(&mut self) -> Result<()> {
        self.input_buffer.clear();
        self.output_buffer.clear();
        for ch_ola in &mut self.ola_buffer {
            ch_ola.fill(0.0);
        }
        // The input stream is interrupted, so the previous analysis is no longer
        // adjacent to the incoming data.
        self.last_input_hop = 0;
        self.accumulated_input_pos = 0.0;
        self.consumed_input_frames = 0;
        self.vocoder.clear();
        Ok(())
    }

    /// Flushes remaining queued audio out of the processing pipeline.
    pub fn flush(&mut self) -> Result<()> {
        let silence_frames = self.block_samples * 4;
        let silence_vecs = vec![vec![0.0f32; silence_frames]; self.channels];
        self.put_samples(&silence_vecs)
    }

    fn process_pipeline(&mut self) -> Result<()> {
        let required_input = self.block_samples + self.interval_samples;
        while self.input_buffer.frames() >= required_input {
            let next_input_pos =
                (self.interval_samples as f64).mul_add(self.tempo, self.accumulated_input_pos);
            let next_input_int = next_input_pos.round() as usize;
            let input_hop = next_input_int
                .saturating_sub(self.consumed_input_frames)
                .max(1);
            let time_factor = (self.interval_samples as f32) / (input_hop as f32);

            self.accumulated_input_pos = next_input_pos;
            self.consumed_input_frames += input_hop;

            // When the previous block was analysed exactly one interval before the
            // current analysis position (`input_hop == interval_samples`),
            // its spectrum matches the `prev_input` required for this step, avoiding
            // redundant forward STFT analysis.
            let reuse_prev_analysis = self.last_input_hop == self.interval_samples;

            if reuse_prev_analysis {
                // Must carry over every channel before `input` is overwritten below.
                for c in 0..self.channels {
                    for b in 0..self.stft.bands() {
                        let band = self.vocoder.band_mut(c, b);
                        band.prev_input = band.input;
                    }
                }
            }

            // Re-analyse input frames
            for (c, ch_slice) in self.input_buffer.current_data_iter().enumerate() {
                if !reuse_prev_analysis {
                    let prev_slice = &ch_slice[..self.block_samples];
                    self.stft.analyse(
                        prev_slice,
                        &self.analysis_window,
                        &mut self.spectrum_scratch,
                    );
                    for b in 0..self.stft.bands() {
                        self.vocoder.band_mut(c, b).prev_input = self.spectrum_scratch[b];
                    }
                }

                let curr_slice =
                    &ch_slice[self.interval_samples..self.interval_samples + self.block_samples];
                self.stft.analyse(
                    curr_slice,
                    &self.analysis_window,
                    &mut self.spectrum_scratch,
                );
                for b in 0..self.stft.bands() {
                    self.vocoder.band_mut(c, b).input = self.spectrum_scratch[b];
                }
            }

            self.vocoder.process_spectrum(
                self.interval_samples,
                self.stft.fft_samples(),
                time_factor,
            );

            // Synthesise spectrum to OLA buffer
            //
            // `synthesise` already accumulates (`+=`), so it can write straight into
            // the OLA buffer; no need to allocate a temporary frame per block.
            for c in 0..self.channels {
                for b in 0..self.stft.bands() {
                    self.spectrum_scratch[b] = self.vocoder.band(c, b).output;
                }

                self.stft.synthesise(
                    &mut self.spectrum_scratch,
                    &self.synthesis_window,
                    &mut self.ola_buffer[c],
                );
            }

            // Push one interval slice from OLA buffer into output buffer
            //
            // Write directly into the tail of the output FIFO and shift the OLA buffer
            // left by one hop, avoiding a temporary Vec and the extra move that
            // `drain` + `resize` would cost.
            let hop = self.interval_samples;
            for (ch_ola, tail) in self
                .ola_buffer
                .iter_mut()
                .zip(self.output_buffer.tail_iter_mut(hop)?)
            {
                tail[..hop].copy_from_slice(&ch_ola[..hop]);
                ch_ola.copy_within(hop.., 0);
                ch_ola[self.block_samples - hop..].fill(0.0);
            }
            self.output_buffer.commit_written_frames(hop);

            // Advance input buffer by input hop
            self.input_buffer.receive_frames(input_hop);
            self.last_input_hop = input_hop;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_stretch_pipeline() {
        let channels = 2;
        let sample_rate = 44100;

        let mut stretch = SpectralStretch::builder(channels, sample_rate)
            .preset(SpectralPreset::Custom {
                block_samples: 256,
                interval_samples: 64,
            })
            .tempo(1.25)
            .pitch_semi_tones(2.0)
            .build()
            .unwrap();

        let input_pcm = vec![vec![0.1f32; 2048]; channels];
        stretch.put_samples(&input_pcm).unwrap();

        assert!(stretch.num_samples() > 0);

        let mut out_pcm = vec![vec![0.0f32; 256]; channels];
        let rx = stretch.receive_samples(&mut out_pcm).unwrap();
        assert_eq!(rx, 256);
    }

    #[test]
    fn test_streaming_chunk_size_independence() {
        let channels = 1;
        let sample_rate = 44100;

        let mut stretch1 = SpectralStretch::builder(channels, sample_rate)
            .tempo(0.8)
            .build()
            .unwrap();
        let mut stretch2 = SpectralStretch::builder(channels, sample_rate)
            .tempo(0.8)
            .build()
            .unwrap();

        let input_full = vec![vec![0.2f32; 8192]; channels];

        // Engine 1: Push full 8192 frames at once
        stretch1.put_samples(&input_full).unwrap();

        // Engine 2: Push in small 256-frame chunks
        for _chunk_idx in 0..32 {
            let chunk = vec![vec![0.2f32; 256]; channels];
            let _ = stretch2.put_samples(&chunk);
        }

        assert_eq!(
            stretch1.num_samples(),
            stretch2.num_samples(),
            "Chunking must not alter total output sample count"
        );
    }

    #[test]
    fn test_spectral_stretch_slow_tempo_0_2x() {
        let channels = 2;
        let sample_rate = 44100;

        let mut stretch = SpectralStretch::builder(channels, sample_rate)
            .preset(SpectralPreset::Custom {
                block_samples: 256,
                interval_samples: 64,
            })
            .tempo(0.2)
            .build()
            .unwrap();

        // Feed impulsive transient: sharp spike followed by silence
        let mut input_pcm = vec![vec![0.0f32; 2048]; channels];
        for ch_buf in &mut input_pcm {
            ch_buf[100] = 1.0;
        }

        stretch.put_samples(&input_pcm).unwrap();
        stretch.flush().unwrap();

        let num_out = stretch.num_samples();
        assert!(num_out > 0);

        let mut out_pcm = vec![vec![0.0f32; num_out]; channels];
        let rx = stretch.receive_samples(&mut out_pcm).unwrap();
        assert_eq!(rx, num_out);

        for ch_buf in &out_pcm {
            for sample in ch_buf {
                assert!(!sample.is_nan());
                assert!(!sample.is_infinite());
            }
        }
    }

    #[test]
    fn test_spectral_stretch_with_confined_gaussian_window() {
        let channels = 2;
        let sample_rate = 44100;

        let mut stretch = SpectralStretch::builder(channels, sample_rate)
            .preset(SpectralPreset::Custom {
                block_samples: 256,
                interval_samples: 64,
            })
            .window_shape(SpectralWindowShape::ConfinedGaussian)
            .tempo(1.5)
            .pitch_semi_tones(-2.0)
            .build()
            .unwrap();

        let input_pcm = vec![vec![0.1f32; 2048]; channels];
        stretch.put_samples(&input_pcm).unwrap();

        assert!(stretch.num_samples() > 0);

        let mut out_pcm = vec![vec![0.0f32; 256]; channels];
        let rx = stretch.receive_samples(&mut out_pcm).unwrap();
        assert_eq!(rx, 256);

        for ch_buf in &out_pcm {
            for sample in ch_buf {
                assert!(!sample.is_nan());
                assert!(!sample.is_infinite());
            }
        }
    }

    #[test]
    fn test_fractional_hop_alternation_at_tempo_0_5() {
        let channels = 1;
        let sample_rate = 44100;

        let mut stretch = SpectralStretch::builder(channels, sample_rate)
            .tempo(0.5)
            .build()
            .unwrap();

        // 10 seconds of silence to trigger multiple blocks
        let input_pcm = vec![vec![0.0f32; 44100 * 2]; channels];
        stretch.put_samples(&input_pcm).unwrap();

        // With interval = 1323 and tempo = 0.5:
        // accumulated_input_pos should advance precisely by 661.5 per block
        // consumed_input_frames should alternate hops between 662 and 661
        assert!(stretch.num_samples() > 0);
        let blocks = stretch.consumed_input_frames / 661;
        assert!(blocks > 0);
    }
}
