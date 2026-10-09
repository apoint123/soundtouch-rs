use realfft::num_complex::Complex;

use crate::spectral::utils::{
    Xorshift32,
    mul_complex,
    norm_complex,
};

#[derive(Debug, Clone, Default)]
pub struct Band {
    pub input: Complex<f32>,
    pub prev_input: Complex<f32>,
    pub output: Complex<f32>,
    pub input_energy: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct PitchMapPoint {
    pub input_bin: f32,
    pub freq_grad: f32,
}

impl Default for PitchMapPoint {
    fn default() -> Self {
        Self {
            input_bin: 0.0,
            freq_grad: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Peak {
    pub input: f32,
    pub output: f32,
}

#[derive(Debug, Clone, Default)]
pub struct Prediction {
    pub energy: f32,
    pub input: Complex<f32>,
}

impl Prediction {
    pub fn make_output(&self, phase: Complex<f32>, noise_floor: f32) -> Complex<f32> {
        let phase_norm = norm_complex(phase);
        let is_noise = phase_norm <= noise_floor;

        let fallback_norm = norm_complex(self.input) + noise_floor;
        let eff_norm = if is_noise { fallback_norm } else { phase_norm };
        let eff_phase = if is_noise { self.input } else { phase };

        let scale = (self.energy / eff_norm).sqrt();
        eff_phase * scale
    }
}

/// Phase Vocoder algorithm processor handling pitch map, peak finding, and phase locking.
pub struct PhaseVocoder {
    bands: usize,
    channels: usize,
    freq_multiplier: f32,
    freq_tonality_limit: f32,
    formant_multiplier: f32,
    formant_compensation: bool,
    noise_floor: f32,
    prng: Xorshift32,
    channel_bands: Vec<Band>,
    output_map: Vec<PitchMapPoint>,
    channel_predictions: Vec<Prediction>,
    energy: Vec<f32>,
    smoothed_energy: Vec<f32>,
    smooth_energy_state: f32,
    peaks: Vec<Peak>,
    rot_table: Vec<Complex<f32>>,
    cached_interval_samples: usize,
    cached_fft_samples: usize,
}

impl PhaseVocoder {
    pub fn new(bands: usize, channels: usize) -> Self {
        Self {
            bands,
            channels,
            freq_multiplier: 1.0,
            freq_tonality_limit: 1.0,
            formant_multiplier: 1.0,
            formant_compensation: false,
            noise_floor: 1e-15,
            prng: Xorshift32::default(),
            channel_bands: vec![Band::default(); bands * channels],
            output_map: vec![PitchMapPoint::default(); bands],
            channel_predictions: vec![Prediction::default(); bands * channels],
            energy: vec![0.0; bands],
            smoothed_energy: vec![0.0; bands],
            smooth_energy_state: 0.0,
            peaks: Vec::with_capacity(bands),
            rot_table: Vec::with_capacity(bands),
            cached_interval_samples: 0,
            cached_fft_samples: 0,
        }
    }

    /// Clears stream history while retaining parameters and allocated scratch space.
    pub fn clear(&mut self) {
        self.prng = Xorshift32::default();
        self.channel_bands.fill(Band::default());
        self.output_map.fill(PitchMapPoint::default());
        self.channel_predictions.fill(Prediction::default());
        self.energy.fill(0.0);
        self.smoothed_energy.fill(0.0);
        self.smooth_energy_state = 0.0;
        self.peaks.clear();
    }

    pub fn set_transpose_factor(&mut self, multiplier: f32) {
        self.freq_multiplier = multiplier;
        self.freq_tonality_limit = 1.0 / multiplier.sqrt();
    }

    pub const fn set_formant_factor(&mut self, multiplier: f32, compensate_pitch: bool) {
        self.formant_multiplier = multiplier;
        self.formant_compensation = compensate_pitch;
    }

    pub fn band(&self, channel: usize, bin: usize) -> &Band {
        &self.channel_bands[channel * self.bands + bin]
    }

    pub fn band_mut(&mut self, channel: usize, bin: usize) -> &mut Band {
        &mut self.channel_bands[channel * self.bands + bin]
    }

    pub fn process_spectrum(
        &mut self,
        interval_samples: usize,
        fft_samples: usize,
        time_factor: f32,
    ) {
        let smoothing_bins = (fft_samples as f32) / (interval_samples as f32);

        self.ensure_rot_table(interval_samples, fft_samples);
        for ch_bands in self.channel_bands.chunks_exact_mut(self.bands) {
            for (band, &rot) in ch_bands.iter_mut().zip(&self.rot_table) {
                band.output = mul_complex(band.output, rot, false);
                band.prev_input = mul_complex(band.prev_input, rot, false);
            }
        }

        if (self.freq_multiplier - 1.0).abs() > 1e-6 {
            self.update_output_map(fft_samples, smoothing_bins);
        } else {
            for ch_bands in self.channel_bands.chunks_exact_mut(self.bands) {
                for band in ch_bands.iter_mut() {
                    band.input_energy = norm_complex(band.input);
                }
            }
            for (b, map_point) in self.output_map.iter_mut().enumerate() {
                *map_point = PitchMapPoint {
                    input_bin: b as f32,
                    freq_grad: 1.0,
                };
            }
        }

        for (ch_bands, ch_preds) in self
            .channel_bands
            .chunks_exact_mut(self.bands)
            .zip(self.channel_predictions.chunks_exact_mut(self.bands))
        {
            for b in 0..self.bands {
                let map_point = self.output_map[b];
                let (input_complex, prev_input, input_energy) =
                    Self::get_fractional_band(ch_bands, map_point.input_bin);

                let prev_energy = ch_preds[b].energy;
                let new_energy = input_energy * map_point.freq_grad.max(0.0);
                ch_preds[b].energy = new_energy;
                ch_preds[b].input = input_complex;

                let freq_twist = mul_complex(input_complex, prev_input, true);
                let denom = prev_energy.max(new_energy) + self.noise_floor;
                let inv_denom = 1.0 / denom;
                let scaled_twist = freq_twist * inv_denom;
                ch_bands[b].output = mul_complex(ch_bands[b].output, scaled_twist, false);
            }
        }

        let long_vertical_step = (smoothing_bins.round() as usize).max(1);
        let max_clean_stretch: f32 = 2.0;
        let time_factor = time_factor.max(1.0 / max_clean_stretch);
        let random_time_factor = time_factor > max_clean_stretch;
        let min_time_factor = if random_time_factor {
            2.0f32.mul_add(max_clean_stretch, -time_factor)
        } else {
            time_factor
        };

        for b in 0..self.bands {
            let mut max_c = 0;
            let mut max_energy = self.channel_predictions[b].energy;
            for c in 1..self.channels {
                let e = self.channel_predictions[c * self.bands + b].energy;
                if e > max_energy {
                    max_c = c;
                    max_energy = e;
                }
            }

            let phase_sum = self.accumulate_vertical_phase(
                b,
                max_c,
                long_vertical_step,
                time_factor,
                min_time_factor,
                random_time_factor,
            );

            let pred = &self.channel_predictions[max_c * self.bands + b];
            let out_complex = pred.make_output(phase_sum, self.noise_floor);
            self.channel_bands[max_c * self.bands + b].output = out_complex;

            for c in 0..self.channels {
                if c != max_c {
                    let channel_pred = &self.channel_predictions[c * self.bands + b];
                    let twist = mul_complex(channel_pred.input, pred.input, true);
                    let channel_phase = mul_complex(out_complex, twist, false);
                    self.channel_bands[c * self.bands + b].output =
                        channel_pred.make_output(channel_phase, self.noise_floor);
                }
            }
        }
    }

    fn ensure_rot_table(&mut self, interval_samples: usize, fft_samples: usize) {
        if self.cached_interval_samples == interval_samples
            && self.cached_fft_samples == fft_samples
            && self.rot_table.len() == self.bands
        {
            return;
        }
        self.cached_interval_samples = interval_samples;
        self.cached_fft_samples = fft_samples;
        self.rot_table.clear();
        let angle_step =
            2.0 * std::f32::consts::PI * (interval_samples as f32) / (fft_samples as f32);
        for b in 0..self.bands {
            let angle = (b as f32) * angle_step;
            let (sin, cos) = angle.sin_cos();
            self.rot_table.push(Complex::new(cos, sin));
        }
    }

    pub fn map_freq(&self, freq: f32) -> f32 {
        if freq > self.freq_tonality_limit {
            (self.freq_multiplier - 1.0).mul_add(self.freq_tonality_limit, freq)
        } else {
            freq * self.freq_multiplier
        }
    }

    pub fn smooth_energy(&mut self, smoothing_bins: f32) {
        let smoothing_slew = 1.0 / smoothing_bins.mul_add(0.5, 1.0);

        self.energy.fill(0.0);

        for ch_bands in self.channel_bands.chunks_exact_mut(self.bands) {
            for (band, e_total) in ch_bands.iter_mut().zip(&mut self.energy) {
                let e = norm_complex(band.input);
                band.input_energy = e;
                *e_total += e;
            }
        }

        self.smoothed_energy.copy_from_slice(&self.energy);

        let mut e = self.smooth_energy_state;
        for b in (0..self.bands).rev() {
            e = (self.smoothed_energy[b] - e).mul_add(smoothing_slew, e);
            self.smoothed_energy[b] = e;
        }
        for b in 0..self.bands {
            e = (self.smoothed_energy[b] - e).mul_add(smoothing_slew, e);
            self.smoothed_energy[b] = e;
        }
        self.smooth_energy_state = e;
    }

    pub fn find_peaks(&mut self, fft_samples: usize) {
        self.peaks.clear();

        let mut start = 0;
        while start < self.bands {
            if self.energy[start] > self.smoothed_energy[start] {
                let mut end = start;
                let mut band_sum = 0.0f32;
                let mut energy_sum = 0.0f32;
                while end < self.bands && self.energy[end] > self.smoothed_energy[end] {
                    band_sum = (end as f32).mul_add(self.energy[end], band_sum);
                    energy_sum += self.energy[end];
                    end += 1;
                }
                if energy_sum > 0.0 {
                    let avg_band = band_sum / energy_sum;
                    let avg_freq = avg_band / (fft_samples as f32);
                    let out_freq = self.map_freq(avg_freq);
                    let out_band = out_freq * (fft_samples as f32);
                    self.peaks.push(Peak {
                        input: avg_band,
                        output: out_band,
                    });
                }
                start = end;
            }
            start += 1;
        }
    }

    pub fn update_output_map(&mut self, fft_samples: usize, smoothing_bins: f32) {
        self.smooth_energy(smoothing_bins);
        self.find_peaks(fft_samples);

        if self.peaks.is_empty() {
            for b in 0..self.bands {
                self.output_map[b] = PitchMapPoint {
                    input_bin: (b as f32) / self.freq_multiplier,
                    freq_grad: 1.0 / self.freq_multiplier,
                };
            }
            return;
        }

        let first_peak = self.peaks[0];
        let bottom_offset = first_peak.input - first_peak.output;
        let limit0 = self.bands.min(first_peak.output.ceil() as usize);
        for b in 0..limit0 {
            self.output_map[b] = PitchMapPoint {
                input_bin: b as f32 + bottom_offset,
                freq_grad: 1.0,
            };
        }

        for p in 1..self.peaks.len() {
            let prev = self.peaks[p - 1];
            let next = self.peaks[p];

            let delta_out = next.output - prev.output;
            if delta_out <= 1e-6 {
                continue;
            }
            let range_scale = 1.0 / delta_out;
            let out_offset = prev.input - prev.output;
            let out_scale = next.input - next.output - prev.input + prev.output;
            let grad_scale = out_scale * range_scale;

            let start_bin = (prev.output.ceil() as usize).min(self.bands);
            let end_bin = (next.output.ceil() as usize).min(self.bands);

            for b in start_bin..end_bin {
                let r = (b as f32 - prev.output) * range_scale;
                let h = r * r * 2.0f32.mul_add(-r, 3.0);
                let out_b = h.mul_add(out_scale, b as f32 + out_offset);

                let grad_h = 6.0 * r * (1.0 - r);
                let grad_b = grad_h.mul_add(grad_scale, 1.0);

                self.output_map[b] = PitchMapPoint {
                    input_bin: out_b,
                    freq_grad: grad_b,
                };
            }
        }

        let last_peak = self.peaks[self.peaks.len() - 1];
        let top_offset = last_peak.input - last_peak.output;
        let start_top = (last_peak.output.max(0.0) as usize).min(self.bands);
        for b in start_top..self.bands {
            self.output_map[b] = PitchMapPoint {
                input_bin: b as f32 + top_offset,
                freq_grad: 1.0,
            };
        }
    }

    fn get_slice_fractional_input(channel_bands: &[Band], input_bin: f32) -> Complex<f32> {
        let low_idx = input_bin.floor() as isize;
        let frac = input_bin - (low_idx as f32);
        let len = channel_bands.len().cast_signed();

        if low_idx >= 0 && low_idx + 1 < len {
            let low = channel_bands[low_idx as usize].input;
            let high = channel_bands[(low_idx + 1) as usize].input;
            low + (high - low) * frac
        } else if low_idx == -1 {
            channel_bands[0].input * frac
        } else if low_idx == len - 1 {
            channel_bands[(len - 1) as usize].input * (1.0 - frac)
        } else {
            Complex::new(0.0, 0.0)
        }
    }

    fn get_fractional_band(
        channel_bands: &[Band],
        input_bin: f32,
    ) -> (Complex<f32>, Complex<f32>, f32) {
        let low_idx = input_bin.floor() as isize;
        let frac = input_bin - (low_idx as f32);
        let len = channel_bands.len().cast_signed();

        if low_idx >= 0 && low_idx + 1 < len {
            let b0 = &channel_bands[low_idx as usize];
            let b1 = &channel_bands[(low_idx + 1) as usize];
            let input = b0.input + (b1.input - b0.input) * frac;
            let prev_input = b0.prev_input + (b1.prev_input - b0.prev_input) * frac;
            let energy = (b1.input_energy - b0.input_energy).mul_add(frac, b0.input_energy);
            (input, prev_input, energy)
        } else if low_idx == -1 {
            let b1 = &channel_bands[0];
            (
                b1.input * frac,
                b1.prev_input * frac,
                b1.input_energy * frac,
            )
        } else if low_idx == len - 1 {
            let b0 = &channel_bands[(len - 1) as usize];
            let inv_frac = 1.0 - frac;
            (
                b0.input * inv_frac,
                b0.prev_input * inv_frac,
                b0.input_energy * inv_frac,
            )
        } else {
            (Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), 0.0)
        }
    }

    fn accumulate_vertical_phase(
        &mut self,
        b: usize,
        max_c: usize,
        long_vertical_step: usize,
        time_factor: f32,
        min_time_factor: f32,
        random_time_factor: bool,
    ) -> Complex<f32> {
        let max_c_bands = &self.channel_bands[max_c * self.bands..(max_c + 1) * self.bands];
        let map_point = self.output_map[b];
        let mut phase_sum = Complex::new(0.0, 0.0);

        if b > 0 {
            let bin_time_factor = if random_time_factor {
                self.prng.next_range(min_time_factor, time_factor)
            } else {
                time_factor
            };
            let down_bin = b - 1;
            let down_input = Self::get_slice_fractional_input(
                max_c_bands,
                map_point.input_bin - bin_time_factor,
            );
            let short_twist = mul_complex(
                self.channel_predictions[max_c * self.bands + b].input,
                down_input,
                true,
            );
            let down_output = max_c_bands[down_bin].output;
            phase_sum += mul_complex(down_output, short_twist, false);

            if b >= long_vertical_step {
                let long_down_bin = b - long_vertical_step;
                let long_down_input = Self::get_slice_fractional_input(
                    max_c_bands,
                    (long_vertical_step as f32).mul_add(-bin_time_factor, map_point.input_bin),
                );
                let long_twist = mul_complex(
                    self.channel_predictions[max_c * self.bands + b].input,
                    long_down_input,
                    true,
                );
                let long_down_output = max_c_bands[long_down_bin].output;
                phase_sum += mul_complex(long_down_output, long_twist, false);
            }
        }

        if b + 1 < self.bands {
            let bin_time_factor = if random_time_factor {
                self.prng.next_range(min_time_factor, time_factor)
            } else {
                time_factor
            };
            let up_bin = b + 1;
            let up_map_point = self.output_map[up_bin];
            let down_input = Self::get_slice_fractional_input(
                max_c_bands,
                up_map_point.input_bin - bin_time_factor,
            );
            let short_twist = mul_complex(
                self.channel_predictions[max_c * self.bands + up_bin].input,
                down_input,
                true,
            );
            let up_output = max_c_bands[up_bin].output;
            phase_sum += mul_complex(up_output, short_twist, true);

            if b + long_vertical_step < self.bands {
                let long_up_bin = b + long_vertical_step;
                let long_up_map_point = self.output_map[long_up_bin];
                let long_down_input = Self::get_slice_fractional_input(
                    max_c_bands,
                    (long_vertical_step as f32)
                        .mul_add(-bin_time_factor, long_up_map_point.input_bin),
                );
                let long_twist = mul_complex(
                    self.channel_predictions[max_c * self.bands + long_up_bin].input,
                    long_down_input,
                    true,
                );
                let long_up_output = max_c_bands[long_up_bin].output;
                phase_sum += mul_complex(long_up_output, long_twist, true);
            }
        }

        phase_sum
    }

    #[cfg(test)]
    pub fn get_fractional_input(&self, channel: usize, input_bin: f32) -> Complex<f32> {
        let start = channel * self.bands;
        Self::get_slice_fractional_input(&self.channel_bands[start..start + self.bands], input_bin)
    }

    #[cfg(test)]
    pub fn get_fractional_input_energy(&self, channel: usize, input_bin: f32) -> f32 {
        let start = channel * self.bands;
        Self::get_fractional_band(&self.channel_bands[start..start + self.bands], input_bin).2
    }

    #[cfg(test)]
    pub fn get_fractional_prev_input(&self, channel: usize, input_bin: f32) -> Complex<f32> {
        let start = channel * self.bands;
        Self::get_fractional_band(&self.channel_bands[start..start + self.bands], input_bin).1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xorshift32_reproducibility() {
        let mut rng1 = Xorshift32::new(12345);
        let mut rng2 = Xorshift32::new(12345);

        for _ in 0..100 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn test_fractional_zero_padding() {
        let bands = 64;
        let channels = 1;
        let mut vocoder = PhaseVocoder::new(bands, channels);

        vocoder.band_mut(0, 0).input = Complex::new(3.0, 4.0);
        vocoder.band_mut(0, 0).input_energy = 25.0;
        vocoder.band_mut(0, bands - 1).input = Complex::new(1.0, 2.0);
        vocoder.band_mut(0, 0).prev_input = Complex::new(3.0, 4.0);
        vocoder.band_mut(0, bands - 1).prev_input = Complex::new(1.0, 2.0);

        let neg_far = vocoder.get_fractional_input(0, -5.0);
        assert_eq!(neg_far, Complex::new(0.0, 0.0));
        assert_eq!(vocoder.get_fractional_input_energy(0, -5.0), 0.0);
        assert_eq!(
            vocoder.get_fractional_prev_input(0, -5.0),
            Complex::new(0.0, 0.0)
        );

        let neg_half = vocoder.get_fractional_input(0, -0.5);
        assert!((neg_half.re - 1.5).abs() < 1e-6);
        assert!((neg_half.im - 2.0).abs() < 1e-6);

        let pos_far = vocoder.get_fractional_input(0, (bands + 5) as f32);
        assert_eq!(pos_far, Complex::new(0.0, 0.0));

        let pos_half = vocoder.get_fractional_input(0, (bands - 1) as f32 + 0.5);
        assert!((pos_half.re - 0.5).abs() < 1e-6);
        assert!((pos_half.im - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_peak_finding_multi_tone() {
        let bands = 128;
        let channels = 1;
        let mut vocoder = PhaseVocoder::new(bands, channels);

        vocoder.band_mut(0, 20).input = Complex::new(10.0, 0.0);
        vocoder.band_mut(0, 50).input = Complex::new(8.0, 0.0);
        vocoder.band_mut(0, 90).input = Complex::new(6.0, 0.0);

        vocoder.set_transpose_factor(1.5); // +7 semitones
        vocoder.update_output_map(256, 4.0);

        assert!(
            vocoder.peaks.len() >= 3,
            "Must identify at least 3 peaks for injected multi-tone chord"
        );
    }

    #[test]
    fn test_high_stretch_phase_randomization_stability() {
        let bands = 128;
        let channels = 2;
        let mut vocoder = PhaseVocoder::new(bands, channels);

        for c in 0..channels {
            for b in 0..bands {
                vocoder.band_mut(c, b).input = Complex::new(1.0, 0.5);
                vocoder.band_mut(c, b).prev_input = Complex::new(0.9, 0.4);
            }
        }

        vocoder.process_spectrum(32, 128, 5.0);

        for c in 0..channels {
            for b in 0..bands {
                let out = vocoder.band(c, b).output;
                assert!(!out.re.is_nan());
                assert!(!out.im.is_nan());
            }
        }
    }
}
